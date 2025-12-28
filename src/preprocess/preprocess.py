import polars as pl
import numpy as np
from pathlib import Path
import pickle
import gc

# ==========================================
# CẤU HÌNH
# ==========================================
RAW_DIR = Path("/home2/congnh/wm/input/ebnerd_testset")
PROCESSED_DIR = Path("/home2/congnh/wm/processed_parquet_test")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_mappings():
    print("⏳ [1/4] Building ID Mappings (Article & User)...")

    # --- 1. ARTICLE MAPPING ---
    df_art = pl.scan_parquet(RAW_DIR / "articles.parquet")
    art_ids = df_art.select("article_id").collect().to_series().unique().to_list()

    # Map: Index 0 dành cho padding/unknown -> Start từ 1
    aid2idx = {aid: i + 1 for i, aid in enumerate(art_ids)}

    with open(PROCESSED_DIR / "aid2idx.pkl", "wb") as f:
        pickle.dump(aid2idx, f)
    print(f"   ✅ Mapped {len(aid2idx)} articles.")

    # --- 2. USER MAPPING ---
    uids = set()
    # Quét Train và Validation để lấy User ID gốc
    for split in ["train", "validation","test"]:
        path = RAW_DIR / split / "behaviors.parquet"
        if path.exists():
            curr_uids = pl.read_parquet(path, columns=["user_id"])["user_id"].unique().to_list()
            uids.update(curr_uids)

    # Map: User Index start từ 0
    uid2idx = {uid: i for i, uid in enumerate(uids)}

    with open(PROCESSED_DIR / "uid2idx.pkl", "wb") as f:
        pickle.dump(uid2idx, f)
    print(f"   ✅ Mapped {len(uid2idx)} users.")

    return aid2idx, uid2idx


def preprocess_articles(aid2idx):
    print("⏳ [2/4] Processing Articles Features...")
    df = pl.scan_parquet(RAW_DIR / "articles.parquet")

    df_processed = df.select([
        # Giữ nguyên map_elements như yêu cầu
        pl.col("article_id").map_elements(lambda x: aid2idx.get(x, 0), return_dtype=pl.Int32).alias("id_encoded"),

        pl.col("category").cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Int32).alias("cat_encoded"),

        np.log1p(pl.col("total_pageviews").fill_null(0)).cast(pl.Float32).alias("norm_views"),
        np.log1p(pl.col("total_inviews").fill_null(0)).cast(pl.Float32).alias("norm_inviews"),
        pl.col("sentiment_score").fill_null(0).cast(pl.Float32),
        np.log1p(pl.col("total_read_time").fill_null(0)).cast(pl.Float32).alias("norm_read_time"),

        (pl.col("published_time").cast(pl.Datetime("us")).cast(pl.Int64) / 1_000_000.0).cast(pl.Float32).alias(
            "published_time")
    ])

    df_processed.sink_parquet(PROCESSED_DIR / "articles_processed.parquet")
    print("   ✅ Articles saved.")


def preprocess_history(aid2idx, uid2idx, split):
    input_path = RAW_DIR / split / "history.parquet"
    if not input_path.exists():
        return

    print(f"⏳ [3/4] Processing History ({split})...")
    output_dir = PROCESSED_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.scan_parquet(input_path)

    def map_ids(id_list):
        if id_list is None: return []
        return [aid2idx.get(i, 0) for i in id_list]

    df_processed = df.select([
        pl.col("user_id").map_elements(lambda x: uid2idx.get(x, 0), return_dtype=pl.Int32),

        pl.col("article_id_fixed").map_elements(map_ids, return_dtype=pl.List(pl.Int32)).alias("hist_ids"),
        (pl.col("scroll_percentage_fixed").list.eval(pl.element() / 100.0).cast(pl.List(pl.Float32))).alias(
            "hist_scroll"),
        (pl.col("read_time_fixed").list.eval(pl.element()).cast(pl.List(pl.Float32))).alias("hist_time"),

        pl.col("impression_time_fixed").list.eval(
            (pl.element().cast(pl.Int64) / 1_000_000.0).cast(pl.Float32)
        ).alias("hist_ts")
    ])

    df_processed.sink_parquet(output_dir / "history_processed.parquet")
    print(f"   ✅ History {split} saved.")


def preprocess_behaviors(aid2idx, uid2idx, split):
    input_path = RAW_DIR / split / "behaviors.parquet"
    if not input_path.exists(): return

    print(f"⏳ [4/4] Processing Behaviors ({split})...")
    output_dir = PROCESSED_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.scan_parquet(input_path)

    # Lấy danh sách cột để kiểm tra xem có cột clicked hay không
    current_columns = df.collect_schema().names()

    def map_ids(id_list):
        if id_list is None: return []
        return [aid2idx.get(i, 0) for i in id_list]

    # Các cột mặc định luôn có
    cols_to_select = [
        pl.col("impression_id").cast(pl.UInt32),

        pl.col("user_id").map_elements(lambda x: uid2idx.get(x, 0), return_dtype=pl.Int32),

        (pl.col("impression_time").cast(pl.Datetime("us")).cast(pl.Int64) / 1_000_000.0).cast(pl.Float32).alias(
            "imp_ts"),

        pl.col("article_ids_inview").map_elements(map_ids, return_dtype=pl.List(pl.Int32)).alias("inv_ids"),

        pl.col("device_type").fill_null(0).cast(pl.Int8),
        (pl.col("age").fill_null(0) / 100.0).cast(pl.Float32).alias("norm_age"),
        pl.col("gender").fill_null(0).cast(pl.Int8),
        pl.col("is_subscriber").fill_null(False).cast(pl.Int8),
        pl.col("is_sso_user").fill_null(False).cast(pl.Int8)
    ]

    # [LOGIC FIX] Nếu là tập Test (không có article_ids_clicked), tạo cột rỗng
    if "article_ids_clicked" in current_columns:
        cols_to_select.append(
            pl.col("article_ids_clicked").map_elements(map_ids, return_dtype=pl.List(pl.Int32)).alias("clk_ids")
        )
    else:
        # Tạo cột clk_ids rỗng để code Dataset không bị lỗi
        cols_to_select.append(
            pl.lit([], dtype=pl.List(pl.Int32)).alias("clk_ids")
        )

    # Thêm session_id nếu tồn tại trong schema
    if "session_id" in current_columns:
        cols_to_select.append(pl.col("session_id").cast(pl.UInt32))

    df_processed = df.select(cols_to_select)
    df_processed.sink_parquet(output_dir / "behaviors_processed.parquet")
    print(f"   ✅ Behaviors {split} saved.")


if __name__ == "__main__":
    aid2idx, uid2idx = build_mappings()
    preprocess_articles(aid2idx)

    # Chạy đủ 3 tập
    for split in ["train", "validation", "test"]:
        preprocess_history(aid2idx, uid2idx, split)
        preprocess_behaviors(aid2idx, uid2idx, split)

    print("\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")