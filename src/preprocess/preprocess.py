import polars as pl
import numpy as np
from pathlib import Path
import pickle
import gc

# ==========================================
# CẤU HÌNH
# ==========================================
RAW_DIR = Path("/home2/congnh/wm/input/ebnerd_large")
PROCESSED_DIR = Path("/home2/congnh/wm/processed_parquet_large")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_mappings():
    print("⏳ [1/4] Building ID Mappings (Article & User)...")

    # --- 1. ARTICLE MAPPING ---
    # Scan file articles để lấy list ID
    df_art = pl.scan_parquet(RAW_DIR / "articles.parquet")
    # Lấy cột article_id, unique và convert sang list
    art_ids = df_art.select("article_id").collect().to_series().unique().to_list()

    # Map: Index 0 dành cho padding/unknown -> Start từ 1
    aid2idx = {aid: i + 1 for i, aid in enumerate(art_ids)}

    with open(PROCESSED_DIR / "aid2idx.pkl", "wb") as f:
        pickle.dump(aid2idx, f)
    print(f"   ✅ Mapped {len(aid2idx)} articles.")

    # --- 2. USER MAPPING ---
    # Phải quét cả Train và Validation để lấy đủ User ID
    uids = set()
    for split in ["train", "validation"]:
        path = RAW_DIR / split / "behaviors.parquet"
        if path.exists():
            # Chỉ đọc cột user_id để tiết kiệm RAM
            curr_uids = pl.read_parquet(path, columns=["user_id"])["user_id"].unique().to_list()
            uids.update(curr_uids)

    # Map: User Index cũng start từ 0 (vì dùng làm index mảng Numpy)
    uid2idx = {uid: i for i, uid in enumerate(uids)}

    with open(PROCESSED_DIR / "uid2idx.pkl", "wb") as f:
        pickle.dump(uid2idx, f)
    print(f"   ✅ Mapped {len(uid2idx)} users.")

    return aid2idx, uid2idx


def preprocess_articles(aid2idx):
    print("⏳ [2/4] Processing Articles Features...")
    df = pl.scan_parquet(RAW_DIR / "articles.parquet")

    # Logic an toàn cho Timestamp (dù input là string hay datetime đều convert về giây)
    # Datetime trong Polars thường là us (microsecond).
    # Cách an toàn nhất: cast Datetime -> cast Int64 (ra micros) -> chia 1e6 -> ra seconds

    df_processed = df.select([
        pl.col("article_id").map_elements(lambda x: aid2idx.get(x, 0), return_dtype=pl.Int32).alias("id_encoded"),

        # Categorical Encoding
        pl.col("category").cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Int32).alias("cat_encoded"),

        # Numerical Features
        np.log1p(pl.col("total_pageviews").fill_null(0)).cast(pl.Float32).alias("norm_views"),
        np.log1p(pl.col("total_inviews").fill_null(0)).cast(pl.Float32).alias("norm_inviews"),
        pl.col("sentiment_score").fill_null(0).cast(pl.Float32),
        np.log1p(pl.col("total_read_time").fill_null(0)).cast(pl.Float32).alias("norm_read_time"),

        # Time: Cast strict to microseconds then to seconds
        (pl.col("published_time").cast(pl.Datetime("us")).cast(pl.Int64) / 1_000_000.0).cast(pl.Float32).alias(
            "published_time")
    ])

    df_processed.sink_parquet(PROCESSED_DIR / "articles_processed.parquet")
    print("   ✅ Articles saved.")


def preprocess_history(aid2idx, uid2idx, split):
    input_path = RAW_DIR / split / "history.parquet"
    if not input_path.exists(): return

    print(f"⏳ [3/4] Processing History ({split})...")
    output_dir = PROCESSED_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.scan_parquet(input_path)

    # Hàm map list (Vẫn dùng map_elements cho đơn giản, chấp nhận chậm một chút)
    def map_ids(id_list):
        if id_list is None: return []
        return [aid2idx.get(i, 0) for i in id_list]

    df_processed = df.select([
        # [FIX] Map User ID sang Int
        pl.col("user_id").map_elements(lambda x: uid2idx.get(x, 0), return_dtype=pl.Int32),

        pl.col("article_id_fixed").map_elements(map_ids, return_dtype=pl.List(pl.Int32)).alias("hist_ids"),
        (pl.col("scroll_percentage_fixed").list.eval(pl.element() / 100.0).cast(pl.List(pl.Float32))).alias(
            "hist_scroll"),
        (pl.col("read_time_fixed").list.eval(pl.element()).cast(pl.List(pl.Float32))).alias("hist_time"),
        # Log time luôn ở đây

        # [FIX] Time safe conversion
        pl.col("impression_time_fixed").list.eval(
            (pl.element().cast(pl.Int64) / 1_000_000.0).cast(pl.Float32)  # Giả sử input gốc là micros như code cũ
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

    def map_ids(id_list):
        if id_list is None: return []
        return [aid2idx.get(i, 0) for i in id_list]

    df_processed = df.select([
        pl.col("impression_id").cast(pl.UInt32),

        # [FIX] Map User ID sang Int
        pl.col("user_id").map_elements(lambda x: uid2idx.get(x, 0), return_dtype=pl.Int32),

        # [FIX] Time safe conversion
        (pl.col("impression_time").cast(pl.Datetime("us")).cast(pl.Int64) / 1_000_000.0).cast(pl.Float32).alias(
            "imp_ts"),

        pl.col("article_ids_inview").map_elements(map_ids, return_dtype=pl.List(pl.Int32)).alias("inv_ids"),
        pl.col("article_ids_clicked").map_elements(map_ids, return_dtype=pl.List(pl.Int32)).alias("clk_ids"),

        # User Metadata
        pl.col("device_type").fill_null(0).cast(pl.Int8),
        (pl.col("age").fill_null(0) / 100.0).cast(pl.Float32).alias("norm_age"),
        pl.col("gender").fill_null(0).cast(pl.Int8),
        pl.col("is_subscriber").fill_null(False).cast(pl.Int8),
        pl.col("is_sso_user").fill_null(False).cast(pl.Int8)
    ])

    df_processed.sink_parquet(output_dir / "behaviors_processed.parquet")
    print(f"   ✅ Behaviors {split} saved.")


if __name__ == "__main__":
    aid2idx, uid2idx = build_mappings()
    preprocess_articles(aid2idx)
    for split in ["train", "validation"]:
        preprocess_history(aid2idx, uid2idx, split)
        preprocess_behaviors(aid2idx, uid2idx, split)

    print("\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")