import polars as pl
import numpy as np
from pathlib import Path
import pickle
import gc
import os

# ==========================================
# CẤU HÌNH
# ==========================================
# [LƯU Ý] Thay đổi đường dẫn phù hợp với máy của bạn
RAW_DIR = Path("/home2/congnh/wm/input/ebnerd_large")
PROCESSED_DIR = Path("/home2/congnh/wm/processed_parquet_large")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Tắt warning của Polars về string conversion (nếu có)
os.environ["POLARS_MAX_THREADS"] = "16"


def build_mappings():
    """
    Tạo mapping ID và trả về cả DataFrame (để Join nhanh) và Dict (để Map list)
    """
    print("⏳ [1/4] Building ID Mappings...")

    # --- 1. ARTICLE MAPPING ---
    print("   ...Scanning Articles")
    df_art = pl.scan_parquet(RAW_DIR / "articles.parquet")

    # Lấy unique ID và tạo index bắt đầu từ 1 (0 dành cho padding/unknown)
    df_aid_map = (
        df_art.select("article_id")
        .unique()
        .collect()  # Collect về RAM để đánh index
        .with_row_index("aid", offset=1)  # article_id -> aid (int)
        .select(["article_id", pl.col("aid").cast(pl.Int32)])  # Cast về Int32 cho nhẹ
    )

    # Tạo Dictionary phụ trợ cho các hàm xử lý List (bắt buộc dùng map_elements)
    # zip() nhanh hơn to_dict()
    aid2idx = dict(zip(df_aid_map["article_id"].to_list(), df_aid_map["aid"].to_list()))

    with open(PROCESSED_DIR / "aid2idx.pkl", "wb") as f:
        pickle.dump(aid2idx, f)
    print(f"   ✅ Mapped {len(aid2idx)} articles.")

    # --- 2. USER MAPPING ---
    print("   ...Scanning Users (Train & Val)")
    # Scan lazy cả 2 file
    q_train = pl.scan_parquet(RAW_DIR / "train/behaviors.parquet").select("user_id")
    q_val = pl.scan_parquet(RAW_DIR / "validation/behaviors.parquet").select("user_id")

    # Concat -> Unique -> Index (User start từ 0)
    df_uid_map = (
        pl.concat([q_train, q_val])
        .unique()
        .collect()
        .with_row_index("uid", offset=0)
        .select(["user_id", pl.col("uid").cast(pl.Int32)])
    )

    uid2idx = dict(zip(df_uid_map["user_id"].to_list(), df_uid_map["uid"].to_list()))

    with open(PROCESSED_DIR / "uid2idx.pkl", "wb") as f:
        pickle.dump(uid2idx, f)
    print(f"   ✅ Mapped {len(uid2idx)} users.")

    return df_aid_map, df_uid_map, aid2idx, uid2idx


def preprocess_articles(df_aid_map):
    print("⏳ [2/4] Processing Articles Features...")
    df = pl.scan_parquet(RAW_DIR / "articles.parquet")

    # [OPTIMIZATION] Dùng JOIN thay vì map_elements -> Tốc độ cực nhanh
    # df_aid_map đang ở RAM, cần lazy() để join với scan_parquet
    df = df.join(df_aid_map.lazy(), on="article_id", how="left")

    df_processed = df.select([
        # ID đã join được (nếu null thì fill 0)
        pl.col("aid").fill_null(0).alias("id_encoded"),

        # Category: String -> Categorical -> Physical Int
        pl.col("category").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("cat_encoded"),

        # Numerical Features (Log normalize)
        np.log1p(pl.col("total_pageviews").fill_null(0)).cast(pl.Float32).alias("norm_views"),
        np.log1p(pl.col("total_inviews").fill_null(0)).cast(pl.Float32).alias("norm_inviews"),
        pl.col("sentiment_score").fill_null(0).cast(pl.Float32),
        np.log1p(pl.col("total_read_time").fill_null(0)).cast(pl.Float32).alias("norm_read_time"),

        # [OPTIMIZATION] Đếm khoảng trắng + 1 nhanh hơn split list
        (pl.col("body").str.count_matches(" ") + 1).fill_null(0).cast(pl.Float32).alias("body_len"),

        # Time conversion
        (pl.col("published_time").cast(pl.Int64) / 1_000_000.0).cast(pl.Float32).alias("published_time")
    ])

    df_processed.sink_parquet(PROCESSED_DIR / "articles_processed.parquet")
    print("   ✅ Articles saved.")


def preprocess_history(df_uid_map, aid2idx, split):
    input_path = RAW_DIR / split / "history.parquet"
    if not input_path.exists(): return

    print(f"⏳ [3/4] Processing History ({split})...")
    output_dir = PROCESSED_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.scan_parquet(input_path)

    # [OPTIMIZATION] Join User ID
    df = df.join(df_uid_map.lazy(), on="user_id", how="left")

    # Hàm map list (Vẫn giữ map_elements vì xử lý list join phức tạp hơn nhiều)
    # Dùng biến global aid2idx bên trong function để tránh pickling overhead nếu có thể
    def map_ids(id_list):
        if id_list is None: return []
        return [aid2idx.get(i, 0) for i in id_list]

    df_processed = df.select([
        # User ID từ phép join
        pl.col("uid").fill_null(0).alias("user_id"),

        # List mapping
        pl.col("article_id_fixed").map_elements(map_ids, return_dtype=pl.List(pl.Int32)).alias("hist_ids"),

        # Normalize scroll & time
        (pl.col("scroll_percentage_fixed").list.eval(pl.element() / 100.0).cast(pl.List(pl.Float32))).alias(
            "hist_scroll"),
        (pl.col("read_time_fixed").cast(pl.List(pl.Float32))).alias("hist_time"),

        # Time safe conversion
        pl.col("impression_time_fixed").list.eval(
            (pl.element().cast(pl.Int64) / 1_000_000.0).cast(pl.Float32)
        ).alias("hist_ts")
    ])

    df_processed.sink_parquet(output_dir / "history_processed.parquet")
    print(f"   ✅ History {split} saved.")


def preprocess_behaviors(df_uid_map, aid2idx, split):
    input_path = RAW_DIR / split / "behaviors.parquet"
    if not input_path.exists(): return

    print(f"⏳ [4/4] Processing Behaviors ({split})...")
    output_dir = PROCESSED_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.scan_parquet(input_path)

    # [OPTIMIZATION] Join User ID
    df = df.join(df_uid_map.lazy(), on="user_id", how="left")

    def map_ids(id_list):
        if id_list is None: return []
        return [aid2idx.get(i, 0) for i in id_list]

    df_processed = df.select([
        pl.col("impression_id").cast(pl.UInt32),

        # User ID từ phép join
        pl.col("uid").fill_null(0).alias("user_id"),

        # Timestamp conversion
        (pl.col("impression_time").cast(pl.Datetime("us")).cast(pl.Int64) / 1_000_000.0).cast(pl.Float32).alias(
            "imp_ts"),

        # List mapping
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
    # 1. Build Mappings (Trả về cả DF để join và Dict để map list)
    df_aid_map, df_uid_map, aid2idx, uid2idx = build_mappings()

    # 2. Process Articles
    preprocess_articles(df_aid_map)

    # 3. Process History & Behaviors
    for split in ["train", "validation"]:
        preprocess_history(df_uid_map, aid2idx, split)
        preprocess_behaviors(df_uid_map, aid2idx, split)
        gc.collect()  # Dọn dẹp RAM sau mỗi split

    print("\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")