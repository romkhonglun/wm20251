import argparse
import polars as pl
import numpy as np
import os
import gc  # Garbage collector
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Split embedded parquet file into numpy arrays for training.")

    parser.add_argument(
        "--input_path",
        type=str,
        default="/kaggle/working/small_articles_embedded.parquet",
        help="Path to the embedded articles parquet file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kaggle/working/processed_data",
        help="Directory to save the processed .npy files."
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=1024,
        help="Dimension of the embedding vectors (e.g., 1024 for e5-large, 768 for bert-base)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Reading Parquet from {args.input_path} ---")
    if not os.path.exists(args.input_path):
        print(f"Error: File not found at {args.input_path}")
        sys.exit(1)

    try:
        # Đọc file parquet
        df = pl.read_parquet(args.input_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)

    # Tạo hàng Padding (Index 0)
    print(f"Creating padding row (vector size: {args.vector_size})...")

    # Chỉ chọn các cột cần thiết để đảm bảo schema khớp
    target_cols = ["article_id", "title_vector", "body_vector", "category_vector"]
    df = df.select(target_cols)

    pad_row = pl.DataFrame({
        "article_id": [-1],
        "title_vector": [np.zeros(args.vector_size, dtype=np.float32)],
        "body_vector": [np.zeros(args.vector_size, dtype=np.float32)],
        "category_vector": [np.zeros(args.vector_size, dtype=np.float32)]
    })

    # Cast schema để tránh lỗi type mismatch (quan trọng khi merge)
    pad_row = pad_row.cast(df.schema)
    df = pl.concat([pad_row, df], how="vertical")

    print(f"Total rows to process: {len(df)}")

    # --- 1. SAVE IDs ---
    print("Saving IDs...")
    ids_path = os.path.join(args.output_dir, "article_ids.npy")
    ids = np.array(df["article_id"].to_list())
    np.save(ids_path, ids)
    del ids
    gc.collect()  # Dọn RAM ngay

    # --- 2. SAVE TITLE (Xử lý xong xóa cột ngay) ---
    print("Saving Title Vectors...")
    title_path = os.path.join(args.output_dir, "title_emb.npy")
    # Lấy numpy array từ cột
    title_matrix = np.stack(df["title_vector"].to_numpy())
    # Xóa cột trong dataframe để giải phóng RAM cho bước sau
    df = df.drop("title_vector")
    np.save(title_path, title_matrix)
    del title_matrix
    gc.collect()

    # --- 3. SAVE BODY ---
    print("Saving Body Vectors...")
    body_path = os.path.join(args.output_dir, "body_emb.npy")
    body_matrix = np.stack(df["body_vector"].to_numpy())
    df = df.drop("body_vector")
    np.save(body_path, body_matrix)
    del body_matrix
    gc.collect()

    # --- 4. SAVE CATEGORY ---
    print("Saving Category Vectors...")
    cat_path = os.path.join(args.output_dir, "cat_emb.npy")
    cat_matrix = np.stack(df["category_vector"].to_numpy())
    df = df.drop("category_vector")
    np.save(cat_path, cat_matrix)
    del cat_matrix
    gc.collect()

    print(f"✅ ALL DONE! Files saved to {args.output_dir}")


if __name__ == "__main__":
    main()