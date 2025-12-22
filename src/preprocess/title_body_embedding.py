import argparse
import polars as pl
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for news articles (Title, Body) and merge with Category embeddings.")

    parser.add_argument(
        "--articles_path",
        type=str,
        default="/kaggle/input/ebnerd-small/articles.parquet",
        help="Path to the source articles parquet file."
    )
    parser.add_argument(
        "--cat_emb_path",
        type=str,
        default="/kaggle/working/small_category_embeddings.pt",
        help="Path to the category embeddings .pt file generated in the previous step."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/kaggle/working/small_articles_embedded.parquet",
        help="Path to save the final embedded articles parquet file."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="HuggingFace model ID for embedding generation."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation."
    )
    parser.add_argument(
        "--max_body_chars",
        type=int,
        default=1000,
        help="Max characters to read from body text (for speed optimization)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. SETUP THIẾT BỊ & MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model {args.model_id}...")
    try:
        model = SentenceTransformer(args.model_id, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 2. LOAD CATEGORY EMBEDDINGS (Đã tạo ở bước trước)
    if not os.path.exists(args.cat_emb_path):
        raise FileNotFoundError(f"Không tìm thấy file {args.cat_emb_path}. Hãy chạy bước trước đó!")

    print(f"Loading category embeddings from {args.cat_emb_path}...")
    cat_data = torch.load(args.cat_emb_path, map_location="cpu", weights_only=False)

    # Chuyển đổi mapping & vector sang dạng dễ dùng với Polars
    cat2id = cat_data['cat2id']
    cat_vectors = cat_data['embeddings'].numpy()  # Chuyển sang numpy

    # Tạo DataFrame lookup cho Category: [category_str, cat_vector]
    # Đảo ngược cat2id để lấy list tên theo đúng thứ tự index
    id2cat = {v: k for k, v in cat2id.items()}
    ordered_cats = [id2cat[i] for i in range(len(id2cat))]

    df_cat_lookup = pl.DataFrame({
        "category_str": ordered_cats,
        "category_vector": list(cat_vectors)  # Polars lưu vector dưới dạng List[Float]
    })

    # 3. ĐỌC & TIỀN XỬ LÝ ARTICLES
    print(f"Reading articles from {args.articles_path}...")
    try:
        # Chỉ lấy các cột cần thiết
        q = pl.scan_parquet(args.articles_path).select([
            pl.col("article_id").cast(pl.Int32),
            pl.col("published_time").cast(pl.Datetime),
            pl.col("category_str").cast(pl.String).fill_null("UNK"),

            # Ghép Title + Subtitle
            (pl.col("title").fill_null("") + " " + pl.col("subtitle").fill_null("")).alias("full_title"),

            # Body (Cắt ngắn để tối ưu tốc độ)
            pl.col("body").fill_null("").cast(pl.String).str.slice(0, args.max_body_chars)
        ])

        df = q.collect(engine="streaming")
        print(f"Loaded {len(df)} articles.")
    except Exception as e:
        print(f"Error reading articles file: {e}")
        sys.exit(1)

    # 4. HÀM EMBEDDING BATCHING
    def get_embeddings(texts, prefix="passage: "):
        # e5 model cần prefix "passage: " cho document
        inputs = [prefix + t for t in texts]

        # SentenceTransformer tự động handle batching và progress bar
        embeddings = model.encode(
            inputs,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize để dùng Cosine Similarity tốt hơn
        )
        return list(embeddings)  # Trả về list of arrays để đưa vào Polars

    # 5. THỰC HIỆN EMBEDDING (TITLE)
    print("\n--- Embedding Titles (Title + Subtitle) ---")
    titles = df["full_title"].to_list()
    title_vectors = get_embeddings(titles)

    # 6. THỰC HIỆN EMBEDDING (BODY)
    print("\n--- Embedding Bodies ---")
    bodies = df["body"].to_list()
    body_vectors = get_embeddings(bodies)

    # 7. GOM DỮ LIỆU & SAVE
    print("\nConstructing final DataFrame...")

    # Thêm cột vector vào DF gốc
    df_final = df.with_columns([
        pl.Series(name="title_vector", values=title_vectors),
        pl.Series(name="body_vector", values=body_vectors)
    ])

    # Join với Category Vectors (Dựa vào category_str)
    # Dùng left join để giữ nguyên số lượng bài báo
    df_final = df_final.join(df_cat_lookup, on="category_str", how="left")

    # Chọn lại các cột cần lưu theo yêu cầu
    output_cols = [
        "article_id",
        "published_time",
        "category_str",
        "title_vector",
        "body_vector",
        "category_vector"
    ]

    df_final = df_final.select(output_cols)

    print(f"Saving to {args.output_path}...")
    # Lưu dưới dạng Parquet (tối ưu nén vector)
    df_final.write_parquet(args.output_path)

    print("Done! Preview:")
    print(df_final.head(1))


if __name__ == "__main__":
    main()