import argparse
import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for news categories using LLM descriptions.")

    parser.add_argument(
        "--articles_path",
        type=str,
        default="/kaggle/input/ebnerd-small/articles.parquet",
        help="Path to the articles parquet file containing category information."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="small_category_embeddings.pt",
        help="Path to save the resulting .pt file."
    )
    parser.add_argument(
        "--gen_model_id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID for text generation (LLM)."
    )
    parser.add_argument(
        "--emb_model_id",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="HuggingFace model ID for embedding generation."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. TRÍCH XUẤT CATEGORY DUY NHẤT
    print(f"Reading unique categories from {args.articles_path}...")
    try:
        df = pl.read_parquet(args.articles_path).select(
            pl.col("category_str").cast(pl.String).fill_null("UNK")
        ).unique()
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)

    categories = df["category_str"].to_list()
    # Thêm UNK vào đầu danh sách để làm index 0 (nếu chưa có)
    if "UNK" not in categories:
        categories.insert(0, "UNK")

    # Sắp xếp để index cố định (Quan trọng cho việc map ID sau này)
    categories = sorted(categories)
    print(f"Found {len(categories)} unique categories: {categories[:5]}...")

    # 2. KHỞI TẠO MODEL GENERATION (QWEN)
    print(f"\nLoading generation model: {args.gen_model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.gen_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.gen_model_id,
            torch_dtype=torch.float16,
            device_map="auto"  # Tự động dùng GPU
        )
    except Exception as e:
        print(f"Error loading generation model: {e}")
        sys.exit(1)

    # Hàm sinh mô tả
    def generate_desc(cat_name):
        if cat_name == "UNK":
            return "General news content with no specific category."

        # Prompt song ngữ (hoặc tiếng Anh tùy dataset của bạn)
        prompt = f"""Beskriv kort, hvilken type indhold man finder i kategorien '{cat_name}' på nyhedsmediet Ekstra Bladet. 
        Svaret skal være på dansk og max én sætning.
        Beskrivelse:"""

        messages = [
            {"role": "system", "content": "Du er en hjælpsom dansk nyhedsredaktør."},
            # Bạn là biên tập viên tin tức Đan Mạch
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            temperature=0.3,  # Giữ creativity thấp để mô tả chính xác
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    # 3. GENERATE DESCRIPTION
    print("\nGenerating descriptions...")
    descriptions = []
    for cat in tqdm(categories):
        desc = generate_desc(cat)
        descriptions.append(desc)
        # print(f"Cat: {cat} -> Desc: {desc}")

    # Giải phóng RAM Qwen
    del model, tokenizer
    torch.cuda.empty_cache()

    # 4. EMBEDDING (Biến Text thành Vector)
    print(f"\nEmbedding descriptions using {args.emb_model_id}...")
    # Dùng Sentence-Transformer để có vector chất lượng cao hơn lấy hidden state của LLM
    embed_model = SentenceTransformer(args.emb_model_id)

    # Prefix "query: " (hoặc passage:) là yêu cầu của model e5 để embedding tốt hơn
    desc_inputs = [f"passage: {d}" for d in descriptions]
    vectors = embed_model.encode(desc_inputs, convert_to_tensor=True)

    print(f"Embedding shape: {vectors.shape}")

    # 5. LƯU KẾT QUẢ
    # Lưu cả Dictionary map (để biết index nào là category nào) và Tensor
    save_data = {
        "cat2id": {c: i for i, c in enumerate(categories)},
        "embeddings": vectors.cpu()  # Chuyển về CPU để lưu
    }
    torch.save(save_data, args.output_path)
    print(f"Saved processed embeddings to {args.output_path}")


if __name__ == "__main__":
    main()