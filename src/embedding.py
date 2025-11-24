from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import torch
import pickle
import os

if __name__ == "__main__":
    VALID_SIZE = ['large','small']
    size = ""
    while size not in VALID_SIZE:
        user_input = input(r"Hay chon kich thuoc dataset (large/small)").strip().lower()

        if user_input in VALID_SIZE:
            size = user_input
        else:
            print(f"Lỗi: '{user_input}' không phải là lựa chọn hợp lệ. Vui lòng thử lại.")

    print(f"\nBạn đã chọn size: {size}")

    train_path = Path.cwd().parent / "input" / f"ebnerd_{size}"
    test_path = Path.cwd().parent/"input"/"ebnerd_testset"

    train_article_df = pd.read_parquet(train_path/"articles.parquet")
    test_article_df = pd.read_parquet(test_path/"articles.parquet")

    if not isinstance(train_article_df,pd.DataFrame) or not isinstance(test_article_df,pd.DataFrame):
        print("article_df problem")

    articles_df = pd.concat([train_article_df, test_article_df])

    articles_df['Text'] = articles_df['title'] + articles_df['subtitle'].fillna('') + articles_df['body'].fillna('')
    # Calculate the maximum number of words in the 'text' column
    max_words = articles_df['Text'].apply(lambda x: len(x.split())).max()

    print(f"The maximum number of words in the 'Text' column is: {max_words}")

    id_list = articles_df['article_id'].to_list()
    text_list = articles_df['Text'].to_list()

    # Load https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    encoder = SentenceTransformer("intfloat/multilingual-e5-large-instruct", model_kwargs={"dtype": "float16"})
    encoder.to('cuda')

    with torch.no_grad():
        embeddings = encoder.encode(text_list, batch_size=16, show_progress_bar=True)

    articles_dict = {}
    for i, embedding in enumerate(embeddings):
        articles_dict[articles_df.iloc[i]['article_id']] = embedding

    with open(Path.cwd().parent/"input"/f"EB_NeRD_{size}_embedded_text.pkl", "wb") as file:
        pickle.dump(articles_dict, file)