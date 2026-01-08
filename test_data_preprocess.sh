uv run src/preprocess/generate_category_description.py \
    --articles_path "/home2/congnh/wm/input/ebnerd_testset/articles.parquet" \
    --output_path "/home2/congnh/wm/test_category_embeddings.pt"
uv run src/preprocess/title_body_embedding.py \
    --articles_path "/home2/congnh/wm/input/ebnerd_testset/articles.parquet" \
    --cat_emb_path "/home2/congnh/wm/test_category_embeddings.pt" \
    --output_path "/home2/congnh/wm/test_articles_embedded.parquet" \
    --batch_size 32
%mkdir -p embedding_test
uv run src/preprocess/prepare_data.py \
    --input_path "/home2/congnh/wm/test_articles_embedded.parquet" \
    --output_dir "/home2/congnh/wm/embedding_test"