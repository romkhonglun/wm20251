#uv run src/baseline/test.py \
#    --checkpoint "/home2/congnh/wm/NewsRecSys/wbbqb9vb/checkpoints/Baseline-epoch=04-val/auc=0.5749.ckpt" \
#    --test-dir "/home2/congnh/wm/processed_parquet_test" \
#    --embedding-dir "/home2/congnh/wm/embedding_test" \
#    --output "/home2/congnh/wm/submission_baseline.zip" \
#    --batch-size 512
#uv run src/method1/test.py \
#    --checkpoint "/home2/congnh/wm/checkpoints/method1/naml-m1-epoch=04-val/auc=0.6249.ckpt" \
#    --test-dir "/home2/congnh/wm/processed_parquet_test" \
#    --embedding-dir "/home2/congnh/wm/embedding_test" \
#    --output "/home2/congnh/wm/submission_method1.zip" \
#    --batch-size 512
uv run src/method2/test.py \
    --checkpoint "/home2/congnh/wm/auc=0.7362.ckpt" \
    --test-dir "/home2/congnh/wm/processed_parquet_test" \
    --articles-path "/home2/congnh/wm/processed_parquet_test/articles_processed.parquet" \
    --embedding-path "/home2/congnh/wm/embedding_test/body_emb.npy" \
    --output "/home2/congnh/wm/submission_method2.zip" \
    --batch-size 512