uv run /home2/congnh/wm/src/method2/train.py \
         --processed-dir /home2/congnh/wm/processed_parquet_large/ \
         --embedding-path /home2/congnh/wm/embedding_large/body_emb.npy \
         --lr-scheduler cosine \
         --epochs 10 \
         --batch_size 512

