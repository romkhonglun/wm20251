uv run /home2/congnh/wm/src/method2/train.py \
         --processed-dir /home2/congnh/wm/processed_parquet/ \
         --embedding-path /home2/congnh/wm/embedding/body_emb.npy \
         --lr-scheduler cosine \
         --epochs 10 \
         --batch_size 512

