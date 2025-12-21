uv run src/baseline/train.py \
    --root-dir "/home2/congnh/wm/processed_parquet" \
    --embedding-dir "/home2/congnh/wm/embedding" \
    --batch-size 32 \
    --epochs 5 \
    --lr 0.001