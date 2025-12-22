import polars as pl

train_behaviors = pl.read_parquet("processed_parquet/train/behaviors_processed.parquet")
train_history = pl.read_parquet("processed_parquet/train/history_processed.parquet")

for idx, x in enumerate(train_behaviors.sample(n=1)):
    print(x)

