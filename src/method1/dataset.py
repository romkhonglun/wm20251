import torch
import polars as pl
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
import pytorch_lightning as L
import torch.distributed as dist
from itertools import islice


# ==========================================
# 1. QUẢN LÝ EMBEDDING (ID MAPPING)
# ==========================================
class NewsEmbeddingManager:
    def __init__(self, processed_dir):
        self.processed_dir = Path(processed_dir)
        # Load ID từ npy (đã được tiền xử lý)
        self.art_ids = np.load(self.processed_dir / "article_ids.npy")
        self.nid2index = {str(nid): i for i, nid in enumerate(self.art_ids)}

    def map_ids_to_indices(self, id_list):
        if not isinstance(id_list, (list, np.ndarray)):
            id_list = [id_list]
        return [self.nid2index.get(str(i), 0) for i in id_list]


# ==========================================
# 2. DATASET (STREAMING & MULTI-TASK)
# ==========================================
class NAMLIterableDataset(IterableDataset):
    def __init__(self, behaviors_path, history_path, articles_path, embedding_manager,
                 history_len=30, neg_ratio=4, batch_size=32,
                 mode='train', shuffle_buffer_size=10000, seed=42):
        super().__init__()
        self.behaviors_path = behaviors_path
        self.articles_path = articles_path
        self.embedding_manager = embedding_manager
        self.history_len = history_len
        self.neg_ratio = neg_ratio
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size if mode == 'train' else 0
        self.seed = seed

        # Load mapping user -> history ngay khi khởi tạo
        self.user_history = self._load_history_with_wpm(history_path)

    def _load_history_with_wpm(self, history_path):
        print(f"[{self.mode.upper()}] Loading history & Pre-calculating interaction features...")

        # Load articles để lấy độ dài văn bản (phục vụ tính WPM)
        lf_art = pl.scan_parquet(self.articles_path).select([
            pl.col("article_id").cast(pl.String),
            pl.col("body").fill_null("").str.split(" ").list.len().alias("word_count")
        ])

        # Load history
        lf_hist = pl.scan_parquet(history_path).select([
            pl.col("user_id"),
            pl.col("article_id_fixed"),
            pl.col("impression_time_fixed"),
            pl.col("scroll_percentage_fixed"),
            pl.col("read_time_fixed")
        ])

        # Explode và xử lý kiểu dữ liệu
        q = lf_hist.explode(["article_id_fixed", "scroll_percentage_fixed", "read_time_fixed", "impression_time_fixed"])
        q = q.rename({"article_id_fixed": "article_id"}).with_columns(pl.col("article_id").cast(pl.String))

        # Join với article info
        q = q.join(lf_art, on="article_id", how="left")
        q = q.with_columns([
            pl.col("read_time_fixed").fill_null(0.0),
            pl.col("word_count").fill_null(0)
        ])

        # Logic WPM và Scroll (như đã thống nhất)
        wpm_expr = (pl.col("word_count") / pl.col("read_time_fixed") * 60).fill_nan(9999).fill_null(0)
        scroll_logic = (
            pl.when(pl.col("read_time_fixed") < 5.0).then(0.0)
            .when(wpm_expr < 300).then(100.0)
            .when(wpm_expr > 500).then(0.0)
            .otherwise((500 - wpm_expr) / 2.0)
        )

        q = q.with_columns(pl.col("scroll_percentage_fixed").fill_null(scroll_logic))
        q = q.filter(pl.col("article_id").is_not_null())

        df = q.sort(["user_id", "impression_time_fixed"]).collect()

        # Chuyển sang format mảng để truy xuất nhanh (CSR-like)
        raw_nids = df["article_id"].to_list()
        mapped_indices = self.embedding_manager.map_ids_to_indices(raw_nids)

        all_indices = np.array(mapped_indices, dtype=np.int32)
        all_scrolls = (df["scroll_percentage_fixed"].to_numpy() / 100.0).astype(np.float32)  # Normalize 0-1
        all_times = np.log1p(df["read_time_fixed"].to_numpy().astype(np.float32))

        user_counts = df.group_by("user_id", maintain_order=True).len()
        users = user_counts["user_id"].to_list()
        lengths = user_counts["len"].to_numpy()

        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        user_map = {u: (offsets[i], lengths[i]) for i, u in enumerate(users)}

        print(f"[{self.mode.upper()}] History loaded. Users: {len(user_map)}")
        return {"map": user_map, "indices": all_indices, "scrolls": all_scrolls, "times": all_times}

    def _process_row(self, user_id, clicked_ids, inview_ids):
        # 1. History Extraction & Padding
        if user_id in self.user_history["map"]:
            start, length = self.user_history["map"][user_id]
            h_idx = self.user_history["indices"][start: start + length]
            h_scr = self.user_history["scrolls"][start: start + length]
            h_tim = self.user_history["times"][start: start + length]
        else:
            h_idx = h_scr = h_tim = np.array([], dtype=np.float32)

        # Truncate/Pad
        if len(h_idx) < self.history_len:
            pad_len = self.history_len - len(h_idx)
            h_idx = np.pad(h_idx, (0, pad_len), 'constant')
            h_scr = np.pad(h_scr, (0, pad_len), 'constant')
            h_tim = np.pad(h_tim, (0, pad_len), 'constant')
        else:
            h_idx = h_idx[-self.history_len:]
            h_scr = h_scr[-self.history_len:]
            h_tim = h_tim[-self.history_len:]

        # 2. Candidate Sampling
        clicked_ids = [str(x) for x in clicked_ids] if clicked_ids else []
        inview_ids = [str(x) for x in inview_ids] if inview_ids else []

        if self.mode == 'train' and clicked_ids:
            pos_id = np.random.choice(clicked_ids)
        else:
            pos_id = clicked_ids[0] if clicked_ids else (inview_ids[0] if inview_ids else "0")

        neg_pool = list(set(inview_ids) - set(clicked_ids))
        if self.mode == 'train':
            if len(neg_pool) >= self.neg_ratio:
                neg_ids = np.random.choice(neg_pool, self.neg_ratio, replace=False).tolist()
            else:
                neg_ids = (neg_pool * (self.neg_ratio // max(1, len(neg_pool)) + 1))[:self.neg_ratio]
        else:
            neg_ids = neg_pool[:self.neg_ratio] if len(neg_pool) >= self.neg_ratio else (neg_pool * self.neg_ratio)[
                                                                                        :self.neg_ratio]

        cand_ids = [pos_id] + neg_ids
        cand_indices = self.embedding_manager.map_ids_to_indices(cand_ids)

        # 3. Return Dictionary (Phải khớp chính xác key với VariantNAML.forward)
        return {
            "hist_indices": torch.LongTensor(h_idx),
            "hist_scroll": torch.FloatTensor(h_scr),  # Đổi từ hist_scrolls -> hist_scroll
            "hist_time": torch.FloatTensor(h_tim),  # Đổi từ hist_times -> hist_time
            "cand_indices": torch.LongTensor(cand_indices),
            "label_click": torch.FloatTensor([1.0] + [0.0] * self.neg_ratio)
        }

    def _stream_from_parquet(self):
        pq_file = pq.ParquetFile(self.behaviors_path)
        for batch in pq_file.iter_batches(batch_size=self.batch_size * 50):
            u_ids = batch["user_id"]
            c_cols = batch["article_ids_clicked"]
            i_cols = batch["article_ids_inview"]
            for i in range(len(batch)):
                yield self._process_row(u_ids[i].as_py(), c_cols[i].as_py(), i_cols[i].as_py())

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # DDP Sharding
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size, rank = 1, 0

        total_shards = world_size * num_workers
        current_shard_id = rank * num_workers + worker_id

        iterator = self._stream_from_parquet()
        sharded_iterator = islice(iterator, current_shard_id, None, total_shards)

        # Shuffle Buffer logic
        if self.shuffle_buffer_size > 0:
            buffer = []
            for item in sharded_iterator:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer_size:
                    idx = np.random.randint(len(buffer))
                    yield buffer.pop(idx)
            np.random.shuffle(buffer)
            yield from buffer
        else:
            yield from sharded_iterator


# ==========================================
# 3. LIGHTNING DATA MODULE
# ==========================================
class NAMLDataModule(L.LightningDataModule):
    def __init__(self, root_path, embedding_path, batch_size=32, history_len=30, neg_ratio=4, num_workers=2):
        super().__init__()
        self.save_hyperparameters()
        self.emb_manager = None

    def setup(self, stage=None):
        if self.emb_manager is None:
            self.emb_manager = NewsEmbeddingManager(self.hparams.embedding_path)

        art_p = Path(self.hparams.root_path) / "articles.parquet"
        root = Path(self.hparams.root_path)

        if stage == "fit" or stage is None:
            self.train_ds = NAMLIterableDataset(
                root / "train/behaviors.parquet", root / "train/history.parquet", art_p,
                self.emb_manager, self.hparams.history_len, self.hparams.neg_ratio,
                self.hparams.batch_size, 'train'
            )
            self.val_ds = NAMLIterableDataset(
                root / "validation/behaviors.parquet", root / "validation/history.parquet", art_p,
                self.emb_manager, self.hparams.history_len, self.hparams.neg_ratio,
                self.hparams.batch_size, 'val'
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=True)