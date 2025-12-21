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
        try:
            self.art_ids = np.load(self.processed_dir / "article_ids.npy")
        except FileNotFoundError:
            print("⚠️ Warning: article_ids.npy not found. Using empty mapping.")
            self.art_ids = np.array([])

        self.nid2index = {str(nid): i for i, nid in enumerate(self.art_ids)}

    def map_ids_to_indices(self, id_list):
        if not isinstance(id_list, (list, np.ndarray)):
            id_list = [id_list]
        return [self.nid2index.get(str(i), 0) for i in id_list]


# ==========================================
# 2. DATASET (STREAMING & SIMPLE FEATURES)
# ==========================================
class NAMLIterableDataset(IterableDataset):
    def __init__(self, behaviors_path, history_path, embedding_manager,
                 history_len=30, neg_ratio=4, batch_size=32,
                 mode='train', shuffle_buffer_size=10000, seed=42):
        super().__init__()
        self.behaviors_path = behaviors_path
        # Không cần articles_path nữa vì không tính WPM
        self.embedding_manager = embedding_manager
        self.history_len = history_len
        self.neg_ratio = neg_ratio
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size if mode == 'train' else 0
        self.seed = seed

        # Load history
        self.user_history = self._load_history_simple(history_path)

    def _load_history_simple(self, history_path):
        """
        Load History và chỉ lấy Time/Scroll thô (Raw Features).
        Không tính toán WPM hay heuristic phức tạp.
        """
        print(f"[{self.mode.upper()}] Loading history (Simple Mode: Raw Time & Scroll)...")

        # 1. Load History từ file Parquet (Chỉ lấy cột cần thiết)
        lf_hist = pl.scan_parquet(history_path).select([
            pl.col("user_id"),
            pl.col("hist_ids"),
            pl.col("hist_ts"),
            pl.col("hist_scroll"),
            pl.col("hist_time")
        ])

        # 2. Explode (Trải phẳng)
        q = lf_hist.explode(["hist_ids", "hist_scroll", "hist_time", "hist_ts"])
        q = q.rename({"hist_ids": "article_id"}).with_columns(pl.col("article_id").cast(pl.String))

        # 3. Clean Data đơn giản (Fill Null = 0)
        # Không còn logic WPM phức tạp, chỉ đảm bảo không có NaN
        q = q.with_columns([
            pl.col("hist_time").fill_null(0.0),
            pl.col("hist_scroll").fill_null(0.0)
        ])

        q = q.filter(pl.col("article_id").is_not_null())

        # Collect về DataFrame
        df = q.sort(["user_id", "hist_ts"]).collect()

        # 4. Chuyển đổi sang Numpy
        raw_nids = df["article_id"].to_list()
        mapped_indices = self.embedding_manager.map_ids_to_indices(raw_nids)

        all_indices = np.array(mapped_indices, dtype=np.int32)

        # Normalize Scroll về [0, 1] (Giả sử dữ liệu gốc là 0-100)
        all_scrolls = (df["hist_scroll"].to_numpy() / 100.0).astype(np.float32)

        # Log Normalize Time để giảm độ lệch (skewness) của dữ liệu thời gian
        all_times = np.log1p(df["hist_time"].to_numpy().astype(np.float32))

        # Tạo Index Map
        user_counts = df.group_by("user_id", maintain_order=True).len()
        users = user_counts["user_id"].to_list()
        lengths = user_counts["len"].to_numpy()

        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        user_map = {u: (offsets[i], lengths[i]) for i, u in enumerate(users)}

        print(f"[{self.mode.upper()}] History loaded. Users: {len(user_map)}")
        return {"map": user_map, "indices": all_indices, "scrolls": all_scrolls, "times": all_times}

    def _process_row(self, user_id, clicked_ids, inview_ids):
        # 1. Truy xuất & Padding History
        if user_id in self.user_history["map"]:
            start, length = self.user_history["map"][user_id]
            h_idx = self.user_history["indices"][start: start + length]
            h_scr = self.user_history["scrolls"][start: start + length]
            h_tim = self.user_history["times"][start: start + length]
        else:
            h_idx = h_scr = h_tim = np.array([], dtype=np.float32)

        # Cắt hoặc Pad
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
            neg_ids = neg_pool[:self.neg_ratio]
            if len(neg_ids) < self.neg_ratio:
                neg_ids += [pos_id] * (self.neg_ratio - len(neg_ids))

        cand_ids = [pos_id] + neg_ids
        cand_indices = self.embedding_manager.map_ids_to_indices(cand_ids)

        return {
            "hist_indices": torch.LongTensor(h_idx),
            # "hist_scroll": torch.FloatTensor(h_scr),
            # "hist_time": torch.FloatTensor(h_tim),
            "cand_indices": torch.LongTensor(cand_indices),
            "label_click": torch.Tensor([1.0] + [0.0] * self.neg_ratio)
        }

    def _stream_from_parquet(self):
        pq_file = pq.ParquetFile(self.behaviors_path)
        for batch in pq_file.iter_batches(batch_size=self.batch_size * 50):
            u_ids = batch["user_id"]
            c_cols = batch["clk_ids"]
            i_cols = batch["inv_ids"]
            for i in range(len(batch)):
                yield self._process_row(u_ids[i].as_py(), c_cols[i].as_py(), i_cols[i].as_py())

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size, rank = 1, 0

        total_shards = world_size * num_workers
        current_shard_id = rank * num_workers + worker_id

        iterator = self._stream_from_parquet()
        sharded_iterator = islice(iterator, current_shard_id, None, total_shards)

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

        root = Path(self.hparams.root_path)

        # Lưu ý: Không cần articles_path nữa
        if stage == "fit" or stage is None:
            self.train_ds = NAMLIterableDataset(
                behaviors_path=root / "train" / "behaviors_processed.parquet",
                history_path=root / "train" / "history_processed.parquet",
                embedding_manager=self.emb_manager,
                history_len=self.hparams.history_len,
                neg_ratio=self.hparams.neg_ratio,
                batch_size=self.hparams.batch_size,
                mode='train'
            )
            self.val_ds = NAMLIterableDataset(
                behaviors_path=root / "validation" / "behaviors_processed.parquet",
                history_path=root / "validation" / "history_processed.parquet",
                embedding_manager=self.emb_manager,
                history_len=self.hparams.history_len,
                neg_ratio=self.hparams.neg_ratio,
                batch_size=self.hparams.batch_size,
                mode='val'
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True)