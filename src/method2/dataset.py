import torch
import polars as pl
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
from pathlib import Path
import pytorch_lightning as L
import torch.distributed as dist
from itertools import islice


# ==========================================
# 1. QUẢN LÝ VECTOR EMBEDDING (Fix OOB)
# ==========================================
class NewsEmbeddingManager:
    def __init__(self, embedding_path, vocab_size=None):
        self.embedding_path = Path(embedding_path)
        print(f"🚀 Loading Article Vectors: {self.embedding_path}")

        # Nếu vocab_size được truyền vào, ta sẽ init matrix an toàn
        self.vocab_size = vocab_size

        try:
            # Load vectors (Numpy format)
            self.vectors = np.load(self.embedding_path).astype(np.float32)

            # Kiểm tra kích thước và pad nếu cần
            if self.vocab_size and self.vectors.shape[0] < self.vocab_size:
                pad_len = self.vocab_size - self.vectors.shape[0]
                print(f"⚠️ Vector dim {self.vectors.shape[0]} < Vocab {self.vocab_size}. Padding with random...")
                pad_vecs = np.random.randn(pad_len, self.vectors.shape[1]).astype(np.float32)
                self.vectors = np.concatenate([self.vectors, pad_vecs], axis=0)

            # L2 Normalize
            norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self.vectors = np.divide(self.vectors, norm, out=np.zeros_like(self.vectors), where=norm != 0)
            self.vectors = np.nan_to_num(self.vectors, nan=0.0)

            print(f"✅ Vectors Ready. Shape: {self.vectors.shape}")
        except Exception as e:
            print(f"⚠️ Warning: {e}. Using Random Vectors.")
            # Fallback size an toàn
            safe_size = self.vocab_size if self.vocab_size else 150000
            self.vectors = np.random.randn(safe_size, 1024).astype(np.float32)

    def get_vectors_by_indices(self, indices):
        # Clip indices để không bao giờ vượt quá size vector
        indices = np.clip(indices, 0, self.vectors.shape[0] - 1)
        return self.vectors[indices]


# ==========================================
# 2. STATIC ARTICLE FEATURES (Updated Schema)
# ==========================================
class StaticArticleFeatures:
    def __init__(self, processed_path):
        print(f"📊 Loading Static Features from {processed_path}...")
        df = pl.read_parquet(processed_path)

        # Lấy Max ID để xác định kích thước ma trận
        # Cần cast về int để chắc chắn
        ids = df["id_encoded"].cast(pl.Int32).to_numpy()
        max_id = ids.max() if len(ids) > 0 else 0
        self.vocab_size = max_id + 1

        if self.vocab_size <= 1:
            print("🚨 CRITICAL: Vocab size is 1. Check articles_processed.parquet content!")

        # 6 Features số: Views, Inviews, Sentiment, ReadTime, PubTime, BodyLen
        self.num_mat = np.zeros((self.vocab_size, 6), dtype=np.float32)
        # 1 Feature category
        self.cat_mat = np.zeros((self.vocab_size, 1), dtype=np.int32)

        # Định nghĩa cột khớp với file preprocess mới
        cols_num = [
            "norm_views", "norm_inviews", "sentiment_score",
            "norm_read_time", "published_time", "body_len"
        ]

        vals_num = df.select(cols_num).to_numpy().astype(np.float32)

        # Fill dữ liệu vào ma trận tại các hàng tương ứng với ID
        # Sử dụng ids làm index
        self.num_mat[ids] = np.nan_to_num(vals_num, nan=0.0)

        vals_cat = df.select("cat_encoded").to_numpy().astype(np.int32)
        self.cat_mat[ids] = vals_cat

        print(f"✅ Features Loaded. Matrix Shape: {self.num_mat.shape}")

    def get(self, indices):
        # indices là list hoặc array các ID bài báo
        # Cần đảm bảo indices không vượt quá vocab_size (Fix lỗi IndexError)
        safe_indices = np.clip(indices, 0, self.vocab_size - 1)
        return self.num_mat[safe_indices], self.cat_mat[safe_indices]


class NewsBaseLogic:
    """Chứa logic xử lý chung cho cả Map và Iterable Dataset"""

    def _init_base(self, history_path, article_features, embedding_manager, history_len, neg_ratio):
        self.art_feats = article_features
        self.emb_manager = embedding_manager
        self.history_len = history_len
        self.neg_ratio = neg_ratio

        # Load toàn bộ lịch sử vào RAM (Numpy)
        self._load_history_to_numpy(history_path)

    def _load_history_to_numpy(self, path):
        print(f"📦 Pre-loading History from {path}...")
        df = pl.read_parquet(path)

        # Lấy User ID max để tạo matrix
        uids = df["user_id"].cast(pl.Int32).to_numpy()
        max_uid = uids.max() if len(uids) > 0 else 0
        num_users = max_uid + 1

        # Init matrices (User x History_Len)
        self.hist_ids_mat = np.zeros((num_users, self.history_len), dtype=np.int32)
        self.hist_scr_mat = np.zeros((num_users, self.history_len), dtype=np.float32)
        self.hist_tm_mat = np.zeros((num_users, self.history_len), dtype=np.float32)
        self.hist_ts_mat = np.zeros((num_users, self.history_len), dtype=np.float64)
        self.hist_lens = np.zeros(num_users, dtype=np.int32)

        # Iterate rows
        rows = df.select([
            "user_id", "hist_ids", "hist_scroll", "hist_time", "hist_ts"
        ]).iter_rows()

        for uid, h_ids, h_scr, h_tm, h_ts in rows:
            if h_ids is None or len(h_ids) == 0:
                continue

            l = len(h_ids)
            # Cắt nếu dài quá
            if l > self.history_len:
                h_ids = h_ids[-self.history_len:]
                h_scr = h_scr[-self.history_len:]
                h_tm = h_tm[-self.history_len:]
                h_ts = h_ts[-self.history_len:]
                l = self.history_len

            # --- [FIX 1: CLEANING LOGIC] ---
            # Helper function để convert list -> numpy -> clean NaN
            def safe_float_arr(arr_list):
                if arr_list is None: return np.zeros(l, dtype=np.float32)
                # Convert sang float32 (None sẽ thành NaN)
                arr = np.array(arr_list[:l], dtype=np.float32)
                # Thay thế NaN, Inf bằng 0.0
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            self.hist_ids_mat[uid, :l] = np.array(h_ids[:l], dtype=np.int32)
            self.hist_scr_mat[uid, :l] = safe_float_arr(h_scr)  # Đã clean
            self.hist_tm_mat[uid, :l] = safe_float_arr(h_tm)  # Đã clean
            self.hist_ts_mat[uid, :l] = np.array(h_ts[:l], dtype=np.float64)  # Timestamp giữ nguyên float64
            self.hist_lens[uid] = l

        print(f"✅ History Loaded. Users: {num_users}")

    def _process_row(self, row):
        # 1. Parse Input
        try:
            user_id = int(row["user_id"])
        except:
            return None

        imp_ts = row["imp_ts"] if row["imp_ts"] is not None else 0.0

        # 2. Lấy History (O(1) Access)
        if user_id >= self.hist_ids_mat.shape[0]:
            # Cold start user
            h_ids = np.zeros(self.history_len, dtype=np.int32)
            h_scr = np.zeros(self.history_len, dtype=np.float32)
            h_tm = np.zeros(self.history_len, dtype=np.float32)
            ts_diff_log = np.zeros(self.history_len, dtype=np.float32)
            curr_len = 0
        else:
            curr_len = self.hist_lens[user_id]
            h_ids = self.hist_ids_mat[user_id]
            h_ts_valid = self.hist_ts_mat[user_id, :curr_len]

            ts_diff_log = np.zeros(self.history_len, dtype=np.float32)
            if curr_len > 0:
                diffs = (imp_ts - h_ts_valid) / 3600.0
                ts_diff_log[:curr_len] = np.log1p(np.clip(diffs, 0, None))

            h_scr = self.hist_scr_mat[user_id]
            h_tm = self.hist_tm_mat[user_id]

        # --- [FIX 2: RUNTIME SANITIZATION] ---
        # Chắc chắn rằng không có NaN nào lọt lưới trước khi vào Tensor
        h_scr = np.nan_to_num(h_scr, nan=0.0)
        h_tm = np.nan_to_num(h_tm, nan=0.0)
        ts_diff_log = np.nan_to_num(ts_diff_log, nan=0.0)

        # 3. Candidate Sampling
        inv_ids = row["inv_ids"] if row["inv_ids"] else []
        clk_ids = row["clk_ids"] if row["clk_ids"] else []

        if not inv_ids and not clk_ids:
            return None

        # Chọn Positive & Negative (Giữ nguyên logic cũ)
        if clk_ids:
            pos_id = np.random.choice(clk_ids)
        else:
            pos_id = inv_ids[0]

        neg_pool = list(set(inv_ids) - set(clk_ids))
        if len(neg_pool) >= self.neg_ratio:
            neg_ids = np.random.choice(neg_pool, self.neg_ratio, replace=False).tolist()
        else:
            pool_source = neg_pool if neg_pool else [pos_id]
            neg_ids = []
            while len(neg_ids) < self.neg_ratio:
                neg_ids.extend(pool_source)
            neg_ids = neg_ids[:self.neg_ratio]

        candidate_ids = [pos_id] + neg_ids

        # 4. Get Features
        cand_nums, cand_cats = self.art_feats.get(candidate_ids)

        pub_times = cand_nums[:, 4]
        time_diffs = np.log1p(np.abs(imp_ts - np.nan_to_num(pub_times, nan=imp_ts)) / 3600.0)
        cand_nums[:, 4] = np.nan_to_num(time_diffs, nan=0.0)  # Clean NaN here too

        cand_vecs = self.emb_manager.get_vectors_by_indices(candidate_ids)
        scores = np.zeros((len(candidate_ids), 1), dtype=np.float32)

        return {
            "hist_indices": torch.from_numpy(h_ids.astype(np.int64)),
            "hist_scroll": torch.from_numpy(h_scr.astype(np.float32)),  # Đảm bảo float32
            "hist_time": torch.from_numpy(h_tm.astype(np.float32)),
            "hist_diff": torch.from_numpy(ts_diff_log.astype(np.float32)),
            "cand_indices": torch.tensor(candidate_ids, dtype=torch.long),
            "cand_num": torch.from_numpy(cand_nums.astype(np.float32)),
            "cand_cat": torch.from_numpy(cand_cats).long(),
            "cand_sim": torch.from_numpy(scores).float(),
            "imp_feats": torch.tensor([
                np.log1p(curr_len),
                (imp_ts % 86400) / 86400.0,
                float(row["norm_age"] or 0.0)
            ], dtype=torch.float),
            "labels": torch.tensor([1.0] + [0.0] * self.neg_ratio, dtype=torch.float)
        }


# ==========================================
# OPTION 1: MAP-STYLE DATASET
# ==========================================
class NewsMapDataset(Dataset, NewsBaseLogic):
    def __init__(self, behaviors_path, history_path, article_features, embedding_manager,
                 history_len=30, neg_ratio=4, **kwargs):
        self._init_base(history_path, article_features, embedding_manager, history_len, neg_ratio)

        print(f"📂 Loading Behaviors Map-style: {behaviors_path}")
        df = pl.read_parquet(behaviors_path)

        # Load columns vào RAM để access nhanh qua index
        self.user_ids = df["user_id"].cast(pl.Int32).to_numpy()  # Đảm bảo int
        self.imp_ts = df["imp_ts"].fill_null(0.0).to_numpy()
        self.inv_ids = df["inv_ids"].to_list()
        self.clk_ids = df["clk_ids"].to_list()
        self.norm_age = df["norm_age"].fill_null(0.0).to_numpy()
        self.length = len(df)
        print(f"✅ Behaviors Loaded. Samples: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = {
            "user_id": self.user_ids[idx],
            "imp_ts": self.imp_ts[idx],
            "inv_ids": self.inv_ids[idx],
            "clk_ids": self.clk_ids[idx],
            "norm_age": self.norm_age[idx]
        }
        res = self._process_row(row)
        # Nếu None (do lỗi data), trả về mẫu tiếp theo hoặc mẫu 0 (MapDataset kị return None)
        if res is None:
            return self.__getitem__((idx + 1) % self.length)
        return res


# ==========================================
# OPTION 2: ITERABLE DATASET
# ==========================================
class NewsStreamDataset(IterableDataset, NewsBaseLogic):
    def __init__(self, behaviors_path, history_path, article_features, embedding_manager,
                 history_len=30, neg_ratio=4, batch_size=32, shuffle_buffer=1000, mode='train'):
        self._init_base(history_path, article_features, embedding_manager, history_len, neg_ratio)
        self.behaviors_path = behaviors_path
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.mode = mode

    def _get_stream(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        pf = pq.ParquetFile(self.behaviors_path)

        # Logic Sharding: Chia file thành các row groups
        # num_row_groups = pf.num_row_groups
        # Ở đây dùng iter_batches đơn giản

        # Skip rows cho từng worker
        start_idx = rank * num_workers + worker_id
        step = world_size * num_workers

        # Đọc chunk lớn để tối ưu I/O
        for batch in pf.iter_batches(batch_size=self.batch_size * 20):
            # Convert pyarrow batch -> python dict list
            # Cách này nhanh hơn to_pydict() full batch nếu chỉ iterate
            u_ids = batch["user_id"]
            imp_ts = batch["imp_ts"]
            inv_ids = batch["inv_ids"]
            clk_ids = batch["clk_ids"]
            norm_ages = batch["norm_age"]

            length = len(batch)
            for i in range(start_idx, length, step):  # Interleaved sampling
                yield {
                    "user_id": u_ids[i].as_py(),
                    "imp_ts": imp_ts[i].as_py(),
                    "inv_ids": inv_ids[i].as_py(),
                    "clk_ids": clk_ids[i].as_py(),
                    "norm_age": norm_ages[i].as_py()
                }
            # Reset start_idx cho batch sau (thực ra start_idx nên reset về relative của worker)
            # Logic step chuẩn hơn: yield i, i+step...
            start_idx = (start_idx - length) % step

    def __iter__(self):
        stream = self._get_stream()
        if self.mode == 'train':
            buffer = []
            for item in stream:
                processed = self._process_row(item)
                if processed:
                    buffer.append(processed)
                    if len(buffer) >= self.shuffle_buffer:
                        yield buffer.pop(np.random.randint(len(buffer)))
            while buffer:
                yield buffer.pop(np.random.randint(len(buffer)))
        else:
            for item in stream:
                processed = self._process_row(item)
                if processed:
                    yield processed


# ==========================================
# DATA MODULE
# ==========================================
class NAMLDataModule(L.LightningDataModule):
    def __init__(self, processed_dir, embedding_path, use_iterable=False, batch_size=64, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        self.processed_dir = Path(processed_dir)
        self.embedding_path = Path(embedding_path)

    def setup(self, stage=None):
        # Init Static Features trước
        art_path = self.processed_dir / "articles_processed.parquet"
        art_feats = StaticArticleFeatures(art_path)

        # Init Embedding với vocab_size lấy từ art_feats để đảm bảo khớp
        emb_manager = NewsEmbeddingManager(self.embedding_path, vocab_size=art_feats.vocab_size)

        DatasetClass = NewsStreamDataset if self.hparams.use_iterable else NewsMapDataset

        if stage in ('fit', None):
            self.train_ds = DatasetClass(
                self.processed_dir / "train/behaviors_processed.parquet",
                self.processed_dir / "train/history_processed.parquet",
                art_feats, emb_manager, mode='train'
            )
            self.val_ds = DatasetClass(
                self.processed_dir / "validation/behaviors_processed.parquet",
                self.processed_dir / "validation/history_processed.parquet",
                art_feats, emb_manager, mode='val'
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=(not self.hparams.use_iterable),
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True
        )