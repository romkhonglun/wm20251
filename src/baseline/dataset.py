import torch
import polars as pl
import numpy as np
import pyarrow.parquet as pq
import pytorch_lightning as L
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch.distributed as dist
from pathlib import Path
import gc


# --- 1. QUẢN LÝ EMBEDDING (Tối giản) ---
class NewsEmbeddingManager:
    def __init__(self, processed_dir):
        # Load xong xóa ngay mảng numpy gốc để tiết kiệm RAM
        # Chỉ giữ lại dictionary mapping (nhẹ hơn nhiều)
        ids_path = Path(processed_dir) / "article_ids.npy"
        temp_ids = np.load(ids_path)
        self.nid2index = {nid: i for i, nid in enumerate(temp_ids)}
        del temp_ids
        gc.collect()

    def map_ids_to_indices(self, id_list):
        # Map ID sang Index, nếu không thấy trả về 0
        return [self.nid2index.get(i, 0) for i in id_list]


# --- 2. DATASET (Tối ưu I/O và RAM) ---
class NAMLIterableDataset(IterableDataset):
    def __init__(self, behaviors_path, history_path, embedding_manager,
                 history_len=30, neg_ratio=4, batch_size=32,
                 mode='train', shuffle_buffer_size=5000):
        super().__init__()
        self.behaviors_path = behaviors_path
        self.embedding_manager = embedding_manager
        self.history_len = history_len
        self.neg_ratio = neg_ratio
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle_buffer_size = shuffle_buffer_size if mode == 'train' else 0

        # Load History 1 lần duy nhất
        self.user_history = self._load_history(history_path)

    def _load_history(self, path):
        # Chỉ đọc 2 cột cần thiết, dùng Polars cho nhanh
        print(f"Loading history from {path}...")
        try:
            col_name = "hist_ids"  # Đảm bảo đúng tên cột trong file parquet
            df = pl.read_parquet(path, columns=["user_id", col_name])

            # --- [SỬA ĐOẠN NÀY] ---
            # CŨ (Gây lỗi): df["col"] trả về Series, zip có thể vẫn giữ tính chất của Polars
            # history_map = dict(zip(df["user_id"], df[col_name]))

            # MỚI (Fix): Ép kiểu sang Python List bằng .to_list() trước khi zip
            history_map = dict(zip(df["user_id"].to_list(), df[col_name].to_list()))
            # ----------------------

            # Xóa dataframe ngay lập tức
            del df
            gc.collect()
            print(f"✅ Loaded history for {len(history_map)} users.")
            return history_map
        except Exception as e:
            print(f"⚠️ Error loading history: {e}")
            return {}

    def _process_batch(self, user_ids, clicked_batch, inview_batch):
        """Xử lý theo batch để tận dụng tốc độ của Numpy/List comprehension"""
        outputs = []

        # Zip 3 list lại để loop 1 lần duy nhất
        for uid, clicked, inview in zip(user_ids, clicked_batch, inview_batch):
            # 1. Xử lý History
            hist = self.user_history.get(uid, [])
            if len(hist) < self.history_len:
                hist = hist + [0] * (self.history_len - len(hist))
            else:
                hist = hist[-self.history_len:]

            # 2. Xử lý Candidate (Pos + Neg)
            if self.mode == 'train':
                # Lấy 1 Positive
                pos = np.random.choice(clicked) if clicked else (inview[0] if inview else 0)

                # Lấy K Negatives
                negs = list(set(inview) - set(clicked))
                if not negs:
                    negs = [pos] * self.neg_ratio
                elif len(negs) >= self.neg_ratio:
                    negs = np.random.choice(negs, self.neg_ratio, replace=False).tolist()
                else:
                    negs = (negs * (self.neg_ratio // len(negs) + 1))[:self.neg_ratio]

                cands = [pos] + negs
                labels = [1.0] + [0.0] * self.neg_ratio
            else:
                # Validation: Đơn giản hóa để chạy mượt
                pos = clicked[0] if clicked else (inview[0] if inview else 0)
                # Fill dummy negatives cho đủ shape model yêu cầu
                cands = [pos] + [pos] * self.neg_ratio
                labels = [1.0] + [0.0] * self.neg_ratio

            # 3. Map sang Index và đóng gói
            outputs.append({
                "hist_indices": torch.tensor(self.embedding_manager.map_ids_to_indices(hist), dtype=torch.long),
                "cand_indices": torch.tensor(self.embedding_manager.map_ids_to_indices(cands), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.float)
            })

        return outputs

    def _stream_data(self, worker_id, num_workers, rank, world_size):
        """Đọc file Parquet theo RowGroup (Không bao giờ đọc trùng lặp)"""
        pq_file = pq.ParquetFile(self.behaviors_path)

        # Chia việc: Worker này phụ trách những RowGroup nào?
        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        # Chỉ đọc các cột thật sự cần
        cols = ["user_id", "clk_ids", "inv_ids"]

        for i in range(pq_file.num_row_groups):
            # Nếu RowGroup này không thuộc về tôi -> Bỏ qua ngay
            if i % total_workers != global_worker_id:
                continue

            # Đọc và convert sang list python (nhanh hơn xử lý từng dòng object)
            table = pq_file.read_row_group(i, columns=cols)

            # Xử lý batch
            items = self._process_batch(
                table["user_id"].to_numpy(),
                table["clk_ids"].to_pylist(),
                table["inv_ids"].to_pylist()
            )

            yield from items

            # Dọn rác sau mỗi RowGroup
            del table, items
            # gc.collect() # Có thể uncomment nếu RAM quá thấp (nhưng sẽ làm chậm 1 chút)

    def __iter__(self):
        # Lấy thông tin worker hiện tại
        info = get_worker_info()
        wid = info.id if info else 0
        num_w = info.num_workers if info else 1

        # Lấy thông tin GPU/Distributed
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
            rank = dist.get_rank()
        else:
            ws, rank = 1, 0

        stream = self._stream_data(wid, num_w, rank, ws)

        # Shuffle Buffer đơn giản
        if self.shuffle_buffer_size > 0:
            buffer = []
            try:
                for _ in range(self.shuffle_buffer_size):
                    buffer.append(next(stream))
            except StopIteration:
                pass

            while buffer:
                try:
                    item = next(stream)
                    idx = np.random.randint(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = item
                except StopIteration:
                    np.random.shuffle(buffer)
                    yield from buffer
                    buffer = []
        else:
            yield from stream


# --- 3. LIGHTNING MODULE (Sạch sẽ) ---
class NAMLDataModule(L.LightningDataModule):
    def __init__(self, root_path, embedding_path, batch_size=32, history_len=30, num_workers=2,neg_ratio=4):
        super().__init__()
        self.save_hyperparameters()
        self.emb_manager = None

    def setup(self, stage=None):
        if not self.emb_manager:
            self.emb_manager = NewsEmbeddingManager(self.hparams.embedding_path)

        # Common config
        ds_args = {
            'embedding_manager': self.emb_manager,
            'history_len': self.hparams.history_len,
            'batch_size': self.hparams.batch_size,
        }

        if stage == 'fit' or stage is None:
            self.train_ds = NAMLIterableDataset(
                behaviors_path=Path(self.hparams.root_path) / "train" / "behaviors_processed.parquet",
                history_path=Path(self.hparams.root_path) / "train" / "history_processed.parquet",
                mode='train',
                neg_ratio =self.hparams.neg_ratio,
                **ds_args
            )
            self.val_ds = NAMLIterableDataset(
                behaviors_path=Path(self.hparams.root_path) / "validation" / "behaviors_processed.parquet",
                history_path=Path(self.hparams.root_path) / "validation" / "history_processed.parquet",
                mode='val',
                neg_ratio =self.hparams.neg_ratio,
                **ds_args
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True)