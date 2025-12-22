import torch
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import polars as pl_lib
import pyarrow.parquet as pq
import numpy as np
import torch.distributed as dist
import random
import gc
import os


# ==========================================
# 1. COMPACT HISTORY (Memory Efficient)
# ==========================================
class CompactHistory:
    """
    Load history dạng CSR (Compressed Sparse Row) để tiết kiệm RAM tối đa.
    """

    def __init__(self, history_path):
        if not os.path.exists(history_path):
            raise FileNotFoundError(f"History file not found: {history_path}")

        print(f"Loading history from {history_path}...")
        df = pl_lib.read_parquet(history_path)

        # Sort theo user_id để đảm bảo thứ tự mapping
        df = df.sort("user_id")

        users = df["user_id"].to_numpy()
        histories = df["hist_ids"].to_list()

        # Mapping User ID -> Index trong mảng offsets
        self.user_map = {u: i for i, u in enumerate(users)}

        # Flatten toàn bộ history thành 1 mảng int32 duy nhất
        self.values = np.concatenate([np.array(h, dtype=np.int32) for h in histories])

        # Tạo mảng offsets
        lens = np.array([len(h) for h in histories], dtype=np.int32)
        self.offsets = np.zeros(len(lens) + 1, dtype=np.int32)
        self.offsets[1:] = np.cumsum(lens)

        print(f" -> Loaded {len(self.user_map)} users. Total clicks: {len(self.values)}")

        del df, histories, users, lens
        gc.collect()

    def get_history(self, user_id):
        idx = self.user_map.get(user_id)
        if idx is None:
            return np.array([], dtype=np.int32)

        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        return self.values[start:end]


# ==========================================
# 2. ITERABLE DATASET
# ==========================================
class NewsRecIterableDataset(IterableDataset):
    def __init__(self,
                 behaviors_path: str,
                 compact_history: CompactHistory,
                 npratio: int = 4,
                 batch_size_parquet: int = 4096,
                 buffer_size: int = 5000,
                 mode: str = 'train'):

        if not os.path.exists(behaviors_path):
            raise FileNotFoundError(f"Behaviors file not found: {behaviors_path}")

        self.behaviors_path = behaviors_path
        self.compact_history = compact_history
        self.npratio = npratio
        self.batch_size_parquet = batch_size_parquet
        self.buffer_size = buffer_size
        self.mode = mode

    def _get_worker_info(self):
        rank = 0
        world_size = 1
        worker_id = 0
        num_workers = 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        return rank, world_size, worker_id, num_workers

    def __iter__(self):
        rank, world_size, worker_id, num_workers = self._get_worker_info()
        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        parquet_file = pq.ParquetFile(self.behaviors_path)
        buffer = []

        # Đọc từng batch từ file parquet (tránh load hết vào RAM)
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=self.batch_size_parquet)):
            # Sharding: Skip các batch không thuộc worker này
            if batch_idx % total_workers != global_worker_id:
                continue

            batch_dict = batch.to_pydict()
            users = batch_dict['user_id']
            inv_lists = batch_dict['inv_ids']
            clk_lists = batch_dict['clk_ids']

            for i in range(len(users)):
                user_id = users[i]
                hist_ids = self.compact_history.get_history(user_id)

                inv_set = set(inv_lists[i])
                clk_set = set(clk_lists[i])

                # --- TRAIN MODE ---
                if self.mode == 'train':
                    if not clk_set: continue
                    neg_candidates = list(inv_set - clk_set)

                    for pos_id in clk_set:
                        # Negative Sampling
                        if not neg_candidates:
                            selected_negs = [pos_id] * self.npratio
                        elif len(neg_candidates) >= self.npratio:
                            selected_negs = random.sample(neg_candidates, self.npratio)
                        else:
                            selected_negs = random.choices(neg_candidates, k=self.npratio)

                        candidate_ids = [pos_id] + selected_negs

                        buffer.append({
                            "hist_ids": hist_ids,
                            "candidate_ids": np.array(candidate_ids, dtype=np.int32),
                            "label": [1.0]+[0.0]*self.npratio
                        })

                # --- VAL MODE ---
                else:
                    # Giữ nguyên toàn bộ inview
                    candidate_ids = list(inv_lists[i])
                    labels = [1 if doc_id in clk_set else 0 for doc_id in candidate_ids]

                    yield {
                        "hist_ids": hist_ids,
                        "candidate_ids": np.array(candidate_ids, dtype=np.int32),
                        "labels": np.array(labels, dtype=np.float32)
                    }

            # Shuffle buffer (chỉ cho Train)
            if self.mode == 'train':
                if len(buffer) >= self.buffer_size:
                    random.shuffle(buffer)
                    for _ in range(len(buffer)):
                        yield buffer.pop()

        # Flush buffer cuối cùng
        if self.mode == 'train' and buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield item


# ==========================================
# 3. LIGHTNING DATA MODULE (Updated Logic)
# ==========================================
class NewsRecDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_data_dir: str,  # Chỉ cần path tới folder cha
                 batch_size: int = 32,
                 npratio: int = 4,
                 num_workers: int = 2,
                 max_hist_len: int = 50):
        super().__init__()
        self.save_hyperparameters()
        self.root = root_data_dir
        self.batch_size = batch_size
        self.npratio = npratio
        self.num_workers = num_workers
        self.max_hist_len = max_hist_len

        # Placeholder cho history objects
        self.train_history_struct = None
        self.val_history_struct = None

    def setup(self, stage=None):
        """
        Tự động map path dựa trên cấu trúc thư mục:
        root/
          train/
            behaviors_processed.parquet
            history_processed.parquet
          validation/
            behaviors_processed.parquet
            history_processed.parquet
        """
        # Xây dựng đường dẫn
        train_hist_path = os.path.join(self.root, "train", "history_processed.parquet")
        val_hist_path = os.path.join(self.root, "validation", "history_processed.parquet")

        # Load History vào Shared Memory (Main Process)
        # Chúng ta load riêng Train History và Val History vì folder tách biệt
        if self.train_history_struct is None:
            print("--- Setup Train Data ---")
            self.train_history_struct = CompactHistory(train_hist_path)

        if self.val_history_struct is None:
            print("--- Setup Validation Data ---")
            self.val_history_struct = CompactHistory(val_hist_path)

    def train_dataloader(self):
        train_behaviors = os.path.join(self.root, "train", "behaviors_processed.parquet")

        dataset = NewsRecIterableDataset(
            behaviors_path=train_behaviors,
            compact_history=self.train_history_struct,  # Truyền train history
            npratio=self.npratio,
            mode='train',
            buffer_size=5000
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self.collate_fn_train, pin_memory=True, persistent_workers=True
        )

    def val_dataloader(self):
        val_behaviors = os.path.join(self.root, "validation", "behaviors_processed.parquet")

        dataset = NewsRecIterableDataset(
            behaviors_path=val_behaviors,
            compact_history=self.val_history_struct,  # Truyền val history
            mode='val'
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self.collate_fn_val, pin_memory=True
        )

    # --- Collate Functions ---
    def collate_fn_train(self, batch):
        hist_ids = [torch.from_numpy(item['hist_ids'][:self.max_hist_len]).long() for item in batch]
        candidate_ids = [torch.from_numpy(item['candidate_ids']).long() for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        hist_padded = pad_sequence(hist_ids, batch_first=True, padding_value=0)
        candidate_tensor = torch.stack(candidate_ids)

        return {
            "hist_ids": hist_padded,
            "candidate_ids": candidate_tensor,
            "labels": labels
        }

    def collate_fn_val(self, batch):
        hist_ids = [torch.from_numpy(item['hist_ids'][:self.max_hist_len]).long() for item in batch]
        cands = [torch.from_numpy(item['candidate_ids']).long() for item in batch]
        lbls = [torch.from_numpy(item['labels']).float() for item in batch]

        hist_padded = pad_sequence(hist_ids, batch_first=True, padding_value=0)
        cands_padded = pad_sequence(cands, batch_first=True, padding_value=0)
        # Padding label bằng -1 để mask sau này
        lbls_padded = pad_sequence(lbls, batch_first=True, padding_value=-1)

        return {
            "hist_ids": hist_padded,
            "candidate_ids": cands_padded,
            "labels": lbls_padded
        }