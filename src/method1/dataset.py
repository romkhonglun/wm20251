import torch
import polars as pl
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
import pytorch_lightning as L
import torch.distributed as dist
from itertools import islice
import os
import gc


# ==========================================
# 1. CẤU HÌNH & HELPER
# ==========================================
class WPMConfig:
    # WPM config cũ không còn dùng cho tính score nữa
    # nhưng vẫn giữ MAX_HISTORY để định hình ma trận
    MAX_HISTORY = 30


def build_history_matrix(history_path, articles_path, output_path):
    """
    Hàm tính toán Cache.
    Thay đổi: Tính log1p(hist_time) thay vì WPM Score.
    """
    print(f"🚀 [INIT] Đang tạo Ma trận Lịch sử (Log1p Time)...")

    # 1. Load Data (Bỏ qua body_len vì không cần dùng để tính time score nữa)
    # Tuy nhiên vẫn cần join article nếu bạn muốn lọc bài viết rác,
    # nhưng ở đây tôi tối giản chỉ lấy history.

    lf_hist = pl.scan_parquet(history_path).select([
        pl.col("user_id").cast(pl.Int32),
        pl.col("hist_ids"),
        pl.col("hist_scroll"),
        pl.col("hist_time"),
        pl.col("hist_ts")
    ])

    # 2. Explode
    q = lf_hist.explode(["hist_ids", "hist_scroll", "hist_time", "hist_ts"])
    q = q.with_columns(pl.col("hist_ids").cast(pl.Int32).alias("article_id_int"))

    # 3. Calculate Time Score (Log1p)
    # Xử lý null/nan/âm -> log1p
    # log1p(x) = log(x + 1) -> giúp thời gian 0s thành 0, phân phối mượt hơn
    time_col = pl.col("hist_time").fill_null(0).fill_nan(0).clip(0, None)
    score_expr = time_col.log1p()

    q = q.with_columns([
        score_expr.cast(pl.Float32).alias("time_log1p"),  # Đổi tên cột
        pl.col("hist_scroll").fill_null(0).cast(pl.Float32)
    ])

    # 4. Group & Slice (Tail 30)
    print("⏳ Đang gom nhóm và cắt dữ liệu (Taking last 30)...")
    df_grouped = (
        q.sort("hist_ts")
        .group_by("user_id")
        .agg([
            pl.col("article_id_int").tail(WPMConfig.MAX_HISTORY).alias("ids"),
            pl.col("hist_scroll").tail(WPMConfig.MAX_HISTORY).alias("scrolls"),
            pl.col("time_log1p").tail(WPMConfig.MAX_HISTORY).alias("scores")
            # Lưu vào alias scores để dùng lại logic dưới
        ])
        .collect(streaming=False)
    )

    # 5. Fill Matrix
    max_uid = df_grouped["user_id"].max()
    if max_uid is None: max_uid = 0
    num_users = max_uid + 1

    print(f"📦 Creating Matrix: [{num_users}, {WPMConfig.MAX_HISTORY}]")

    mat_ids = np.zeros((num_users, WPMConfig.MAX_HISTORY), dtype=np.int32)
    mat_scr = np.zeros((num_users, WPMConfig.MAX_HISTORY), dtype=np.float32)
    mat_sco = np.zeros((num_users, WPMConfig.MAX_HISTORY), dtype=np.float32)

    uids = df_grouped["user_id"].to_numpy()
    vals_ids = df_grouped["ids"].to_list()
    vals_scr = df_grouped["scrolls"].to_list()
    vals_sco = df_grouped["scores"].to_list()

    for i, uid in enumerate(uids):
        r_ids = vals_ids[i]
        length = len(r_ids)
        if length == 0: continue

        # Pre-padding logic (Điền vào cuối)
        mat_ids[uid, -length:] = r_ids
        mat_scr[uid, -length:] = vals_scr[i]
        mat_sco[uid, -length:] = vals_sco[i]

    print(f"💾 Saving cache: {output_path}")
    np.savez_compressed(
        output_path,
        matrix_ids=mat_ids,
        matrix_scrolls=mat_scr,
        matrix_scores=mat_sco  # Vẫn giữ tên key file là matrix_scores cho đồng bộ
    )

    del df_grouped, mat_ids, mat_scr, mat_sco
    gc.collect()
    print("✅ Cache built successfully!")


# ==========================================
# 2. DATASET (Reading Pre-computed Matrix)
# ==========================================
class NAMLIterableDataset(IterableDataset):
    def __init__(self, behaviors_path, cache_path,
                 neg_ratio=4, batch_size=32,
                 mode='train', shuffle_buffer_size=10000):
        super().__init__()
        self.behaviors_path = behaviors_path
        self.cache_path = cache_path
        self.neg_ratio = neg_ratio
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size if mode == 'train' else 0

        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(f"❌ Cache missing: {self.cache_path}")

        # Map file vào RAM
        data = np.load(self.cache_path, mmap_mode='r')
        self.mat_ids = data['matrix_ids']
        self.mat_scr = data['matrix_scrolls']
        self.mat_sco = data['matrix_scores']  # Đây bây giờ chứa log1p(time)
        self.num_users_cache = self.mat_ids.shape[0]

    def _get_user_history(self, user_id):
        if user_id >= self.num_users_cache:
            # Cold start user -> Trả về mảng 0
            return (
                np.zeros(WPMConfig.MAX_HISTORY, dtype=np.int32),
                np.zeros(WPMConfig.MAX_HISTORY, dtype=np.float32),
                np.zeros(WPMConfig.MAX_HISTORY, dtype=np.float32)
            )
        return (
            self.mat_ids[user_id],
            self.mat_scr[user_id],
            self.mat_sco[user_id]
        )

    def _process_batch(self, batch):
        u_ids = batch["user_id"]
        c_cols = batch["clk_ids"]
        i_cols = batch["inv_ids"]

        for i in range(len(batch)):
            uid = u_ids[i].as_py()
            c_list = c_cols[i].as_py() or []
            i_list = i_cols[i].as_py() or []

            # Negative Sampling Logic
            if self.mode == 'train' and c_list:
                pos_id = np.random.choice(c_list)
            elif c_list:
                pos_id = c_list[0]
            elif i_list:
                pos_id = i_list[0]
            else:
                continue

            neg_pool = list(set(i_list) - set(c_list))

            if len(neg_pool) >= self.neg_ratio:
                if self.mode == 'train':
                    neg_ids = np.random.choice(neg_pool, self.neg_ratio, replace=False).tolist()
                else:
                    neg_ids = neg_pool[:self.neg_ratio]
            else:
                neg_ids = neg_pool + [0] * (self.neg_ratio - len(neg_pool))

            cand_ids = [pos_id] + neg_ids
            labels = [1.0] + [0.0] * self.neg_ratio

            # Load History from Matrix
            h_ids, h_scr, h_sco = self._get_user_history(uid)

            # Yield sample
            yield {
                "hist_indices": torch.from_numpy(h_ids).long(),
                "hist_scroll": torch.from_numpy(h_scr).float(),
                # ĐỔI TÊN KEY Ở ĐÂY CHO RÕ NGHĨA
                "hist_time_log1p": torch.from_numpy(h_sco).float(),
                "cand_indices": torch.tensor(cand_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.float)
            }

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        np.random.seed(42 + worker_id)

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size, rank = 1, 0

        pq_file = pq.ParquetFile(self.behaviors_path)
        iter_batches = pq_file.iter_batches(batch_size=self.batch_size * 20)

        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id
        sharded_iter = islice(iter_batches, global_worker_id, None, total_workers)

        buffer = []
        for batch in sharded_iter:
            for sample in self._process_batch(batch):
                if self.shuffle_buffer_size > 0:
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield buffer.pop(np.random.randint(len(buffer)))
                else:
                    yield sample

        if self.shuffle_buffer_size > 0:
            np.random.shuffle(buffer)
            yield from buffer


# ==========================================
# 3. LIGHTNING DATA MODULE
# ==========================================
class NAMLDataModule(L.LightningDataModule):
    def __init__(self, root_path, batch_size=32, neg_ratio=4, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        self.root = Path(root_path)

    def prepare_data(self):
        train_hist = self.root / "train" / "history_processed.parquet"
        val_hist = self.root / "validation" / "history_processed.parquet"
        articles = self.root / "articles_processed.parquet"

        # Đổi tên file cache để tránh dùng nhầm cache cũ (chứa wpm_score)
        train_cache = self.root / "train_matrix_log1p.npz"
        val_cache = self.root / "val_matrix_log1p.npz"

        if not train_cache.exists():
            build_history_matrix(train_hist, articles, train_cache)

        if not val_cache.exists():
            build_history_matrix(val_hist, articles, val_cache)

    def setup(self, stage=None):
        train_beh = self.root / "train" / "behaviors_processed.parquet"
        val_beh = self.root / "validation" / "behaviors_processed.parquet"

        train_cache = self.root / "train_matrix_log1p.npz"
        val_cache = self.root / "val_matrix_log1p.npz"

        if stage == "fit" or stage is None:
            self.train_ds = NAMLIterableDataset(
                train_beh, train_cache,
                neg_ratio=self.hparams.neg_ratio,
                batch_size=self.hparams.batch_size,
                mode='train'
            )
            self.val_ds = NAMLIterableDataset(
                val_beh, val_cache,
                neg_ratio=self.hparams.neg_ratio,
                batch_size=self.hparams.batch_size,
                mode='val'
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)


# ==========================================
# 4. MAIN execution (TESTING)
# ==========================================
if __name__ == "__main__":
    # Cập nhật hàm in để match với key mới
    def pretty_print_sample(batch, sample_idx=0):
        h_ids = batch['hist_indices'][sample_idx].tolist()
        h_scr = batch['hist_scroll'][sample_idx].tolist()
        # Key mới: hist_time_log1p
        h_sco = batch['hist_time_log1p'][sample_idx].tolist()

        c_ids = batch['cand_indices'][sample_idx].tolist()
        lbls = batch['labels'][sample_idx].tolist()

        display_len = 10
        print(f"   👤 [SAMPLE {sample_idx}] USER CONTEXT (Last {display_len}/{len(h_ids)} items):")
        print(f"      ORDER :  " + "  ".join([f"{i:>5}" for i in range(len(h_ids) - display_len, len(h_ids))]))
        print(f"      🆔 ID :  " + "  ".join([f"{x:>5}" for x in h_ids[-display_len:]]))
        print(f"      📜 SCR:  " + "  ".join([f"{x:>5.1f}" for x in h_scr[-display_len:]]))
        # In tiêu đề là LOG1P
        print(f"      🕒 LOG:  " + "  ".join([f"{x:>5.2f}" for x in h_sco[-display_len:]]))

        print(f"\n   🎯 TARGET (Candidates & Labels):")
        print(f"      🆔 CAND: " + "  ".join([f"{c:>7}" for c in c_ids]))
        print(f"      🏷️ LABL: " + "  ".join([f"{l:>7.0f}" for l in lbls]))

    # --- CHẠY THỬ ---
    # DATA_ROOT = "/path/to/your/data"
    # BATCH_SIZE = 4
    # ... (Phần chạy thử giữ nguyên logic, chỉ cần gọi đúng hàm in) ...