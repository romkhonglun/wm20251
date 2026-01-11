import sys
import os
import argparse
import torch
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import importlib.util
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence

# --- CẤU HÌNH MẶC ĐỊNH ---
DEFAULT_VAL_DIR = "processed_parquet_large"  # Folder chứa folder 'val'
DEFAULT_EMBEDDING_DIR = "embedding_large"
DEFAULT_EMBEDDING_PATH = "embedding_large/body_emb.npy"
DEFAULT_ARTICLES_PATH = "processed_parquet_large/articles_processed.parquet"

# Đường dẫn checkpoint (SỬA LẠI CHO ĐÚNG)
CKPT_BASELINE = "NewsRecSys/wbbqb9vb/checkpoints/Baseline-epoch=04-val/auc=0.5749.ckpt"
CKPT_METHOD1 = "checkpoints/method1/naml-m1-epoch=04-val/auc=0.6249.ckpt"
CKPT_METHOD2 = "auc=0.7362.ckpt"


# --- HELPER: LOAD MODULE ---
def load_method_module(name, path):
    for module_name in ['model', 'dataset', 'utils', 'lightning_module']:
        if module_name in sys.modules: del sys.modules[module_name]
    module_dir = os.path.dirname(os.path.abspath(path))
    if module_dir not in sys.path: sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- DATASETS ---

# 1. BASELINE DATASET
class BaselineValDataset(IterableDataset):
    def __init__(self, behaviors_path, compact_history, max_hist_len=50):
        self.behaviors_path = behaviors_path
        self.compact_history = compact_history
        self.max_hist_len = max_hist_len

    def __iter__(self):
        pf = pq.ParquetFile(self.behaviors_path)
        for batch in pf.iter_batches(batch_size=1024):
            df = pl.from_arrow(batch)

            # 1. Lấy Inview IDs
            inv_col = 'article_ids_inview' if 'article_ids_inview' in df.columns else 'inv_ids'
            inv_lists = df[inv_col].to_list()

            # 2. Lấy User & Impression
            user_ids = df['user_id'].to_list()
            imp_ids = df['impression_id'].to_list() if 'impression_id' in df.columns else range(len(df))

            # 3. Lấy Click IDs (Target Label)
            # Ưu tiên cột có sẵn, nếu không thì tính từ labels
            if 'clk_ids' in df.columns:
                clk_ids_list = df['clk_ids'].to_list()
            elif 'article_ids_clicked' in df.columns:
                clk_ids_list = df['article_ids_clicked'].to_list()
            elif 'labels' in df.columns:
                # Fallback: Tự tính clk_ids từ inview và labels
                lbl_lists = df['labels'].to_list()
                clk_ids_list = []
                for idx, lbls in enumerate(lbl_lists):
                    # Lấy article id tại vị trí label == 1
                    c_ids = [inv_lists[idx][i] for i, l in enumerate(lbls) if l == 1.0]
                    clk_ids_list.append(c_ids)
            else:
                clk_ids_list = [[] for _ in range(len(df))]

            for i in range(len(df)):
                hist_ids = self.compact_history.get_history(user_ids[i])
                yield {
                    "impression_id": imp_ids[i],
                    "hist_ids": hist_ids[:self.max_hist_len],
                    "candidate_ids": np.array(inv_lists[i], dtype=np.int32),
                    "clk_ids": clk_ids_list[i]  # Chỉ cần cái này để in ra
                }


def collate_fn_base_val(batch):
    base = [torch.from_numpy(item['hist_ids']).long() for item in batch]
    cands = [torch.from_numpy(item['candidate_ids']).long() for item in batch]

    return {
        "impression_ids": [item['impression_id'] for item in batch],
        "hist_ids": pad_sequence(base, batch_first=True, padding_value=0),
        "candidate_ids": pad_sequence(cands, batch_first=True, padding_value=0),
        "cand_lens": [len(c) for c in cands],
        "raw_cands": [item['candidate_ids'] for item in batch],
        "clk_ids": [item['clk_ids'] for item in batch]  # Pass through
    }


# 2. METHOD 1 DATASET
def create_method1_dataset(pkg, beh_path, cache_path, batch_size):
    class Method1ValDataset(pkg.NAMLIterableDataset):
        def __init__(self, behaviors_path, cache_path, batch_size):
            super().__init__(behaviors_path, cache_path, neg_ratio=0, batch_size=batch_size, mode='val')

        def _process_batch(self, batch):
            u_ids = batch["user_id"]
            i_cols = batch["article_ids_inview"] if "article_ids_inview" in batch.column_names else batch["inv_ids"]
            imp_ids = batch["impression_id"] if "impression_id" in batch.column_names else [0] * len(u_ids)

            # Xử lý clk_ids
            if "clk_ids" in batch.column_names:
                c_cols = batch["clk_ids"]
            elif "article_ids_clicked" in batch.column_names:
                c_cols = batch["article_ids_clicked"]
            elif "labels" in batch.column_names:
                # Nếu chỉ có labels, Method 1 dataset gốc khó xử lý ở đây vì nó trả về tensor
                # Ta sẽ lấy labels để truyền ra ngoài
                c_cols = batch["labels"]
            else:
                c_cols = None

            for i in range(len(batch)):
                uid = u_ids[i].as_py()
                i_list = i_cols[i].as_py() or []
                imp_id = imp_ids[i].as_py()

                # Logic lấy clk_ids đơn giản
                if c_cols:
                    raw_val = c_cols[i].as_py()
                    # Nếu là list int (clk_ids)
                    if raw_val and isinstance(raw_val[0], int):
                        clk_ids = raw_val
                    # Nếu là list float/int (labels)
                    elif raw_val and (isinstance(raw_val[0], float) or isinstance(raw_val[0], int)):
                        clk_ids = [i_list[k] for k, l in enumerate(raw_val) if l == 1]
                    else:
                        clk_ids = []
                else:
                    clk_ids = []

                h_ids, h_scr, h_sco = self._get_user_history(uid)

                yield {
                    "impression_id": imp_id,
                    "hist_indices": torch.from_numpy(h_ids).long(),
                    "hist_scroll": torch.from_numpy(h_scr).float(),
                    "hist_time_log1p": torch.from_numpy(h_sco).float(),
                    "cand_indices": torch.tensor(i_list, dtype=torch.long),
                    "clk_ids": clk_ids
                }

    return Method1ValDataset(beh_path, cache_path, batch_size)


# 3. METHOD 2 DATASET (Fix lỗi _process_row_test)
def create_method2_dataset(pkg, beh_path, hist_path, art_feats, emb_manager, batch_size):
    class Method2ValDataset(IterableDataset, pkg.NewsBaseLogic):
        def __init__(self, behaviors_path, history_path, article_features, embedding_manager, history_len=30):
            self._init_base(history_path, article_features, embedding_manager, history_len, neg_ratio=0)
            self.behaviors_path = behaviors_path
            self.history_len = history_len

        def __iter__(self):
            pf = pq.ParquetFile(self.behaviors_path)
            for batch in pf.iter_batches(batch_size=batch_size):
                batch_dict = batch.to_pydict()
                if 'article_ids_inview' in batch_dict: batch_dict['inv_ids'] = batch_dict.pop('article_ids_inview')

                keys = list(batch_dict.keys())
                for i in range(len(batch_dict[keys[0]])):
                    row = {k: batch_dict[k][i] for k in keys}

                    # Gọi hàm xử lý đã fix bên dưới
                    processed = self._process_row_test_fixed(row)
                    if processed:
                        processed["impression_id"] = row.get("impression_id", 0)

                        # Xử lý lấy clk_ids từ row
                        if "clk_ids" in row:
                            processed["clk_ids"] = row["clk_ids"]
                        elif "article_ids_clicked" in row:
                            processed["clk_ids"] = row["article_ids_clicked"]
                        elif "labels" in row and "inv_ids" in row:
                            processed["clk_ids"] = [row["inv_ids"][k] for k, l in enumerate(row["labels"]) if l == 1]
                        else:
                            processed["clk_ids"] = []

                        yield processed

        # [FIX] Tự định nghĩa hàm này để không phụ thuộc vào Method2TestDataset
        def _process_row_test_fixed(self, row):
            try:
                user_id = int(row["user_id"])
            except:
                return None

            imp_ts = row.get("imp_ts", 0.0) or 0.0

            # 1. History
            if user_id >= self.hist_ids_mat.shape[0]:
                h_ids = np.zeros(self.history_len, dtype=np.int32)
                h_scr = np.zeros(self.history_len)
                h_tm = np.zeros(self.history_len)
                ts_diff_log = np.zeros(self.history_len, dtype=np.float32)
                curr_len = 0
            else:
                h_ids = self.hist_ids_mat[user_id]
                h_scr = self.hist_scr_mat[user_id]
                h_tm = self.hist_tm_mat[user_id]
                h_ts = self.hist_ts_mat[user_id]
                curr_len = self.hist_lens[user_id]

                ts_diff_log = np.zeros(self.history_len, dtype=np.float32)
                if curr_len > 0:
                    diffs = (imp_ts - h_ts[:curr_len]) / 3600.0
                    ts_diff_log[:curr_len] = np.log1p(np.clip(diffs, 0, None))

            # 2. Candidates
            candidate_ids = row.get("inv_ids", [])
            if not candidate_ids: return None

            cand_nums, cand_cats = self.art_feats.get(candidate_ids)
            cand_nums[:, 4] = np.log1p(np.abs(imp_ts - np.nan_to_num(cand_nums[:, 4], nan=imp_ts)) / 3600.0)
            scores = np.zeros((len(candidate_ids), 1), dtype=np.float32)

            return {
                "hist_indices": torch.from_numpy(h_ids.astype(np.int64)),
                "hist_scroll": torch.from_numpy(h_scr),
                "hist_time": torch.from_numpy(h_tm),
                "hist_diff": torch.from_numpy(ts_diff_log),
                "cand_indices": torch.tensor(candidate_ids, dtype=torch.long),
                "cand_num": torch.from_numpy(cand_nums),
                "cand_cat": torch.from_numpy(cand_cats).long(),
                "cand_sim": torch.from_numpy(np.nan_to_num(scores, 0.0)).float(),
                "imp_feats": torch.tensor(
                    [np.log1p(curr_len), (imp_ts % 86400) / 86400.0, float(row.get("norm_age", 0.0) or 0.0)],
                    dtype=torch.float)
            }

    return Method2ValDataset(beh_path, hist_path, art_feats, emb_manager, 30)


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {device}")

    beh_path = os.path.join(args.val_dir, "val", "behaviors_processed.parquet")
    if not os.path.exists(beh_path): beh_path = os.path.join(args.val_dir, "validation", "behaviors_processed.parquet")
    hist_path = beh_path.replace("behaviors", "history")

    # 1. BASELINE
    print("\n🔵 Loading BASELINE...")
    try:
        base_pkg = load_method_module("baseline_pkg", "src/baseline/test.py")
        model_base = base_pkg.NAMLLightningModule.load_from_checkpoint(
            CKPT_BASELINE, config=base_pkg.VariantNAMLConfig(), embedding_dir=DEFAULT_EMBEDDING_DIR, map_location=device
        ).eval().to(device)
        ds_base = BaselineValDataset(beh_path, base_pkg.CompactHistory(hist_path))
        dl_base = DataLoader(ds_base, batch_size=args.samples, collate_fn=collate_fn_base_val)
    except Exception as e:
        print(f"❌ Error Baseline: {e}");
        return

    # 2. METHOD 1
    print("🟢 Loading METHOD 1...")
    try:
        m1_pkg = load_method_module("method1_pkg", "src/method1/test.py")
        model_m1 = m1_pkg.NAMLLightningModule.load_from_checkpoint(
            CKPT_METHOD1, config=m1_pkg.UnifiedConfig(), embedding_dir=DEFAULT_EMBEDDING_DIR, map_location=device
        ).eval().to(device)
        cache_path = os.path.join(args.val_dir, "val_matrix_cache.npz")
        if not os.path.exists(cache_path): m1_pkg.build_history_matrix(hist_path, DEFAULT_ARTICLES_PATH, cache_path)
        ds_m1 = create_method1_dataset(m1_pkg, beh_path, cache_path, args.samples)
        dl_m1 = DataLoader(ds_m1, batch_size=args.samples, collate_fn=m1_pkg.collate_fn_test)
    except Exception as e:
        print(f"❌ Error Method 1: {e}");
        return

    # 3. METHOD 2
    print("🟠 Loading METHOD 2...")
    try:
        m2_pkg = load_method_module("method2_pkg", "src/method2/test.py")
        model_m2 = m2_pkg.NAMLLightningModule.load_from_checkpoint(
            CKPT_METHOD2, config=m2_pkg.TIME_FEATURE_NAMLConfig(), embedding_path=DEFAULT_EMBEDDING_PATH,
            use_compile=False, map_location=device
        ).eval().to(device)
        art_feats = m2_pkg.StaticArticleFeatures(DEFAULT_ARTICLES_PATH)
        emb_manager = m2_pkg.NewsEmbeddingManager(DEFAULT_EMBEDDING_PATH)
        ds_m2 = create_method2_dataset(m2_pkg, beh_path, hist_path, art_feats, emb_manager, args.samples)
        dl_m2 = DataLoader(ds_m2, batch_size=args.samples, collate_fn=m2_pkg.collate_fn_test)
    except Exception as e:
        print(f"❌ Error Method 2: {e}");
        return

    # 4. RUN
    print("\n" + "=" * 80)
    print(f"🚀 DEMO COMPARISON (Ground Truth: clk_ids)")
    print("=" * 80)

    try:
        batch_b = next(iter(dl_base))
        batch_m1 = next(iter(dl_m1))
        batch_m2 = next(iter(dl_m2))
    except Exception as e:
        print(f"❌ Error fetching batch: {e}");
        return

    with torch.no_grad():
        inp_b = {"hist_ids": batch_b["hist_ids"].to(device), "candidate_ids": batch_b["candidate_ids"].to(device)}
        sc_b = model_base(inp_b).cpu().numpy() if not isinstance(model_base(inp_b), dict) else model_base(inp_b)[
            "preds"].cpu().numpy()

        for k in ["hist_indices", "hist_scroll", "hist_time_log1p", "cand_indices"]: batch_m1[k] = batch_m1[k].to(
            device)
        sc_m1 = model_m1(batch_m1)["preds"].cpu().numpy()

        for k, v in batch_m2.items():
            if isinstance(v, torch.Tensor): batch_m2[k] = v.to(device)
        sc_m2 = model_m2(batch_m2)["preds"].cpu().numpy()

    # PRINT RESULT
    ids = batch_b['impression_ids']
    for i, imp_id in enumerate(ids):
        valid_len = batch_b["cand_lens"][i]
        cands = batch_b["raw_cands"][i]
        # Lấy clk_ids trực tiếp từ batch
        clk_ids = batch_b["clk_ids"][i]

        # Nếu clk_ids là tensor hoặc array, convert sang list
        if hasattr(clk_ids, 'tolist'): clk_ids = clk_ids.tolist()

        print(f"📌 Impression ID: {imp_id}")
        print(f"🎯 GT Clicked IDs: {clk_ids}")
        print("-" * 60)

        def print_rank(name, scores):
            scores = scores[:valid_len]
            # Rank: 1 = cao nhất
            ranks = np.argsort(np.argsort(-scores)) + 1

            print(f"  > {name}:")
            # In rank của các bài đã click (nếu có)
            found_any = False
            for cid in clk_ids:
                # Tìm index của clk_id trong list candidates
                indices = np.where(cands == cid)[0]
                if len(indices) > 0:
                    idx = indices[0]
                    rank = ranks[idx]
                    mark = "🌟" if rank <= 5 else ""
                    print(f"    - Clicked Item {cid}: Rank {rank:>2}/{valid_len} {mark}")
                    found_any = True

            if not found_any and len(clk_ids) > 0:
                print(f"    - (Không tìm thấy Clicked Item trong Candidate list?)")

            # In top 3 gợi ý
            top3 = cands[np.argsort(-scores)[:3]]
            print(f"    - Top 3 Recs: {top3.tolist()}")

        print_rank("Baseline", sc_b[i])
        print_rank("Method 1", sc_m1[i])
        print_rank("Method 2", sc_m2[i])
        print("=" * 80)


if __name__ == "__main__":
    main()