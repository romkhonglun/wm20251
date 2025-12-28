import sys
import os
import argparse
import numpy as np
import torch
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm
import zipfile
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightning_module import NAMLLightningModule
from model import TIME_FEATURE_NAMLConfig
from dataset import NewsBaseLogic, StaticArticleFeatures, NewsEmbeddingManager


# --- HÀM HỖ TRỢ ---
def calculate_ranks(scores):
    scores = np.array(scores)
    indices_desc = np.argsort(-scores)
    ranks = np.empty_like(indices_desc)
    ranks[indices_desc] = np.arange(len(scores)) + 1
    return ranks.tolist()


def write_submission(predictions, output_path):
    print(f"📦 Đang nén file submission vào: {output_path}")
    lines = []
    for imp_id, ranks in predictions:
        rank_str = "[" + ",".join(map(str, ranks)) + "]"
        lines.append(f"{imp_id} {rank_str}")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("predictions.txt", "\n".join(lines))


# --- DATASET CHO TEST METHOD 2 ---
class Method2TestDataset(IterableDataset, NewsBaseLogic):
    def __init__(self, behaviors_path, history_path, article_features, embedding_manager, history_len=30):
        # Init logic cơ bản từ NewsBaseLogic, neg_ratio=0 vì là test
        self._init_base(history_path, article_features, embedding_manager, history_len, neg_ratio=0)
        self.behaviors_path = behaviors_path

    def __iter__(self):
        pf = pq.ParquetFile(self.behaviors_path)
        for batch in pf.iter_batches(batch_size=2048):
            batch_dict = batch.to_pydict()

            # Map cột test set (article_ids_inview -> inv_ids)
            if 'article_ids_inview' in batch_dict:
                batch_dict['inv_ids'] = batch_dict.pop('article_ids_inview')

            keys = list(batch_dict.keys())
            batch_len = len(batch_dict[keys[0]])

            for i in range(batch_len):
                row = {k: batch_dict[k][i] for k in keys}
                imp_id = row.get("impression_id", 0)

                # Logic xử lý row cho test (khác train là ko có clk_ids và label)
                processed = self._process_row_test(row)
                if processed:
                    processed["impression_id"] = imp_id
                    yield processed

    def _process_row_test(self, row):
        try:
            user_id = int(row["user_id"])
        except:
            return None

        imp_ts = row.get("imp_ts", 0.0) or 0.0

        # 1. Truy xuất History
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

        # 2. Candidates (Chỉ Inview)
        candidate_ids = row.get("inv_ids", [])
        if not candidate_ids: return None

        cand_nums, cand_cats = self.art_feats.get(candidate_ids)
        # Tính feature recency
        cand_nums[:, 4] = np.log1p(np.abs(imp_ts - np.nan_to_num(cand_nums[:, 4], nan=imp_ts)) / 3600.0)

        # Sim (Tạm để 0 cho test speed, model sẽ tự tính lại nếu cần hoặc bỏ qua)
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


def collate_fn_test(batch):
    imp_ids = [item['impression_id'] for item in batch]

    # Stack features cố định
    h_idx = torch.stack([item['hist_indices'] for item in batch])
    h_scr = torch.stack([item['hist_scroll'] for item in batch])
    h_tm = torch.stack([item['hist_time'] for item in batch])
    h_dif = torch.stack([item['hist_diff'] for item in batch])
    imp_f = torch.stack([item['imp_feats'] for item in batch])

    # Pad features thay đổi theo số lượng candidates
    c_idx = pad_sequence([item['cand_indices'] for item in batch], batch_first=True, padding_value=0)
    c_num = pad_sequence([item['cand_num'] for item in batch], batch_first=True, padding_value=0)
    c_cat = pad_sequence([item['cand_cat'] for item in batch], batch_first=True, padding_value=0)

    return {
        "impression_ids": imp_ids,
        "hist_indices": h_idx,
        "hist_scroll": h_scr,
        "hist_time": h_tm,
        "hist_diff": h_dif,
        "imp_feats": imp_f,
        "cand_indices": c_idx,
        "cand_num": c_num,
        "cand_cat": c_cat,
        "cand_lens": [len(item['cand_indices']) for item in batch]
    }


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Test Script for Method 2")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--articles-path", type=str,
                        default="/home2/congnh/wm/processed_parquet/articles_processed.parquet")
    parser.add_argument("--embedding-path", type=str, default="/home2/congnh/wm/embedding/body_emb.npy")
    parser.add_argument("--output", type=str, default="submission_method2.zip")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    print("🔄 Đang load model Method 2...")
    model = NAMLLightningModule.load_from_checkpoint(
        args.checkpoint,
        config=TIME_FEATURE_NAMLConfig(),
        embedding_path=args.embedding_path,
        use_compile=False,  # Tắt compile khi inference để tránh lỗi shape động
        map_location=device
    )
    model.eval()
    model.to(device)

    # 2. Load Static Features & Dataset
    print("📚 Đang load Static Features...")
    art_feats = StaticArticleFeatures(args.articles_path)
    emb_manager = NewsEmbeddingManager(args.embedding_path)

    beh_path = os.path.join(args.test_dir,"test", "behaviors_processed.parquet")
    hist_path = os.path.join(args.test_dir,"test", "history_processed.parquet")

    dataset = Method2TestDataset(beh_path, hist_path, art_feats, emb_manager)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_test)

    # 3. Inference
    predictions = []
    print("🚀 Bắt đầu dự đoán...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move to GPU (trừ impression_ids và cand_lens)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            output = model(batch)
            scores = output["preds"].cpu().numpy()

            for i, imp_id in enumerate(batch["impression_ids"]):
                valid_len = batch["cand_lens"][i]
                valid_scores = scores[i, :valid_len]
                ranks = calculate_ranks(valid_scores)
                predictions.append((imp_id, ranks))

    write_submission(predictions, args.output)


if __name__ == "__main__":
    main()