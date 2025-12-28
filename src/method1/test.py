import sys
import os
import argparse
import numpy as np
import torch
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm
import zipfile
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Thêm đường dẫn để import các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightning_module import NAMLLightningModule
from model import UnifiedConfig
from dataset import build_history_matrix, NAMLIterableDataset


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


# --- DATASET CHO TEST METHOD 1 ---
class Method1TestDataset(NAMLIterableDataset):
    def __init__(self, behaviors_path, cache_path, batch_size=32):
        # Override init để set mode='test' và tắt negative sampling
        super().__init__(behaviors_path, cache_path, neg_ratio=0, batch_size=batch_size, mode='test')

    def _process_batch(self, batch):
        u_ids = batch["user_id"]

        # Xử lý tên cột linh hoạt cho tập Test
        if "article_ids_inview" in batch.column_names:
            i_cols = batch["article_ids_inview"]
        elif "inv_ids" in batch.column_names:
            i_cols = batch["inv_ids"]
        else:
            # Fallback nếu không tìm thấy cột (dù hiếm khi xảy ra)
            print("⚠️ Warning: Không tìm thấy cột inview, skipping batch...")
            return

        if "impression_id" in batch.column_names:
            imp_ids = batch["impression_id"]
        else:
            imp_ids = [0] * len(u_ids)

        for i in range(len(batch)):
            uid = u_ids[i].as_py()
            i_list = i_cols[i].as_py() or []
            imp_id = imp_ids[i].as_py()

            h_ids, h_scr, h_sco = self._get_user_history(uid)

            yield {
                "impression_id": imp_id,
                "hist_indices": torch.from_numpy(h_ids).long(),
                "hist_scroll": torch.from_numpy(h_scr).float(),
                "hist_time_log1p": torch.from_numpy(h_sco).float(),
                "cand_indices": torch.tensor(i_list, dtype=torch.long),
            }


def collate_fn_test(batch):
    # Batch ở đây là một list các dictionary
    imp_ids = [item['impression_id'] for item in batch]

    hist_idx = pad_sequence([item['hist_indices'] for item in batch], batch_first=True, padding_value=0)
    hist_scr = pad_sequence([item['hist_scroll'] for item in batch], batch_first=True, padding_value=0)
    hist_tim = pad_sequence([item['hist_time_log1p'] for item in batch], batch_first=True, padding_value=0)
    cand_idx = pad_sequence([item['cand_indices'] for item in batch], batch_first=True, padding_value=0)

    return {
        "impression_ids": imp_ids,
        "hist_indices": hist_idx,
        "hist_scroll": hist_scr,
        "hist_time_log1p": hist_tim,
        "cand_indices": cand_idx,
        "cand_lens": [len(item['cand_indices']) for item in batch]
    }


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Test Script for Method 1")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--articles-path", type=str,
                        default="/home2/congnh/wm/processed_parquet/articles_processed.parquet")
    parser.add_argument("--embedding-dir", type=str, default="/home2/congnh/wm/embedding")
    parser.add_argument("--output", type=str, default="submission_method1.zip")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Tạo Cache Matrix cho Test nếu chưa có
    test_hist_path = os.path.join(args.test_dir,"test", "history_processed.parquet")
    cache_path = os.path.join(args.test_dir, "test_matrix_log1p.npz")

    if not os.path.exists(cache_path):
        print(f"⚠️ Chưa thấy cache test tại {cache_path}, đang tạo mới...")
        build_history_matrix(test_hist_path, args.articles_path, cache_path)
    else:
        print(f"✅ Đã tìm thấy cache: {cache_path}")

    # 2. Load Model
    print("🔄 Đang load model Method 1...")
    model = NAMLLightningModule.load_from_checkpoint(
        args.checkpoint,
        config=UnifiedConfig(),
        embedding_dir=args.embedding_dir,
        map_location=device
    )
    model.eval()
    model.to(device)

    # 3. Load Data
    beh_path = os.path.join(args.test_dir,"test", "behaviors_processed.parquet")
    dataset = Method1TestDataset(beh_path, cache_path, batch_size=args.batch_size)

    # [FIX QUAN TRỌNG]: Dùng args.batch_size thay vì None để DataLoader gom batch đúng cách
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_test)

    # 4. Inference
    predictions = []
    print("🚀 Bắt đầu dự đoán...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move tensors to device
            for k in ["hist_indices", "hist_scroll", "hist_time_log1p", "cand_indices"]:
                batch[k] = batch[k].to(device)

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