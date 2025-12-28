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

# Thêm đường dẫn hiện tại vào sys.path để import các module cùng thư mục
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import trực tiếp từ các file trong cùng thư mục baseline
from lightning_module import NAMLLightningModule
from model import VariantNAMLConfig
from dataset import CompactHistory


# --- HÀM HỖ TRỢ ---
def calculate_ranks(scores):
    """Chuyển đổi scores thành ranks (1 = score cao nhất)."""
    scores = np.array(scores)
    # Lấy indices để sort giảm dần
    indices_desc = np.argsort(-scores)
    # Tạo mảng rank
    ranks = np.empty_like(indices_desc)
    # Gán rank: rank 1 cho phần tử lớn nhất
    ranks[indices_desc] = np.arange(len(scores)) + 1
    return ranks.tolist()


def write_submission(predictions, output_path):
    print(f"📦 Đang nén file submission vào: {output_path}")
    lines = []
    for imp_id, ranks in predictions:
        # Format: impression_id [rank1,rank2,rank3]
        rank_str = "[" + ",".join(map(str, ranks)) + "]"
        lines.append(f"{imp_id} {rank_str}")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("predictions.txt", "\n".join(lines))
    print("✅ Đã tạo file submission thành công!")


# --- DATASET CHO TEST ---
class BaselineTestDataset(IterableDataset):
    def __init__(self, behaviors_path, compact_history, max_hist_len=50):
        self.behaviors_path = behaviors_path
        self.compact_history = compact_history
        self.max_hist_len = max_hist_len

    def __iter__(self):
        pf = pq.ParquetFile(self.behaviors_path)
        # Đọc từng batch lớn để tối ưu I/O
        for batch in pf.iter_batches(batch_size=4096):
            df = pl.from_arrow(batch)

            # Xử lý tên cột (Test set gốc thường là 'article_ids_inview')
            inv_col = 'article_ids_inview' if 'article_ids_inview' in df.columns else 'inv_ids'

            # Kiểm tra impression_id
            if 'impression_id' not in df.columns:
                print("⚠️ Cảnh báo: Không tìm thấy cột 'impression_id'. Sẽ dùng dummy ID.")
                imp_ids = [0] * len(df)
            else:
                imp_ids = df['impression_id'].to_list()

            user_ids = df['user_id'].to_list()
            inv_lists = df[inv_col].to_list()

            for i in range(len(df)):
                hist_ids = self.compact_history.get_history(user_ids[i])
                yield {
                    "impression_id": imp_ids[i],
                    "hist_ids": hist_ids[:self.max_hist_len],
                    "candidate_ids": np.array(inv_lists[i], dtype=np.int32),
                }


def collate_fn_test(batch):
    imp_ids = [item['impression_id'] for item in batch]
    hist_ids = [torch.from_numpy(item['hist_ids']).long() for item in batch]
    cands = [torch.from_numpy(item['candidate_ids']).long() for item in batch]

    # Pad history và candidates
    hist_padded = pad_sequence(hist_ids, batch_first=True, padding_value=0)
    cands_padded = pad_sequence(cands, batch_first=True, padding_value=0)

    return {
        "impression_ids": imp_ids,
        "hist_ids": hist_padded,
        "candidate_ids": cands_padded,
        "cand_lens": [len(c) for c in cands]  # Lưu độ dài thật để cắt rank sau này
    }


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Test Script for Baseline")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path tới file .ckpt")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Folder chứa behaviors.parquet và history.parquet của tập test")
    parser.add_argument("--embedding-dir", type=str, default="/home2/congnh/wm/embedding_test",
                        help="Folder chứa embedding npy")
    parser.add_argument("--output", type=str, default="submission_baseline.zip")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    print("🔄 Đang load model Baseline...")
    model = NAMLLightningModule.load_from_checkpoint(
        args.checkpoint,
        config=VariantNAMLConfig(),
        embedding_dir=args.embedding_dir,
        map_location=device
    )
    model.eval()
    model.to(device)

    # 2. Load Data
    # Lưu ý: Nếu tên file khác (ví dụ behaviors_processed.parquet), hãy sửa lại ở đây
    beh_path = os.path.join(args.test_dir,"test", "behaviors_processed.parquet")
    hist_path = os.path.join(args.test_dir,"test", "history_processed.parquet")

    print(f"📚 Đang load History từ {hist_path}...")
    compact_hist = CompactHistory(hist_path)
    print(f"📚 Đang load Behaviors từ {beh_path}...")
    dataset = BaselineTestDataset(beh_path, compact_hist)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_test)

    # 3. Inference
    predictions = []
    print("🚀 Bắt đầu dự đoán...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move data to GPU
            model_input = {
                "hist_ids": batch["hist_ids"].to(device),
                "candidate_ids": batch["candidate_ids"].to(device),
                "labels": None  # Test không cần labels
            }

            output = model(model_input)

            # Lấy output preds
            if isinstance(output, dict):
                scores = output["preds"].cpu().numpy()
            else:
                scores = output.cpu().numpy()

            # Xử lý từng sample trong batch
            for i, imp_id in enumerate(batch["impression_ids"]):
                valid_len = batch["cand_lens"][i]
                valid_scores = scores[i, :valid_len]  # Cắt bỏ phần padding
                ranks = calculate_ranks(valid_scores)
                predictions.append((imp_id, ranks))

    # 4. Save
    write_submission(predictions, args.output)


if __name__ == "__main__":
    main()