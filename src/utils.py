import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

# Setup Logger
logger = logging.getLogger(__name__)


def my_collate_fn(batch):
    """
    Gộp một list các dictionary thành một batch duy nhất.
    Tối ưu: Sử dụng pin_memory=True trong DataLoader để tận dụng tốt nhất.
    """
    histories = torch.stack([item['history'] for item in batch])
    candidates = torch.stack([item['candidate'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'history': histories,  # Shape: (Batch, History_Size, Emb_Dim)
        'candidate': candidates,  # Shape: (Batch, 1 + Neg_Ratio, Emb_Dim)
        'label': labels  # Shape: (Batch, 1 + Neg_Ratio)
    }


def create_user_history_map(hist_df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Tạo map user_id -> sorted article_ids.
    """
    logger.info("Bắt đầu tạo user_history_map...")

    user_history_map = {}

    # Pre-calculate stats for logging
    total_rows = len(hist_df)

    # Lưu ý: itertuples nhanh hơn iterrows rất nhiều
    for row in hist_df.itertuples(index=False):
        user_id = row.user_id
        # Giả sử tên cột trong parquet là 'impression_time_fixed' và 'article_id_fixed'
        # Cần đảm bảo tên cột khớp với dataframe thực tế của bạn
        times = getattr(row, 'impression_time_fixed', [])
        articles = getattr(row, 'article_id_fixed', [])

        try:
            # Kiểm tra nhanh nếu list rỗng hoặc độ dài không khớp
            if len(articles) == 0:
                user_history_map[user_id] = []
                continue

            # Convert sang numpy array để xử lý nhanh hơn list python
            if isinstance(articles, list):
                articles = np.array(articles)
            if isinstance(times, list):
                times = np.array(times)

            # Argsort dựa trên thời gian
            # Lưu ý: Nếu dữ liệu đầu vào đã sort sẵn thì có thể bỏ bước này để tăng tốc
            sorted_indices = np.argsort(times)
            sorted_articles = articles[sorted_indices]

            user_history_map[user_id] = sorted_articles.tolist()

        except Exception as e:
            # Log warning nhưng không crash luồng chính
            # logger.warning(f"Lỗi xử lý user {user_id}: {e}")
            user_history_map[user_id] = []

    logger.info(f"Hoàn thành user_history_map: {len(user_history_map)} users.")
    return user_history_map


class RecommendationMetrics:
    """
    Phiên bản tối ưu hóa Vectorization (chạy song song trên GPU).
    Loại bỏ hoàn toàn vòng lặp Python chậm chạp.
    """

    @staticmethod
    def calculate_metrics(predictions: torch.Tensor,
                          labels: torch.Tensor,
                          k: int = 10):
        """
        Args:
            predictions: (Batch, Num_Items) - Logits/Scores
            labels: (Batch, Num_Items) - One-hot encoding
            k: Top-k
        """
        # Đảm bảo tính toán trên cùng device và không lưu gradient
        device = predictions.device

        # 1. Xác định vị trí bài báo đúng (Ground Truth Index)
        # labels là one-hot, argmax lấy ra index của số 1
        target_indices = torch.argmax(labels, dim=1)  # Shape: (Batch,)

        # 2. Sắp xếp dự đoán (Sorting)
        # Chỉ cần lấy Top-K + 1 để check rank, nhưng sort all để tính AUC dễ hơn
        # sorted_indices: Chứa index của các bài báo sau khi sort theo điểm giảm dần
        _, sorted_indices = torch.sort(predictions, descending=True, dim=1)

        # 3. Tính Rank của bài đúng
        # Tìm xem target_index nằm ở đâu trong sorted_indices
        # logic: (sorted_indices == target_indices.view(-1, 1)) tạo ra ma trận True/False
        # nonzero() trả về toạ độ, cột thứ 2 chính là rank (0-based)
        hits_mask = (sorted_indices == target_indices.view(-1, 1))

        # nonzero()[:, 1] lấy ra chỉ số cột (rank) nơi hits_mask == True
        # Cộng 1 để ra rank thực tế (1-based)
        # Lưu ý: Mỗi hàng chỉ có đúng 1 True vì label one-hot
        ranks = hits_mask.nonzero(as_tuple=True)[1] + 1
        ranks = ranks.float()

        # --- BẮT ĐẦU TÍNH METRICS (VECTORIZED) ---

        # A. Hit Rate @ K (HR)
        # 1 nếu rank <= k, ngược lại 0
        hr_score = (ranks <= k).float().mean().item()

        # B. MRR (Mean Reciprocal Rank)
        # 1 / rank
        mrr_score = (1.0 / ranks).mean().item()

        # C. NDCG @ K
        # NDCG = 1 / log2(rank + 1) nếu rank <= k, ngược lại 0
        # Tạo mask cho các rank <= k
        ndcg_mask = (ranks <= k).float()
        # Tính điểm
        ndcg_vals = (1.0 / torch.log2(ranks + 1)) * ndcg_mask
        ndcg_score = ndcg_vals.mean().item()

        # D. MAP (Mean Average Precision)
        # Với bài toán tìm 1 item đúng (One-hot), MAP chính là MRR
        map_score = mrr_score

        # E. AUC (Area Under Curve) - Optimized broadcast
        # Lấy score của bài dương
        pos_scores = torch.gather(predictions, 1, target_indices.view(-1, 1))  # (Batch, 1)

        # Tính hiệu số giữa bài dương và TẤT CẢ bài khác
        # diff > 0 nghĩa là bài dương có điểm cao hơn -> Win
        diff = pos_scores - predictions

        # Bỏ qua so sánh với chính nó (diff == 0 tại vị trí target)
        # Cách tính AUC nhanh: (Số lượng Neg < Pos) / Tổng Neg

        # Đếm số lượng bài (bao gồm cả chính nó) mà Pos Score > Other Score
        wins = (diff > 0).sum(dim=1).float()

        # Đếm trường hợp bằng điểm (Tie) - tính 0.5
        ties = (diff == 0).sum(dim=1).float() - 1.0  # Trừ chính nó
        ties = torch.clamp(ties, min=0.0)  # An toàn

        # Tổng số sample âm = Tổng cột - 1
        num_neg = predictions.size(1) - 1

        auc_per_user = (wins + 0.5 * ties) / num_neg
        auc_score = auc_per_user.mean().item()

        return map_score, ndcg_score, mrr_score, hr_score, auc_score


# Test nhanh logic (khi chạy trực tiếp file này)
if __name__ == "__main__":
    # Mock data
    preds = torch.tensor([
        [0.1, 0.9, 0.2, 0.1],  # User 1: Đúng là index 1 (score 0.9 - Rank 1)
        [0.8, 0.1, 0.1, 0.9]  # User 2: Đúng là index 0 (score 0.8 - Rank 2, thua index 3)
    ])
    lbls = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ])

    m = RecommendationMetrics()
    results = m.calculate_metrics(preds, lbls, k=1)

    print(f"MAP: {results[0]}")  # Exp: (1/1 + 1/2)/2 = 0.75
    print(f"NDCG@1: {results[1]}")  # Exp: (1 + 0)/2 = 0.5 (User 2 rank 2 > k=1 nên = 0)
    print(f"MRR: {results[2]}")  # Exp: 0.75
    print(f"HR@1: {results[3]}")  # Exp: 0.5
    print(f"AUC: {results[4]}")