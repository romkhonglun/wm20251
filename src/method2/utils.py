import torch
import torch.nn.functional as F
import torchmetrics


# --- 1. Custom Vectorized Metrics (Thuần PyTorch - Siêu nhanh) ---

def fast_mrr(preds, labels):
    """
    Tính MRR cho cả batch cùng lúc bằng phép toán ma trận.
    Args:
        preds: (Batch, List_Size) - Điểm số dự đoán
        labels: (Batch, List_Size) - Nhãn (0 hoặc 1)
    """
    # 1. Sắp xếp điểm dự đoán từ cao xuống thấp
    # indices: vị trí index của các bài báo sau khi sort
    _, indices = torch.sort(preds, dim=1, descending=True)

    # 2. Sắp xếp lại nhãn theo thứ tự của preds
    # labels_sorted: nhãn tương ứng với các bài báo đã sort
    labels_sorted = torch.gather(labels, dim=1, index=indices)

    # 3. Tìm vị trí (rank) của bài báo dương (label=1) đầu tiên
    # nonzero() trả về indices của các phần tử != 0
    # Vì mỗi user có thể có nhiều bài dương, ta lấy bài có rank cao nhất
    # Tuy nhiên, trong News RecSys thường chỉ có 1 bài dương.

    # Tạo ma trận ranks: [[1, 2, 3...], [1, 2, 3...]]
    batch_size, list_size = labels.shape
    ranks = torch.arange(1, list_size + 1, device=preds.device).unsqueeze(0).expand(batch_size, list_size)

    # Chỉ lấy rank ở những chỗ label=1
    # Những chỗ label=0 thì cho ranks = vô cực để không chọn phải
    hits_ranks = ranks.clone().float()
    hits_ranks[labels_sorted == 0] = float('inf')

    # Lấy min rank (vị trí đầu tiên tìm thấy bài dương) cho mỗi user
    min_rank, _ = hits_ranks.min(dim=1)

    # 4. Tính Reciprocal Rank (1 / rank)
    # Nếu không tìm thấy bài dương nào (vô cực), rr = 0
    mrr = 1.0 / min_rank
    mrr[min_rank == float('inf')] = 0.0

    return mrr.mean()


def fast_ndcg(preds, labels, k=10):
    """
    Tính NDCG@K vector hóa hoàn toàn.
    """
    batch_size, list_size = preds.shape
    k = min(k, list_size)

    # 1. Sort preds và lấy top k
    _, indices = torch.topk(preds, k, dim=1)

    # 2. Lấy nhãn tương ứng với top k
    topk_labels = torch.gather(labels, dim=1, index=indices)

    # 3. Tính DCG (Discounted Cumulative Gain)
    # Công thức: sum( (2^rel - 1) / log2(rank + 1) )
    # Vì rel chỉ là 0 hoặc 1 nên 2^rel - 1 chính là rel (topk_labels)
    gains = topk_labels.float()
    discounts = torch.log2(torch.arange(2, k + 2, device=preds.device).float())
    dcg = (gains / discounts).sum(dim=1)

    # 4. Tính IDCG (Ideal DCG - trường hợp lý tưởng nhất)
    # Sort labels thật để đưa các số 1 lên đầu
    ideal_labels, _ = torch.sort(labels, dim=1, descending=True)
    ideal_topk = ideal_labels[:, :k]
    ideal_gains = ideal_topk.float()
    idcg = (ideal_gains / discounts).sum(dim=1)

    # 5. Tính NDCG
    ndcg = dcg / (idcg + 1e-4)  # Cộng epsilon để tránh chia 0
    return ndcg.mean()


def fast_auc(preds, labels):
    """
    Tính AUC Global (trên toàn bộ batch đã flatten).
    Dùng hàm có sẵn của torchmetrics vì hàm này ổn định.
    """
    from torchmetrics.functional.classification import binary_auroc
    return binary_auroc(preds, labels)


# --- 2. Loss Function ---

def binary_listnet_loss(y_pred, y_true, eps=1e-4, padded_value_indicator=-1):
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    mask = y_true == padded_value_indicator
    y_pred[mask] = -1e4
    y_true[mask] = 0.0
    normalizer = torch.unsqueeze(y_true.sum(dim=-1), 1)
    normalizer[normalizer == 0.0] = 1.0
    normalizer = normalizer.expand(-1, y_true.shape[1])
    y_true = torch.div(y_true, normalizer)
    preds_log = F.log_softmax(y_pred, dim=1)
    return torch.mean(-torch.sum(y_true * preds_log, dim=1))


# --- 3. MetricsMeter Class ---

class MetricsMeter(torch.nn.Module):
    def __init__(self, loss_weights: dict = None):
        super().__init__()
        self.loss_weights = loss_weights or {"bce_loss": 1.0, "listnet_loss": 0.5}
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        # MeanMetric lưu trữ scalar trung bình, rất nhẹ
        self.auc_metric = torchmetrics.MeanMetric()
        self.mrr_metric = torchmetrics.MeanMetric()
        self.ndcg5_metric = torchmetrics.MeanMetric()
        self.ndcg10_metric = torchmetrics.MeanMetric()

    def reset(self):
        self.auc_metric.reset()
        self.mrr_metric.reset()
        self.ndcg5_metric.reset()
        self.ndcg10_metric.reset()

    def update(self, batch: dict, n_samples: int = None):
        metrics = {}
        preds = batch["preds"]
        labels = batch["labels"]

        # --- A. Tính Loss ---
        mask = labels != -1
        if "bce_loss" in self.loss_weights:
            bce = self.bce_loss(preds, labels.float())
            bce = bce.masked_fill(~mask, 0)
            metrics["bce_loss"] = bce.sum() / (mask.sum() + 1e-4)

        if "listnet_loss" in self.loss_weights:
            metrics["listnet_loss"] = binary_listnet_loss(
                y_pred=preds, y_true=labels.float(), padded_value_indicator=-1
            )

        metrics["loss"] = sum(w * metrics.get(k, 0.0) for k, w in self.loss_weights.items())

        # --- B. Chuẩn bị Data cho Metrics ---
        if n_samples:
            preds = preds[:n_samples]
            labels = labels[:n_samples]
            mask = mask[:n_samples]

        # 1. Masking Padding trong Preds
        # Gán điểm cực thấp (-infinity) cho các vị trí padding để khi sort nó luôn nằm cuối
        # clone để không ảnh hưởng gradient của loss
        eval_preds = preds.clone().detach()
        eval_preds[~mask] = -1e4

        # 2. Chuẩn bị Labels (Binary 0/1)
        # Convert label > 0 thành 1, padding (-1) thành 0
        eval_labels = (labels > 0).long()
        # Đảm bảo padding không được tính là positive
        eval_labels[~mask] = 0

        # --- C. Tính Metrics (Native PyTorch) ---
        # Chỉ tính nếu batch có dữ liệu
        if preds.shape[0] > 0:
            # 1. MRR
            batch_mrr = fast_mrr(eval_preds, eval_labels)
            self.mrr_metric.update(batch_mrr, weight=preds.shape[0])

            # 2. NDCG
            batch_ndcg5 = fast_ndcg(eval_preds, eval_labels, k=5)
            self.ndcg5_metric.update(batch_ndcg5, weight=preds.shape[0])

            batch_ndcg10 = fast_ndcg(eval_preds, eval_labels, k=10)
            self.ndcg10_metric.update(batch_ndcg10, weight=preds.shape[0])

            # 3. AUC (Vẫn dùng thư viện cho tiện, flatten batch)
            # Chỉ tính AUC trên phần tử hợp lệ (không phải padding)
            flat_preds = eval_preds[mask]
            flat_labels = eval_labels[mask]

            # Cần check xem có cả class 0 và 1 không
            if len(torch.unique(flat_labels)) == 2:
                batch_auc = fast_auc(flat_preds, flat_labels)
                self.auc_metric.update(batch_auc, weight=flat_preds.numel())

        return metrics

    def compute(self, suffix: str = ""):
        return {
            f"auc{suffix}": self.auc_metric.compute(),
            f"mrr{suffix}": self.mrr_metric.compute(),
            f"ndcg@5{suffix}": self.ndcg5_metric.compute(),
            f"ndcg@10{suffix}": self.ndcg10_metric.compute(),
        }
#
# import torch
# import torch.nn.functional as F
# import torchmetrics
# from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG, RetrievalAUROC
#
#
# # --- utils.py giữ nguyên binary_listnet_loss ---
# def binary_listnet_loss(y_pred, y_true, eps=1e-4, padded_value_indicator=-1):
#     y_pred = y_pred.clone()
#     y_true = y_true.clone()
#     mask = y_true == padded_value_indicator
#     y_pred[mask] = -1e4
#     y_true[mask] = 0.0
#     normalizer = torch.unsqueeze(y_true.sum(dim=-1), 1)
#     normalizer[normalizer == 0.0] = 1.0
#     normalizer = normalizer.expand(-1, y_true.shape[1])
#     y_true = torch.div(y_true, normalizer)
#     preds_log = F.log_softmax(y_pred, dim=1)
#     return torch.mean(-torch.sum(y_true * preds_log, dim=1))
#
#
# class MetricsMeter(torch.nn.Module):
#     def __init__(self, loss_weights: dict = None):
#         super().__init__()
#         if loss_weights is None:
#             self.loss_weights = {"bce_loss": 1.0, "listnet_loss": 0.5}
#         else:
#             self.loss_weights = loss_weights
#
#         self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
#
#         # SỬ DỤNG CÁC CLASS CÓ SẴN CỦA TORCHMETRICS
#         # Các metric này tự động hỗ trợ DDP (chạy nhiều GPU)
#         self.eval_metrics = torchmetrics.MetricCollection({
#             "auc": RetrievalAUROC(),
#             "mrr": RetrievalMRR(),
#             "ndcg@5": RetrievalNormalizedDCG(top_k=5),
#             "ndcg@10": RetrievalNormalizedDCG(top_k=10),
#         })
#         self.reset()
#
#     def reset(self):
#         self.eval_metrics.reset()
#
#     def update(self, batch: dict, n_samples: int = None):
#         metrics = {}
#         preds = batch["preds"]  # Shape: [Batch_Size, List_Size]
#         labels = batch["labels"]  # Shape: [Batch_Size, List_Size]
#
#         # --- Xử lý Loss (Giữ nguyên logic cũ) ---
#         mask = labels != -1
#         if "bce_loss" in self.loss_weights:
#             bce = self.bce_loss(preds, labels.float())
#             bce = bce.masked_fill(~mask, 0)
#             metrics["bce_loss"] = bce.sum() / (mask.sum() + 1e-4)
#
#         if "listnet_loss" in self.loss_weights:
#             metrics["listnet_loss"] = binary_listnet_loss(
#                 y_pred=preds,
#                 y_true=labels.float(),
#                 padded_value_indicator=-1
#             )
#
#         metrics["loss"] = sum(w * metrics.get(k, 0.0) for k, w in self.loss_weights.items())
#
#         # --- TỐI ƯU HÓA PHẦN METRICS UPDATE ---
#
#         # 1. Cắt n_samples nếu cần
#         if n_samples:
#             _preds = preds[:n_samples]
#             _labels = labels[:n_samples]
#         else:
#             _preds, _labels = preds, labels
#
#         # 2. Tạo mask loại bỏ padding (-1)
#         # Ví dụ: labels = [[1, 0, -1], [0, 1, 0]]
#         valid_mask = _labels != -1
#
#         # 3. Flatten dữ liệu (Làm phẳng)
#         # Chỉ giữ lại các phần tử hợp lệ (không phải padding)
#         flat_preds = _preds[valid_mask]
#         flat_labels = _labels[valid_mask].long()  # Target retrieval cần long hoặc bool
#
#         # 4. Tạo indexes để biết phần tử nào thuộc về user (batch) nào
#         # Tạo tensor index tương ứng với batch dimension:
#         # Ví dụ batch_size=2, list_size=3 -> batch_idx = [[0,0,0], [1,1,1]]
#         batch_size, list_size = _preds.shape
#         indexes = torch.arange(batch_size, device=_preds.device).unsqueeze(1).expand(batch_size, list_size)
#         flat_indexes = indexes[valid_mask]  # Lọc theo mask luôn
#
#         # 5. Update vào metric collection (Chạy song song hoàn toàn)
#         # Hàm này sẽ tự gom nhóm theo flat_indexes để tính AUC/MRR cho từng user rồi mean lại
#         self.eval_metrics.update(
#             preds=flat_preds,
#             target=flat_labels,
#             indexes=flat_indexes
#         )
#
#         return metrics
#
#     def compute(self, suffix: str = ""):
#         return {f"{k}{suffix}": v for k, v in self.eval_metrics.compute().items()}