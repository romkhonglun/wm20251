import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------
# PHẦN 1: CLASS EBNERD DATASET
# ----------------------------------------------------------------------

class EbnerdDataset(Dataset):
    def __init__(self, behaviors_df: pd.DataFrame, user_history_map: dict, articles_dict: dict,
                 history_size:int = 50, negative_ratio:int = 4):
        """
        Khởi tạo Dataset.
        Args:
            behaviors_df: DataFrame behaviors (chứa impressions)
            user_history_map: Dict {user_id: [list_history_ids]}
            articles_dict: Dict {article_id: embedding_vector}
            history_size: Số lượng bài báo lịch sử tối đa (cắt bớt hoặc padding)
            negative_ratio: Số lượng mẫu âm cho mỗi mẫu dương
        """
        self.behaviors = behaviors_df
        self.user_history_map = user_history_map
        self.articles_dict = articles_dict
        self.history_size = history_size
        self.negative_ratio = negative_ratio

        # Lưu lại các ID hợp lệ để lấy ngẫu nhiên
        self.all_article_ids = list(articles_dict.keys())

        if not self.all_article_ids:
            raise ValueError("articles_dict không được rỗng!")

        # Tự động phát hiện kích thước embedding
        self.embedding_dim = len(articles_dict[self.all_article_ids[0]])

    def __len__(self):
        return len(self.behaviors)

    def _get_random_embedding(self):
        """Lấy một embedding ngẫu nhiên từ dict."""
        random_id = random.choice(self.all_article_ids)
        return self.articles_dict[random_id]

    def _get_embedding(self, article_id):
        """
        Lấy embedding. Nếu không tìm thấy, trả về một embedding ngẫu nhiên.
        """
        emb = self.articles_dict.get(article_id)
        if emb is None:
            return self._get_random_embedding()
        return emb

    def __getitem__(self, idx):
        # Lấy 1 dòng hành vi
        row = self.behaviors.iloc[idx]
        user_id = row['user_id']

        # --- 1. Xử lý Lịch sử (User History) ---
        history_ids = self.user_history_map.get(user_id, [])

        # Cắt bớt nếu quá dài
        if len(history_ids) > self.history_size:
            history_ids = history_ids[-self.history_size:]  # Lấy mới nhất

        history_embs = [self._get_embedding(aid) for aid in history_ids]

        # Padding bằng embedding ngẫu nhiên nếu quá ngắn
        padding_count = self.history_size - len(history_embs)
        if padding_count > 0:
            history_embs.extend([self._get_random_embedding() for _ in range(padding_count)])

        history_embs = np.array(history_embs)

        # --- 2. Xử lý Candidate (Positive + Negatives) ---
        clicked_ids = row['article_ids_clicked']
        inview_ids = row['article_ids_inview']

        if not isinstance(clicked_ids, list):
            clicked_ids = []
        if not isinstance(inview_ids, list):
            inview_ids = []
        # Các bài báo âm = inview - clicked
        neg_ids_pool = list(set(inview_ids) - set(clicked_ids))

        # Chọn 1 bài positive ngẫu nhiên
        if len(clicked_ids):
            pos_id = random.choice(clicked_ids)
        else:
            # Fallback: Nếu không có click, lấy tạm 1 bài trong inview
            pos_id = random.choice(inview_ids) if inview_ids else ""

        pos_emb = self._get_embedding(pos_id)
        pos_emb = torch.tensor(pos_emb, dtype=torch.float32).unsqueeze(0)

        # Lấy K mẫu âm (cho phép lặp lại)
        negative_embs = []
        if neg_ids_pool:
            # Dùng random.choices: Tự động lặp lại nếu k > len(pool)
            neg_ids = random.choices(neg_ids_pool, k=self.negative_ratio)
            negative_embs = [self._get_embedding(aid) for aid in neg_ids]
        else:
            # Nếu không có negative nào, pad bằng K embedding ngẫu nhiên
            negative_embs = [self._get_random_embedding() for _ in range(self.negative_ratio)]
        negative_embs = np.array(negative_embs)
        negative_embs = torch.tensor(negative_embs, dtype=torch.float32)

        random_pos_position = np.random.randint(0, self.negative_ratio + 1)
        # Ghép Positive (luôn ở vị trí 0) và Negatives
        candidate_embs = torch.cat((negative_embs[:random_pos_position], pos_emb, negative_embs[random_pos_position:]),
                                   dim=0)

        # --- 3. Tạo Label ---
        label = torch.zeros(self.negative_ratio + 1, dtype=torch.float32)
        label[random_pos_position] = 1  # Vì Positive luôn ở vị trí 0

        return {
            'history': torch.tensor(history_embs, dtype=torch.float32),
            'candidate': candidate_embs,
            'label': label
        }