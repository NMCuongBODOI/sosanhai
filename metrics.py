import torch
import numpy as np
from math import log2

def get_metrics(predictions, targets, k=20):
    """
    Tính toán Recall@K, Precision@K, và NDCG@K
    Args:
        predictions (torch.Tensor): Ma trận dự đoán [num_users, num_items]
        targets (torch.Tensor / list of lists): ground truth items của mỗi user.
        k (int): Số lượng item gợi ý (Top K)
    """
    recall_list, precision_list, ndcg_list = [], [], []
    
    # Lấy top K dự đoán
    _, top_k_indices = torch.topk(predictions, k=k, dim=1)
    top_k_indices = top_k_indices.cpu().numpy()
    
    for i, user_targets in enumerate(targets):
        if len(user_targets) == 0:
            continue
            
        pred_items = top_k_indices[i]
        
        # Số lượng hit (đoán trúng)
        hits = len(set(pred_items) & set(user_targets))
        
        # 1. Precision@K
        precision = hits / k
        precision_list.append(precision)
        
        # 2. Recall@K
        recall = hits / len(user_targets)
        recall_list.append(recall)
        
        # 3. NDCG@K
        dcg = 0.0
        for rank, item in enumerate(pred_items):
            if item in user_targets:
                dcg += 1.0 / log2(rank + 2) # rank 0 -> log2(2)
                
        idcg = 0.0
        for rank in range(min(len(user_targets), k)):
            idcg += 1.0 / log2(rank + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_list.append(ndcg)
        
    return {
        f'Recall@{k}': np.mean(recall_list),
        f'Precision@{k}': np.mean(precision_list),
        f'NDCG@{k}': np.mean(ndcg_list)
    }

def evaluate_model(model, user_indices, item_indices, all_interactions, k=20):
    """
    Hàm tiện ích để đánh giá nhanh một mô hình.
    Trong đồ án thực tế, ta thường chỉ đánh giá trên 1 tập subset User để tiết kiệm thời gian.
    """
    pass # Hàm này sẽ tuỳ chỉnh trong từng script
