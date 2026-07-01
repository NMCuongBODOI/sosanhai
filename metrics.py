import torch
import numpy as np
from math import log2

def get_metrics(predictions, targets, k=20, total_items=None):
    """
    Tính toán 6 độ đo: Recall@K, Precision@K, NDCG@K, MRR@K, HitRate@K, Coverage@K
    Args:
        predictions (torch.Tensor): Ma trận dự đoán [num_users, num_items]
        targets (list of lists): ground truth items của mỗi user.
        k (int): Số lượng item gợi ý (Top K)
        total_items (int): Tổng số món ăn trong hệ thống (dùng để tính Coverage)
    """
    recall_list, precision_list, ndcg_list = [], [], []
    mrr_list, hr_list = [], []
    recommended_items = set()
    
    # Lấy top K dự đoán
    _, top_k_indices = torch.topk(predictions, k=k, dim=1)
    top_k_indices = top_k_indices.cpu().numpy()
    
    for i, user_targets in enumerate(targets):
        if len(user_targets) == 0:
            continue
            
        pred_items = top_k_indices[i]
        recommended_items.update(pred_items)
        
        # Số lượng hit (đoán trúng)
        hits = set(pred_items) & set(user_targets)
        num_hits = len(hits)
        
        # 1. Precision@K & 2. Recall@K
        precision_list.append(num_hits / k)
        recall_list.append(num_hits / len(user_targets))
        
        # 3. Hit Rate@K
        hr_list.append(1.0 if num_hits > 0 else 0.0)
        
        # 4. NDCG@K
        dcg = 0.0
        for rank, item in enumerate(pred_items):
            if item in user_targets:
                dcg += 1.0 / log2(rank + 2) # rank 0 -> log2(2)
                
        idcg = sum(1.0 / log2(rank + 2) for rank in range(min(len(user_targets), k)))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
        
        # 5. MRR@K
        mrr = 0.0
        for rank, item in enumerate(pred_items):
            if item in user_targets:
                mrr = 1.0 / (rank + 1)
                break
        mrr_list.append(mrr)
        
    # 6. Coverage@K
    coverage = len(recommended_items) / total_items if total_items else 0.0
    
    return {
        f'Recall@{k}': np.mean(recall_list),
        f'Precision@{k}': np.mean(precision_list),
        f'NDCG@{k}': np.mean(ndcg_list),
        f'MRR@{k}': np.mean(mrr_list),
        f'HitRate@{k}': np.mean(hr_list),
        f'Coverage@{k}': coverage
    }

def evaluate_model(model, user_indices, item_indices, all_interactions, k=20):
    pass
