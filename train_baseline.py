import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from metrics import get_metrics
import time

# ==========================================
# 1. Cấu trúc Mô hình Matrix Factorization (MF)
# ==========================================
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_recipes, embedding_dim):
        super(MatrixFactorization, self).__init__()
        # Nhúng (Embeddings) cho User và Recipe
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        
        # Khởi tạo trọng số ngẫu nhiên nhỏ
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.recipe_embedding.weight, std=0.1)

    def forward(self, users, recipes):
        # Tính tích vô hướng (dot product) giữa user và recipe embedding để ra điểm số
        u_emb = self.user_embedding(users)
        r_emb = self.recipe_embedding(recipes)
        return (u_emb * r_emb).sum(dim=1)

# ==========================================
# 2. BPR Loss (Bayesian Personalized Ranking)
# ==========================================
def bpr_loss(pos_scores, neg_scores):
    # Khuyến khích điểm số của item dương (Positive) lớn hơn item âm (Negative)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

# ==========================================
# 3. Chu trình Huấn luyện và Đánh giá Baseline
# ==========================================
def train_baseline(graph_path="food_graph.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # Tải đồ thị
    print("Đang tải dữ liệu đồ thị...")
    graph = torch.load(graph_path, map_location=device)
    num_users = graph['user'].num_nodes
    num_recipes = graph['recipe'].num_nodes
    
    # 3.1. Chia tập Dữ liệu (Train/Val/Test Split)
    # Ta chỉ quan tâm đến cạnh user->recipe để train MF.
    print("Chia tập tương tác thành Train(80%) / Val(10%) / Test(10%)...")
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        edge_types=[('user', 'rates', 'recipe')],
        rev_edge_types=None
    )
    train_data, val_data, test_data = transform(graph)
    
    # Lấy edge_index của tập Train
    train_edges = train_data['user', 'rates', 'recipe'].edge_index.to(device)
    
    # 3.2. Khởi tạo mô hình MF
    embedding_dim = 32
    model = MatrixFactorization(num_users, num_recipes, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    epochs = 100  # Để demo ta chạy ít epoch. Thực tế cần ~50-100 epochs.
    batch_size = 2048
    
    print("\n--- BẮT ĐẦU HUẤN LUYỆN MATRIX FACTORIZATION ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Xáo trộn các cạnh
        perm = torch.randperm(train_edges.size(1))
        
        start_time = time.time()
        for i in range(0, train_edges.size(1), batch_size):
            optimizer.zero_grad()
            
            # Positive samples (người dùng thực sự có tương tác với món này)
            batch_indices = perm[i:i + batch_size]
            users = train_edges[0, batch_indices]
            pos_recipes = train_edges[1, batch_indices]
            
            # Negative Sampling (random các món mà người dùng chưa chắc đã tương tác)
            neg_recipes = torch.randint(0, num_recipes, (len(users),), device=device)
            
            # Tính điểm số
            pos_scores = model(users, pos_recipes)
            neg_scores = model(users, neg_recipes)
            
            # Tính BPR loss
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(users)
            
        avg_loss = total_loss / train_edges.size(1)
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Thời gian: {time.time()-start_time:.2f}s")
        
    # 3.3. Đánh giá (Evaluation) trên tập Test
    print("\n--- ĐÁNH GIÁ MÔ HÌNH (EVALUATION) ---")
    model.eval()
    
    # Gom tất cả ground truth (items người dùng thích trong tập Test)
    test_edges = test_data['user', 'rates', 'recipe'].edge_index.cpu()
    test_users = test_edges[0].numpy()
    test_items = test_edges[1].numpy()
    
    user_targets = {}
    for u, i in zip(test_users, test_items):
        if u not in user_targets:
            user_targets[u] = []
        user_targets[u].append(i)
        
    test_users_list = list(user_targets.keys())
    # Giới hạn số lượng user đánh giá để demo không chạy quá lâu
    # Trên thực tế sẽ đánh giá toàn bộ
    eval_users = test_users_list
    
    targets = [user_targets[u] for u in eval_users]
    
    # Dự đoán (tính điểm từ tất cả user tới tất cả recipe)
    users_tensor = torch.tensor(eval_users, dtype=torch.long, device=device)
    
    # Để tránh OOM khi tính [1000, 41000], ta tính theo từng user hoặc nhân ma trận
    with torch.no_grad():
        u_emb = model.user_embedding(users_tensor) # [1000, 32]
        r_emb = model.recipe_embedding.weight      # [41240, 32]
        # Nhân ma trận để ra điểm số cho toàn bộ recipe: [1000, 32] x [32, 41240] = [1000, 41240]
        predictions = torch.matmul(u_emb, r_emb.t()) 
        
    metrics = get_metrics(predictions, targets, k=20, total_items=num_recipes)
    print("Kết quả trên Baseline (MF):")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
        
    # Lưu file kết quả metrics tạm ra txt để so sánh sau
    with open("baseline_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    train_baseline()
