import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import ast
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.transforms import RandomLinkSplit
from metrics import get_metrics

# ==========================================
# 1. Mô hình Matrix Factorization (SVD)
# ==========================================
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_recipes, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.recipe_embedding.weight, std=0.1)

    def forward(self, users, recipes):
        u_emb = self.user_embedding(users)
        r_emb = self.recipe_embedding(recipes)
        return (u_emb * r_emb).sum(dim=1)

def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

# ==========================================
# 2. Hàm Chuẩn Hóa Min-Max (Cho Từng Dòng/User)
# ==========================================
def min_max_normalize(tensor_2d):
    """Chuẩn hóa giá trị của mỗi dòng (user) về khoảng [0, 1]"""
    min_vals, _ = torch.min(tensor_2d, dim=1, keepdim=True)
    max_vals, _ = torch.max(tensor_2d, dim=1, keepdim=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8 # Tránh chia cho 0
    return (tensor_2d - min_vals) / range_vals

# ==========================================
# 3. Chu trình Huấn luyện & Kết hợp Hybrid
# ==========================================
def train_hybrid(graph_path="food_graph.pt", csv_path="RAW_recipes.csv", mappings_path="mappings.pkl", alpha=0.5):
    print("\n=== BẮT ĐẦU CHẠY MÔ HÌNH HYBRID (SVD + CONTENT-BASED) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    start_time = time.time()
    
    # ----------------------------------------------------
    # PHẦN A: TẢI VÀ CHUẨN BỊ DỮ LIỆU
    # ----------------------------------------------------
    print("\n[1/4] Đang tải Dữ liệu và Xây dựng TF-IDF Content...")
    graph = torch.load(graph_path, map_location=device)
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
        
    num_users = graph['user'].num_nodes
    num_recipes = graph['recipe'].num_nodes
    recipe_id_to_idx = mappings['recipe']
    
    # Đọc thông tin văn bản từ CSV
    recipes_df = pd.read_csv(csv_path)
    recipes_df['tags_list'] = recipes_df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    recipes_df['ingredients_list'] = recipes_df['ingredients'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    recipes_df['content_text'] = recipes_df.apply(
        lambda row: " ".join(row['tags_list'] + row['ingredients_list']).replace("-", "").replace(" ", "_"), axis=1)
    
    # Căn chỉnh Index đúng với Đồ thị
    valid_ids = set(recipe_id_to_idx.keys())
    valid_df = recipes_df[recipes_df['id'].isin(valid_ids)].copy()
    valid_df['node_idx'] = valid_df['id'].map(recipe_id_to_idx)
    valid_df = valid_df.sort_values(by='node_idx')
    corpus = valid_df['content_text'].tolist()
    
    # Tính TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    item_tfidf = vectorizer.fit_transform(corpus)
    print(f"   Shape TF-IDF Món ăn: {item_tfidf.shape}")
    
    # Chia tập Train/Test/Val (Cùng Seed 42 để công bằng với các mô hình khác)
    print("\n[2/4] Chia tập Train/Test (Seed: 42)...")
    torch.manual_seed(42)
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1, is_undirected=False,
        edge_types=[('user', 'rates', 'recipe')], rev_edge_types=None 
    )
    train_data, val_data, test_data = transform(graph)
    
    train_edges = train_data['user', 'rates', 'recipe'].edge_index
    test_edges = test_data['user', 'rates', 'recipe'].edge_index.cpu().numpy()
    
    # Xây dựng User Profiles cho Content-Based
    train_users_np = train_edges[0].cpu().numpy()
    train_items_np = train_edges[1].cpu().numpy()
    train_df = pd.DataFrame({'user': train_users_np, 'item': train_items_np})
    user_item_dict = train_df.groupby('user')['item'].apply(list).to_dict()

    # ----------------------------------------------------
    # PHẦN B: HUẤN LUYỆN SVD (MATRIX FACTORIZATION)
    # ----------------------------------------------------
    print("\n[3/4] Huấn luyện Mô hình SVD (Matrix Factorization)...")
    embedding_dim = 32
    model = MatrixFactorization(num_users, num_recipes, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    epochs = 100
    batch_size = 2048
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        perm = torch.randperm(train_edges.size(1))
        
        for i in range(0, train_edges.size(1), batch_size):
            optimizer.zero_grad()
            batch_indices = perm[i:i + batch_size]
            users = train_edges[0, batch_indices]
            pos_recipes = train_edges[1, batch_indices]
            neg_recipes = torch.randint(0, num_recipes, (len(users),), device=device)
            
            pos_scores = model(users, pos_recipes)
            neg_scores = model(users, neg_recipes)
            
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(users)
            
        if (epoch+1) % 25 == 0 or epoch == 0:
            avg_loss = total_loss / train_edges.size(1)
            print(f"   Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f}")
            
    # ----------------------------------------------------
    # PHẦN C: ĐÁNH GIÁ (EVALUATION) VÀ KẾT HỢP HYBRID
    # ----------------------------------------------------
    print("\n[4/4] Tính điểm Hybrid (Late Fusion) và Đánh giá...")
    model.eval()
    
    # Tìm ground truth của tập Test
    test_users = test_edges[0]
    test_items = test_edges[1]
    
    user_targets = {}
    for u, i in zip(test_users, test_items):
        if u not in user_targets:
            user_targets[u] = []
        user_targets[u].append(i)
        
    eval_users = list(user_targets.keys())
    targets = [user_targets[u] for u in eval_users]
    
    batch_size_eval = 500
    all_hybrid_predictions = []
    
    with torch.no_grad():
        r_emb = model.recipe_embedding.weight.cpu() # Lấy item embedding của SVD [41240, 32]
        
        for i in range(0, len(eval_users), batch_size_eval):
            batch_users = eval_users[i:i+batch_size_eval]
            
            # --- TÍNH ĐIỂM SVD (MF) ---
            users_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            u_emb = model.user_embedding(users_tensor).cpu()
            svd_scores = torch.matmul(u_emb, r_emb.t()) # [batch_size, 41240]
            
            # --- TÍNH ĐIỂM CONTENT-BASED ---
            batch_profiles = []
            for u in batch_users:
                if u in user_item_dict:
                    profile = item_tfidf[user_item_dict[u]].mean(axis=0)
                else:
                    profile = np.zeros((1, item_tfidf.shape[1]))
                batch_profiles.append(np.asarray(profile).flatten())
                
            batch_profiles_matrix = np.vstack(batch_profiles)
            cb_scores = cosine_similarity(batch_profiles_matrix, item_tfidf)
            cb_scores = torch.tensor(cb_scores)
            
            # --- CHUẨN HÓA VÀ KẾT HỢP (HYBRID) ---
            norm_svd_scores = min_max_normalize(svd_scores)
            norm_cb_scores = min_max_normalize(cb_scores)
            
            # Trọng số kết hợp Alpha (0.5 nghĩa là chia đều sức mạnh 50-50)
            hybrid_scores = alpha * norm_svd_scores + (1 - alpha) * norm_cb_scores
            all_hybrid_predictions.append(hybrid_scores)
            
    predictions_tensor = torch.cat(all_hybrid_predictions, dim=0)
    
    # ----------------------------------------------------
    # KẾT QUẢ
    # ----------------------------------------------------
    print("\n=== KẾT QUẢ TRÊN MÔ HÌNH HYBRID (SVD + CB) ===")
    metrics = get_metrics(predictions_tensor, targets, k=20, total_items=num_recipes)
    
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
        
    with open("hybrid_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            
    print(f"\nTổng thời gian chạy: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    train_hybrid(alpha=0.5)
