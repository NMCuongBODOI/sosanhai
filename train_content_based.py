import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.transforms import RandomLinkSplit
import time
from metrics import get_metrics

def train_content_based(graph_path="food_graph.pt", csv_path="RAW_recipes.csv", mappings_path="mappings.pkl"):
    print("\n--- BẮT ĐẦU CHẠY MÔ HÌNH LỌC NỘI DUNG (CONTENT-BASED FILTERING) ---")
    start_time = time.time()
    
    # 1. Tải Dữ liệu
    print("1. Đang tải Dữ liệu Đồ thị và Mappings...")
    graph = torch.load(graph_path, map_location='cpu')
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
        
    num_recipes = graph['recipe'].num_nodes
    recipe_id_to_idx = mappings['recipe']
    
    print("2. Đang đọc thông tin Món ăn từ CSV...")
    recipes_df = pd.read_csv(csv_path)
    # Rút trích các thuộc tính Text (Tag + Ingredient)
    recipes_df['tags_list'] = recipes_df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    recipes_df['ingredients_list'] = recipes_df['ingredients'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Gộp tag và nguyên liệu thành 1 chuỗi document để làm Content
    recipes_df['content_text'] = recipes_df.apply(
        lambda row: " ".join(row['tags_list'] + row['ingredients_list']).replace("-", "").replace(" ", "_"), 
        axis=1
    )
    
    # Sắp xếp các document theo đúng thứ tự Index của Mạng (từ 0 đến 41239)
    valid_ids = set(recipe_id_to_idx.keys())
    valid_df = recipes_df[recipes_df['id'].isin(valid_ids)].copy()
    valid_df['node_idx'] = valid_df['id'].map(recipe_id_to_idx)
    valid_df = valid_df.sort_values(by='node_idx')
    
    corpus = valid_df['content_text'].tolist()
    
    # 2. Xây dựng Đặc trưng TF-IDF
    print("3. Đang xây dựng Ma trận Đặc trưng TF-IDF (TfidfVectorizer)...")
    vectorizer = TfidfVectorizer(max_features=5000) # Lấy 5000 đặc trưng phổ biến nhất
    item_tfidf = vectorizer.fit_transform(corpus) # [41240, 5000] scipy sparse matrix
    print(f"   Shape của Ma trận Item: {item_tfidf.shape}")
    
    # 3. Chia tập Dữ liệu (Giống hệt MF và GNN)
    print("4. Đang chia tập Train/Val/Test (Cùng Seed: 42)...")
    torch.manual_seed(42)
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        edge_types=[('user', 'rates', 'recipe')],
        rev_edge_types=None 
    )
    train_data, val_data, test_data = transform(graph)
    
    train_edges = train_data['user', 'rates', 'recipe'].edge_index.numpy() # [2, num_train_edges]
    test_edges = test_data['user', 'rates', 'recipe'].edge_index.numpy()
    
    # Xây dựng User Profile từ tập Train (Trung bình cộng TF-IDF của các món họ đã đánh giá)
    print("5. Đang xây dựng Hồ sơ Người dùng (User Profiles) từ Lịch sử Huấn luyện...")
    user_profiles = {}
    
    # Nhóm các item theo user trong tập train
    train_users = train_edges[0]
    train_items = train_edges[1]
    
    # Dùng Pandas groupby cho nhanh thay vì vòng lặp for
    train_df = pd.DataFrame({'user': train_users, 'item': train_items})
    user_item_dict = train_df.groupby('user')['item'].apply(list).to_dict()
    
    # Tập Test
    test_users = test_edges[0]
    test_items = test_edges[1]
    
    test_user_targets = {}
    for u, i in zip(test_users, test_items):
        if u not in test_user_targets:
            test_user_targets[u] = []
        test_user_targets[u].append(i)
        
    eval_users = list(test_user_targets.keys())
    targets = [test_user_targets[u] for u in eval_users]
    
    # 4. Dự đoán và Chấm điểm (Làm theo Batch để tránh tràn RAM)
    print("6. Đang tính điểm Tương đồng Cosine (Cosine Similarity) cho tập Test...")
    batch_size = 500
    all_predictions = []
    
    for i in range(0, len(eval_users), batch_size):
        batch_users = eval_users[i:i+batch_size]
        batch_profiles = []
        
        for u in batch_users:
            if u in user_item_dict:
                interacted_items = user_item_dict[u]
                # Trung bình cộng vector TF-IDF
                profile = item_tfidf[interacted_items].mean(axis=0)
            else:
                profile = np.zeros((1, item_tfidf.shape[1]))
            batch_profiles.append(np.asarray(profile).flatten())
            
        batch_profiles_matrix = np.vstack(batch_profiles) # [batch_size, 5000]
        
        # Tính Cosine Similarity giữa batch_profiles và toàn bộ 41240 items
        # Kết quả là [batch_size, 41240]
        sim_scores = cosine_similarity(batch_profiles_matrix, item_tfidf)
        all_predictions.append(torch.tensor(sim_scores))
        
    predictions_tensor = torch.cat(all_predictions, dim=0) # [num_eval_users, 41240]
    
    print("\n--- ĐÁNH GIÁ MÔ HÌNH (EVALUATION) ---")
    metrics = get_metrics(predictions_tensor, targets, k=20, total_items=num_recipes)
    
    print("Kết quả trên Lọc Nội Dung (Content-Based):")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
        
    with open("content_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            
    print(f"Tổng thời gian chạy: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    train_content_based()
