import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit
from metrics import get_metrics
import time

# ==========================================
# 1. Định nghĩa Mạng nơ-ron Đồ thị (GNN Layer)
# ==========================================
class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # 2 lớp SAGEConv để học thông tin lân cận (2-hop neighborhood)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class HeteroGNNRecommender(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_nodes_dict):
        super().__init__()
        # Khởi tạo Node Embeddings (Đặc trưng nhúng ban đầu) cho tất cả các loại Node
        self.node_emb = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes, hidden_channels)
            for node_type, num_nodes in num_nodes_dict.items()
        })
        
        # Mạng GNN cơ bản
        self.encoder = GNNEncoder(hidden_channels, out_channels)
        
        # Chuyển đổi thành Mạng đa loại (Heterogeneous) tự động
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')

    def forward(self, edge_index_dict):
        # Lấy embeddings hiện tại của tất cả các node
        x_dict = {node_type: emb.weight for node_type, emb in self.node_emb.items()}
        
        # Đưa qua mạng GNN để thông tin truyền đi dọc theo các cạnh
        # (Ingredient/Tag -> Recipe -> User)
        x_dict = self.encoder(x_dict, edge_index_dict)
        return x_dict

# ==========================================
# 2. BPR Loss
# ==========================================
def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

# ==========================================
# 3. Chu trình Huấn luyện GNN
# ==========================================
def train_gnn(graph_path="food_graph.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    print("Đang tải dữ liệu đồ thị đa loại (Heterogeneous Graph)...")
    graph = torch.load(graph_path, map_location=device)
    
    # Chuyển đồ thị thành vô hướng (Undirected) để Message Passing đi được 2 chiều
    # (user <-> recipe, recipe <-> ingredient)
    graph = T.ToUndirected()(graph)
    
    # 3.1. Chia tập Dữ liệu
    print("Chia tập tương tác thành Train/Val/Test...")
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        edge_types=[('user', 'rates', 'recipe')],
        rev_edge_types=[('recipe', 'rev_rates', 'user')] 
    )
    train_data, val_data, test_data = transform(graph)
    
    num_nodes_dict = {
        'user': graph['user'].num_nodes,
        'recipe': graph['recipe'].num_nodes,
        'ingredient': graph['ingredient'].num_nodes,
        'tag': graph['tag'].num_nodes
    }
    
    train_edge_index_dict = train_data.edge_index_dict
    user_recipe_train_edges = train_edge_index_dict[('user', 'rates', 'recipe')]
    
    # 3.2. Khởi tạo mô hình GNN
    hidden_dim = 32
    model = HeteroGNNRecommender(hidden_dim, hidden_dim, graph.metadata(), num_nodes_dict).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    epochs = 100 # Tăng số vòng huấn luyện lên 100 để AI "khôn" hơn
    batch_size = 2048
    num_recipes = num_nodes_dict['recipe']
    
    print("\n--- BẮT ĐẦU HUẤN LUYỆN HETEROGENEOUS GNN ---")
    for epoch in range(epochs):
        model.train()
        
        start_time = time.time()
        
        # 1 Lần Forward GNN cho toàn đồ thị mỗi Epoch
        optimizer.zero_grad()
        out_dict = model(train_edge_index_dict)
        user_emb = out_dict['user']
        recipe_emb = out_dict['recipe']
        
        # Lấy mẫu số cạnh nhất định để train nhanh (VD: Lấy ngẫu nhiên 100,000 cạnh thay vì toàn bộ 400,000)
        # Giúp demo chạy nhanh trên CPU mà vẫn học được nhiều dữ liệu
        sample_size = min(100000, user_recipe_train_edges.size(1))
        perm = torch.randperm(user_recipe_train_edges.size(1))[:sample_size]
        
        users = user_recipe_train_edges[0, perm]
        pos_recipes = user_recipe_train_edges[1, perm]
        
        # Negative Sampling
        neg_recipes = torch.randint(0, num_recipes, (len(users),), device=device)
        
        # Tính điểm
        pos_scores = (user_emb[users] * recipe_emb[pos_recipes]).sum(dim=1)
        neg_scores = (user_emb[users] * recipe_emb[neg_recipes]).sum(dim=1)
        
        # Tính loss chung cho toàn bộ sample và backward 1 lần duy nhất
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f} | Thời gian: {time.time()-start_time:.2f}s")

    # 3.3. Đánh giá (Evaluation)
    print("\n--- ĐÁNH GIÁ MÔ HÌNH (EVALUATION) ---")
    model.eval()
    
    with torch.no_grad():
        out_dict = model(train_edge_index_dict) # Thông qua GNN lần cuối
        final_user_emb = out_dict['user']
        final_recipe_emb = out_dict['recipe']
        
    test_edges = test_data['user', 'rates', 'recipe'].edge_index.cpu()
    test_users = test_edges[0].numpy()
    test_items = test_edges[1].numpy()
    
    user_targets = {}
    for u, i in zip(test_users, test_items):
        if u not in user_targets:
            user_targets[u] = []
        user_targets[u].append(i)
        
    eval_users = list(user_targets.keys())
    targets = [user_targets[u] for u in eval_users]
    
    users_tensor = torch.tensor(eval_users, dtype=torch.long, device=device)
    
    with torch.no_grad():
        u_e = final_user_emb[users_tensor]
        r_e = final_recipe_emb
        predictions = torch.matmul(u_e, r_e.t())
        
    metrics = get_metrics(predictions, targets, k=20)
    print("Kết quả trên Heterogeneous GNN:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
        
    with open("gnn_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            
    # --- LƯU EMBEDDINGS CHO PHẦN DEMO ---
    print("\nLưu các Vector nhúng (Embeddings) để phục vụ cho Demo Inference...")
    embeddings = {
        'user': final_user_emb.cpu(),
        'recipe': final_recipe_emb.cpu(),
        'ingredient': out_dict['ingredient'].cpu(),
        'tag': out_dict['tag'].cpu()
    }
    torch.save(embeddings, "gnn_embeddings.pt")
    print("Đã lưu thành công tại 'gnn_embeddings.pt'")

if __name__ == "__main__":
    train_gnn()
