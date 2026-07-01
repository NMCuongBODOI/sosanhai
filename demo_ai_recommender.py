import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
import ast
import os

def load_data():
    print("Đang tải dữ liệu đồ thị và mô hình...")
    # Tải mappings (từ điển)
    with open("mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
        
    # Tải GNN Embeddings (Vector đã được huấn luyện)
    if not os.path.exists("gnn_embeddings.pt"):
        print("LỖI: Không tìm thấy file gnn_embeddings.pt. Bạn cần chạy 'python train_gnn.py' trước!")
        sys.exit(1)
        
    embeddings = torch.load("gnn_embeddings.pt", map_location='cpu')
    
    # Tải danh sách món ăn (để in tên và mô tả)
    recipes_df = pd.read_csv("RAW_recipes.csv")
    recipes_df['tags_list'] = recipes_df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    recipes_df['ingredients_list'] = recipes_df['ingredients'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Lọc lại df chỉ lấy những món ăn có trong đồ thị
    valid_recipe_ids = set(mappings['recipe'].keys())
    recipes_df = recipes_df[recipes_df['id'].isin(valid_recipe_ids)]
    
    return mappings, embeddings, recipes_df

def nlp_entity_extraction(query):
    # Bộ từ điển giả lập (Mock NER / Translation)
    demo_dict = {
        "gà": {"type": "ingredient", "en": "chicken"},
        "bò": {"type": "ingredient", "en": "beef"},
        "heo": {"type": "ingredient", "en": "pork"},
        "lợn": {"type": "ingredient", "en": "pork"},
        "cá": {"type": "ingredient", "en": "fish"},
        "trứng": {"type": "ingredient", "en": "egg"},
        "cà chua": {"type": "ingredient", "en": "tomato"},
        "cay": {"type": "tag", "en": "spicy"},
        "nhanh": {"type": "tag", "en": "15-minutes-or-less"},
        "nướng": {"type": "tag", "en": "oven"},
        "ngọt": {"type": "tag", "en": "sweet"},
        "chay": {"type": "tag", "en": "vegan"},
        "lành mạnh": {"type": "tag", "en": "healthy"}
    }
    
    extracted_ingredients = []
    extracted_tags = []
    
    query_lower = query.lower()
    for vn_word, data in demo_dict.items():
        if vn_word in query_lower:
            if data["type"] == "ingredient":
                extracted_ingredients.append(data["en"])
            else:
                extracted_tags.append(data["en"])
                
    return extracted_ingredients, extracted_tags

def recommend_recipes(query, mappings, embeddings, recipes_df, top_k=5):
    print(f"\n[AI] Phân tích yêu cầu: '{query}'")
    
    # 1. Trích xuất NLP
    ingredients, tags = nlp_entity_extraction(query)
    print(f"  > Nhận diện thành phần: {ingredients}")
    print(f"  > Nhận diện đặc tính : {tags}")
    
    if not ingredients and not tags:
        print("  > Lỗi: Không nhận diện được từ khoá nào. Vui lòng thử từ khoá khác (ví dụ: bò nướng cay).")
        return

    # 2. Lấy Embeddings của các Nodes tương ứng
    query_vectors = []
    
    for ing in ingredients:
        if ing in mappings['ingredient']:
            node_id = mappings['ingredient'][ing]
            vector = embeddings['ingredient'][node_id]
            query_vectors.append(vector)
            
    for tag in tags:
        if tag in mappings['tag']:
            node_id = mappings['tag'][tag]
            vector = embeddings['tag'][node_id]
            query_vectors.append(vector)
            
    if not query_vectors:
        print("  > Lỗi: Các từ khoá này chưa có trong không gian dữ liệu đồ thị!")
        return
        
    # 3. Kết hợp (Aggregation) để tạo thành Query Vector
    # Gộp tất cả vector lại bằng cách lấy trung bình cộng
    query_vector = torch.stack(query_vectors).mean(dim=0).unsqueeze(0) # [1, hidden_dim]
    
    # Lấy toàn bộ vector của 41.000 món ăn
    all_recipe_vectors = embeddings['recipe'] # [num_recipes, hidden_dim]
    
    # 4. Tính độ tương đồng (Cosine Similarity)
    print("  > Đang quét không gian Vector GNN để tìm sự tương đồng...")
    similarities = F.cosine_similarity(query_vector, all_recipe_vectors) # [num_recipes]
    
    # Lấy top K món ăn có góc Cosine nhỏ nhất (độ tương đồng cao nhất)
    top_scores, top_indices = torch.topk(similarities, k=top_k)
    
    print("\n" + "="*50)
    print("🎉 KẾT QUẢ TƯ VẤN (DỰA TRÊN ĐỒ THỊ GNN)")
    print("="*50)
    
    # 5. In kết quả
    for rank, (score, node_id) in enumerate(zip(top_scores, top_indices)):
        # Tìm ID thật của món ăn từ mapped node_id
        # node_id là một tensor (khi lấy item() sẽ ra số nguyên)
        node_id_int = node_id.item()
        
        # Cần tìm ngược từ new_id ra old_id (recipe ID nguyên gốc)
        real_recipe_id = None
        for old_id, n_id in mappings['recipe'].items():
            if n_id == node_id_int:
                real_recipe_id = old_id
                break
                
        if real_recipe_id is not None:
            # Tra cứu thông tin trong DataFrame
            recipe_info = recipes_df[recipes_df['id'] == real_recipe_id].iloc[0]
            print(f"⭐ Top {rank+1}: {recipe_info['name'].title()}")
            print(f"   - Độ tương đồng: {score.item():.4f}")
            print(f"   - Thời gian làm: {recipe_info['minutes']} phút")
            print(f"   - Tags: {', '.join(recipe_info['tags_list'][:4])}...")
            print(f"   - Nguyên liệu: {', '.join(recipe_info['ingredients_list'][:4])}...")
            print("-" * 30)

if __name__ == "__main__":
    print("Khởi động Hệ thống Tư vấn AI Recommender...")
    try:
        mappings, embeddings, recipes_df = load_data()
        print("Sẵn sàng!")
        
        while True:
            user_input = input("\n[Bạn] Mời nhập câu (VD: 'Tôi thèm gà nướng' hoặc 'q' để thoát): ")
            if user_input.strip().lower() in ['q', 'quit', 'exit']:
                print("Đã thoát AI.")
                break
            if user_input.strip() == "":
                continue
                
            recommend_recipes(user_input, mappings, embeddings, recipes_df)
            
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
