import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import ast
import pickle
import torch
from torch_geometric.data import HeteroData

def preprocess_and_build_graph(interactions_path, recipes_path, output_graph_path, output_mapping_path):
    print("1. Đọc dữ liệu...")
    # Đọc dữ liệu từ file CSV
    interactions = pd.read_csv(interactions_path)
    recipes = pd.read_csv(recipes_path)

    print("2. Lọc dữ liệu (5-core filtering) và xử lý null...")
    # Xóa các dòng có giá trị rỗng ở các cột quan trọng
    interactions = interactions.dropna(subset=['user_id', 'recipe_id', 'rating'])
    recipes = recipes.dropna(subset=['id', 'tags', 'ingredients'])
    
    # Áp dụng 5-core filtering: Lặp lại cho đến khi cả user và recipe đều xuất hiện >= 5 lần
    iteration = 1
    while True:
        user_counts = interactions['user_id'].value_counts()
        recipe_counts = interactions['recipe_id'].value_counts()
        
        valid_users = user_counts[user_counts >= 5].index
        valid_recipes = recipe_counts[recipe_counts >= 5].index
        
        # Nếu số lượng hợp lệ bằng với số lượng hiện tại, tức là không có gì bị loại đi -> Dừng vòng lặp
        if len(valid_users) == len(user_counts) and len(valid_recipes) == len(recipe_counts):
            break
            
        print(f"  - Lọc lần {iteration}: Giữ lại {len(valid_users)} users và {len(valid_recipes)} recipes.")
        interactions = interactions[
            (interactions['user_id'].isin(valid_users)) & 
            (interactions['recipe_id'].isin(valid_recipes))
        ]
        iteration += 1
        
    print(f"Sau khi lọc: {len(interactions)} tương tác, {interactions['user_id'].nunique()} users, {interactions['recipe_id'].nunique()} recipes.")

    # Lọc lại bảng recipes để chỉ giữ các món ăn còn lại sau 5-core filtering
    recipes = recipes[recipes['id'].isin(interactions['recipe_id'])]

    print("3. Trích xuất đặc trưng (Feature Extraction)...")
    # ast.literal_eval dùng để chuyển đổi an toàn chuỗi danh sách dạng "['chicken', 'garlic']" thành Python list thật
    recipes['tags_list'] = recipes['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    recipes['ingredients_list'] = recipes['ingredients'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    print("4. Xây dựng Mapping ID cho các Node...")
    # Tạo mapping chuyển ID thực tế thành một chuỗi ID liên tục bắt đầu từ 0 cho GNN
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(interactions['user_id'].unique())}
    recipe_mapping = {old_id: new_id for new_id, old_id in enumerate(recipes['id'].unique())}
    
    # Lấy toàn bộ danh sách nguyên liệu và tags duy nhất (bằng cách dùng set comprehension)
    all_ingredients = set(ing for sublist in recipes['ingredients_list'] for ing in sublist)
    all_tags = set(tag for sublist in recipes['tags_list'] for tag in sublist)
    
    ingredient_mapping = {name: idx for idx, name in enumerate(all_ingredients)}
    tag_mapping = {name: idx for idx, name in enumerate(all_tags)}

    print(f"Số lượng Nodes - User: {len(user_mapping)}, Recipe: {len(recipe_mapping)}, Ingredient: {len(ingredient_mapping)}, Tag: {len(tag_mapping)}")

    print("5. Tạo các Cạnh (Edges)...")
    # --- Cạnh 1: User -> (rates) -> Recipe ---
    mapped_user_ids = interactions['user_id'].map(user_mapping).values
    mapped_recipe_ids = interactions['recipe_id'].map(recipe_mapping).values
    
    # Định dạng edge_index của PyTorch: matrix kích thước [2, số lượng cạnh]
    edge_index_user_recipe = torch.tensor(np.vstack((mapped_user_ids, mapped_recipe_ids)), dtype=torch.long)
    # Trọng số của cạnh lấy từ lượt rating
    edge_weight_user_recipe = torch.tensor(interactions['rating'].values, dtype=torch.float)

    # --- Cạnh 2: Recipe -> (contains) -> Ingredient ---
    # --- Cạnh 3: Recipe -> (has_attribute) -> Tag ---
    recipe_ingr_edges = []
    recipe_tag_edges = []
    
    for _, row in recipes.iterrows():
        rec_id = recipe_mapping[row['id']]
        
        # Liên kết với nguyên liệu
        for ing in row['ingredients_list']:
            ing_id = ingredient_mapping[ing]
            recipe_ingr_edges.append((rec_id, ing_id))
            
        # Liên kết với tag
        for tag in row['tags_list']:
            tag_id = tag_mapping[tag]
            recipe_tag_edges.append((rec_id, tag_id))

    edge_index_recipe_ingr = torch.tensor(recipe_ingr_edges, dtype=torch.long).t().contiguous()
    edge_index_recipe_tag = torch.tensor(recipe_tag_edges, dtype=torch.long).t().contiguous()

    print("6. Đóng gói vào HeteroData của PyTorch Geometric...")
    data = HeteroData()
    
    # Khai báo số lượng node cho từng loại để HeteroData quản lý
    data['user'].num_nodes = len(user_mapping)
    data['recipe'].num_nodes = len(recipe_mapping)
    data['ingredient'].num_nodes = len(ingredient_mapping)
    data['tag'].num_nodes = len(tag_mapping)
    
    # Đưa các matrix edge_index và weights vào object
    data['user', 'rates', 'recipe'].edge_index = edge_index_user_recipe
    data['user', 'rates', 'recipe'].edge_attr = edge_weight_user_recipe
    
    data['recipe', 'contains', 'ingredient'].edge_index = edge_index_recipe_ingr
    data['recipe', 'has_attribute', 'tag'].edge_index = edge_index_recipe_tag
    
    print("7. Lưu dữ liệu ra file...")
    # Lưu lại để lần sau đem ra training mô hình (ví dụ: dùng LightGCN)
    torch.save(data, output_graph_path)
    
    # Lưu từ điển ID mapping để phục vụ bước Recommender/NLP
    mappings = {
        'user': user_mapping,
        'recipe': recipe_mapping,
        'ingredient': ingredient_mapping,
        'tag': tag_mapping
    }
    with open(output_mapping_path, 'wb') as f:
        pickle.dump(mappings, f)
        
    print("Hoàn tất! Dữ liệu đã được lưu tại:", output_graph_path, "và", output_mapping_path)
    return data, mappings

# ==========================================
# Phần mô phỏng xử lý truy vấn tự nhiên (NLP) và Gợi ý cơ bản
# ==========================================
def simulate_nlp_query(query, mapping_path, recipes_df=None):
    """
    Minh hoạ cách hệ thống ánh xạ (map) từ câu văn của người dùng 
    thành các Node trên đồ thị.
    Trong thực tế, bước này sẽ dùng LLM (ví dụ: ChatGPT) hoặc mô hình NLP chuyên dụng (NER, PhoBERT).
    """
    print(f"\n[{'='*50}]")
    print(f"  MÔ PHỎNG NLP XỬ LÝ TRUY VẤN CỦA NGƯỜI DÙNG")
    print(f"[{'='*50}]")
    print(f"Khách hàng nhập: '{query}'\n")
    
    # Tải lại mappings để tra cứu ID
    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
        
    # [Giả lập] Một từ điển nhỏ (Dictionary) dịch từ tiếng Việt sang entity tiếng Anh trong Dataset
    # Thực tế bạn sẽ dùng API của ChatGPT hoặc các mô hình PhoBERT để làm việc này tự động và thông minh hơn.
    demo_dict = {
        "gà": {"type": "ingredient", "en": "chicken"},
        "bò": {"type": "ingredient", "en": "beef"},
        "heo": {"type": "ingredient", "en": "pork"},
        "cá": {"type": "ingredient", "en": "fish"},
        "trứng": {"type": "ingredient", "en": "egg"},
        "cay": {"type": "tag", "en": "spicy"},
        "nhanh": {"type": "tag", "en": "15-minutes-or-less"},
        "nướng": {"type": "tag", "en": "oven"},
        "ngọt": {"type": "tag", "en": "sweet"},
        "chay": {"type": "tag", "en": "vegan"},
        "lành mạnh": {"type": "tag", "en": "healthy"}
    }
    
    extracted_ingredients = []
    extracted_tags = []
    
    # Tìm kiếm từ khoá trong câu truy vấn (chuyển về chữ thường để dễ so sánh)
    query_lower = query.lower()
    for vn_word, data in demo_dict.items():
        if vn_word in query_lower:
            if data["type"] == "ingredient":
                extracted_ingredients.append(data["en"])
            else:
                extracted_tags.append(data["en"])
                
    # Nếu không tìm thấy gì, gán mặc định để khỏi bị lỗi hiển thị
    if not extracted_ingredients and not extracted_tags:
        print(">> BƯỚC 1: Mô hình NLP chưa nhận diện được từ khoá nào trong từ điển demo.")
        return
    
    print(">> BƯỚC 1: Mô hình NLP phân tách thành các Entity:")
    print(f"   - Ingredient Entities : {extracted_ingredients}")
    print(f"   - Tag Entities        : {extracted_tags}\n")
    
    # Map sang graph IDs
    active_ingredient_nodes = []
    for ing in extracted_ingredients:
        if ing in mappings['ingredient']:
            active_ingredient_nodes.append((ing, mappings['ingredient'][ing]))
            
    active_tag_nodes = []
    for tag in extracted_tags:
        if tag in mappings['tag']:
            active_tag_nodes.append((tag, mappings['tag'][tag]))
            
    print(">> BƯỚC 2: Kích hoạt các Đỉnh (Nodes) tương ứng trên Đồ Thị GNN:")
    print(f"   - Node Ingredients được chọn : {active_ingredient_nodes}")
    print(f"   - Node Tags được chọn        : {active_tag_nodes}\n")
    
    print(">> BƯỚC 3: Đầu vào cho Mô Hình GNN:")
    print("   Hệ thống GNN thực tế sẽ dùng các Node ID này để chạy Random Walk hoặc Message Passing.")
    print("   Từ đó tính vector nhúng (embeddings) và tính cosine similarity để tìm ra Recipe gần nhất.\n")
    
    print(">> BƯỚC 4: Gợi ý Món ăn (Mô phỏng bằng Graph Traversal / Rule-based cho Demo):")
    if recipes_df is not None and (extracted_ingredients or extracted_tags):
        # Tính điểm phù hợp đơn giản: Nếu món ăn có chứa ingredient yêu cầu thì +2 điểm, chứa tag thì +1 điểm.
        def calc_score(row):
            score = 0
            if 'ingredients_list' in row and isinstance(row['ingredients_list'], list):
                for ing in extracted_ingredients:
                    if ing in row['ingredients_list']: score += 2
            if 'tags_list' in row and isinstance(row['tags_list'], list):
                for tag in extracted_tags:
                    if tag in row['tags_list']: score += 1
            return score
            
        # Áp dụng tính điểm
        scores = recipes_df.apply(calc_score, axis=1)
        recipes_df_scored = recipes_df.copy()
        recipes_df_scored['match_score'] = scores
        
        # Lọc ra top 5 món có điểm cao nhất
        top_recipes = recipes_df_scored[recipes_df_scored['match_score'] > 0].sort_values(by='match_score', ascending=False).head(5)
        
        if len(top_recipes) > 0:
            print("   🎉 CÁC MÓN ĂN GỢI Ý CHO BẠN:")
            for _, row in top_recipes.iterrows():
                print(f"   🍲 Tên món : {row['name']} (Độ phù hợp: {row['match_score']} điểm)")
                print(f"      - TG làm: {row['minutes']} phút")
                print(f"      - Tags  : {row['tags_list'][:3]}...")
                print("")
        else:
            print("   Rất tiếc, không tìm thấy món ăn nào khớp với yêu cầu trong dataset.")
    else:
        print("   (Không có dữ liệu Recipe hoặc không nhận diện được keyword để gợi ý).")


if __name__ == "__main__":
    # Cấu hình đường dẫn file
    interactions_csv = "RAW_interactions.csv"
    recipes_csv = "RAW_recipes.csv"
    
    # Nơi lưu file sau khi chạy
    graph_pt = "food_graph.pt"
    mappings_pkl = "mappings.pkl"
    
    print("Bắt đầu khởi chạy Script tiền xử lý Dataset cho GNN...")
    try:
        # Gọi hàm tiền xử lý chính
        # Chú ý: Việc xử lý có thể mất 1-3 phút do dữ liệu lớn, cần RAM lớn (> 4GB)
        data, map_dict = preprocess_and_build_graph(interactions_csv, recipes_csv, graph_pt, mappings_pkl)
        
        # Load lại dataframe recipes để phục vụ hiển thị tên món ăn
        print("\nĐang tải lại dữ liệu Recipes để phục vụ hiển thị demo...")
        recipes_df = pd.read_csv(recipes_csv)
        import ast
        recipes_df['tags_list'] = recipes_df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        recipes_df['ingredients_list'] = recipes_df['ingredients'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        print("\nCấu trúc Đồ thị đa loại (HeteroData) đã tạo:")
        print(data)
        
        # Minh hoạ NLP Pipeline với input từ người dùng
        while True:
            user_input = input("\n[?] Mời bạn nhập câu truy vấn món ăn (VD: 'cho tôi món bò nướng cay' hoặc nhập 'q' để thoát): ")
            if user_input.strip().lower() in ['q', 'quit', 'exit']:
                print("Đã thoát mô phỏng NLP.")
                break
            if user_input.strip() == "":
                continue
                
            simulate_nlp_query(user_input, mappings_pkl, recipes_df)
        
    except Exception as e:
        print("Đã xảy ra lỗi:", e)
