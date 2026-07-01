# Báo Cáo Đánh Giá & So Sánh Hệ Tư Vấn: Baseline (MF) vs Heterogeneous GNN

Trong đồ án này, chúng ta đã tiến hành xây dựng và so sánh hai phương pháp cho bài toán Recommender Systems trên bộ dữ liệu đồ thị Food.com:
1. **Mô hình cơ sở (Baseline):** Matrix Factorization (MF) kết hợp BPR Loss.
2. **Mô hình đề xuất:** Heterogeneous GNN (Mạng nơ-ron đồ thị đa loại) tận dụng CKG (Collaborative Knowledge Graph).

Cả hai mô hình đều được huấn luyện hội tụ (100 Epochs) và đánh giá trên toàn bộ tập khách hàng của tập Test. Để đảm bảo tính chặt chẽ về mặt học thuật, chúng tôi áp dụng 6 độ đo tiêu chuẩn trong hệ tư vấn.

## 1. Kết quả Đánh giá (Metrics)

| Độ đo (Metric) | Ý nghĩa | Matrix Factorization (Baseline) | Heterogeneous GNN (Đề xuất) | Tỷ lệ Cải thiện (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Recall@20** | Tỷ lệ món ngon được AI gợi ý trúng | 0.0470 | **0.0654** | + 39.1% |
| **Precision@20**| Độ chuẩn xác của danh sách Top 20 | 0.0404 | **0.0484** | + 19.8% |
| **NDCG@20** | Chất lượng xếp hạng (Món thích nhất phải ở trên) | 0.0590 | **0.0802** | + 35.9% |
| **MRR@20** | (Mean Reciprocal Rank) Món trúng đích đầu tiên xuất hiện sớm thế nào | 0.1393 | **0.1868** | + 34.1% |
| **Hit Rate@20** | Tỷ lệ User nhận được ít nhất 1 gợi ý đúng gu | 0.4090 | **0.4999** | + 22.2% |
| **Coverage@20** | Độ phủ danh mục (Tỷ lệ món ăn được AI lấy ra gợi ý) | 0.1190 | **0.5211** | **+ 337.8%** |

> **Nhận xét Tổng quan:** Kết quả thực nghiệm cho thấy sự vượt trội hoàn toàn của mô hình **Heterogeneous GNN**. Đặc biệt, chỉ số Coverage tăng đột biến hơn 300%, chứng tỏ GNN đã khắc phục thành công yếu điểm cốt lõi của các hệ thống cũ: Sự thiên kiến phổ biến (Popularity Bias).

## 2. Lý luận bảo vệ: Tại sao GNN vượt trội hơn MF một cách rõ rệt? (Phục vụ thuyết trình)

Từ bảng số liệu trên, chúng ta có thể khẳng định và bảo vệ thuật toán GNN trước hội đồng thông qua 3 luận điểm toán học và cấu trúc dữ liệu sau:

### A. Giải quyết Vấn đề Khởi động lạnh và Dữ liệu thưa (Data Sparsity)
Nhìn vào chỉ số **Coverage** (0.1190 so với 0.5211), ta thấy mô hình MF chỉ loanh quanh gợi ý ~11% số lượng món ăn trong hệ thống (thường là các món phổ biến nhất). Ngược lại, GNN khai thác được tới 52% số lượng món ăn. 
- **Giải thích:** GNN tận dụng thông tin "bên lề" (Side information) thông qua đồ thị tri thức CKG. Kể cả khi món ăn hiếm (Long-tail items) chưa có ai rate, GNN vẫn tiếp nhận thông tin từ các node `Ingredient` (Nguyên liệu) và `Tag` (Đặc tính). Nhờ vậy, Vector nhúng (Embedding) của những món ăn này vẫn cực kỳ chất lượng và sẵn sàng được gợi ý cho những User có cùng sở thích nguyên liệu.

### B. Khai thác Kết nối bậc cao (High-order Connectivity) qua Message Passing
Mô hình MF chỉ giới hạn ở mức kết nối bậc 1 (`User -> Recipe`). Trong khi đó, với GNN:
- Lượt thích của *User A* truyền năng lượng cho *Món X*.
- Từ *Món X*, năng lượng truyền sang node *Nguyên liệu B* (ví dụ: Thịt Gà).
- Từ *Nguyên liệu B*, năng lượng truyền ngược về *Món Y*.
Chính sự lan truyền thông điệp (Message Passing) này tự động phát hiện ra các User có cùng Gu ẩm thực, giúp đẩy các món ăn trúng đích lên vị trí cao hơn. Điều này lý giải tại sao chỉ số **MRR** (xếp hạng món trúng đích đầu tiên) tăng từ 0.1393 lên **0.1868** và **NDCG** tăng mạnh **35.9%**.

### C. Khả năng tích hợp Tư vấn NLP tự nhiên (NLP Integration)
- Khác với MF (một hộp đen thuần tuý), cấu trúc Đồ thị Đa loại sinh ra một không gian vector (Embedding Space) thống nhất cho cả User, Recipe, Ingredient và Tag. 
- Điều này cho phép chúng tôi phát triển thành công module **Tư vấn bằng ngôn ngữ tự nhiên**. Khi người dùng nhập yêu cầu (VD: *"Tôi thèm món gà cay"*), hệ thống lập tức tính toán **Cosine Similarity** giữa toạ độ của `chicken`, `spicy` với 41.000 món ăn để chọn ra Top 5. Đây là minh chứng rõ rệt nhất cho trí thông minh mà Đồ thị mang lại so với ma trận truyền thống.

---
**Kết luận:** Việc đưa Collaborative Knowledge Graph và Heterogeneous GNN vào bài toán gợi ý món ăn không chỉ mang tính học thuật cao, mà còn giải quyết được những bài toán thực tiễn cốt lõi (Sparsity, Coverage). Kết quả 6 độ đo chuẩn quốc tế đã minh chứng đây là một hệ thống AI xuất sắc cho đồ án này.
