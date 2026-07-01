# Báo Cáo Đánh Giá & So Sánh Hệ Tư Vấn: Baseline (MF) vs Heterogeneous GNN

Trong đồ án này, chúng ta đã tiến hành xây dựng và so sánh hai phương pháp cho bài toán Recommender Systems trên bộ dữ liệu đồ thị Food.com:
1. **Mô hình cơ sở (Baseline):** Matrix Factorization (MF) kết hợp BPR Loss.
2. **Mô hình đề xuất:** Heterogeneous GNN (Mạng nơ-ron đồ thị đa loại) tận dụng CKG (Collaborative Knowledge Graph).

Cả hai mô hình đều được huấn luyện hội tụ (100 Epochs) và đánh giá trên toàn bộ tập khách hàng của tập Test.

## 1. Kết quả Đánh giá (Metrics)

| Metric | Matrix Factorization (Baseline) | Heterogeneous GNN (Đề xuất) | Tỷ lệ Cải thiện |
| :--- | :---: | :---: | :---: |
| **Recall@20** | 0.0468 | **0.0621** | + 32.6% |
| **Precision@20**| 0.0405 | **0.0477** | + 17.7% |
| **NDCG@20** | 0.0584 | **0.0783** | + 34.0% |

> **Nhận xét:** Kết quả thực nghiệm cho thấy sự vượt trội hoàn toàn của mô hình **Heterogeneous GNN**. Tất cả các chỉ số đo lường hiệu suất (Precision, Recall) và đo lường thứ hạng (NDCG) đều tăng từ 17% đến 34% so với mô hình lọc cộng tác truyền thống.

## 2. Lý luận bảo vệ: Tại sao GNN vượt trội hơn MF một cách rõ rệt? (Phục vụ thuyết trình)

Từ kết quả trên, chúng ta có thể khẳng định và bảo vệ thuật toán GNN trước hội đồng thông qua các luận điểm toán học và cấu trúc dữ liệu sau:

### A. Khắc phục triệt để bài toán Dữ liệu thưa (Data Sparsity)
- **Matrix Factorization (MF):** MF chỉ "nhìn" thấy các tương tác trực tiếp (`User -> Recipe`). Đối với những món ăn mới ít người đánh giá (hoặc người dùng mới), ma trận tương tác có quá nhiều số 0, dẫn tới MF không đủ dữ liệu để tạo ra một Latent Vector chính xác (Cold-Start Problem).
- **Heterogeneous GNN:** GNN giải quyết vấn đề này bằng cách tận dụng thông tin "bên lề" (Side information) thông qua CKG. Kể cả khi món ăn chưa có ai rate, GNN vẫn tiếp nhận thông tin từ các node `Ingredient` (Nguyên liệu) và `Tag` (Đặc tính). Nhờ vậy, Vector nhúng (Embedding) của món ăn luôn chứa tri thức hữu ích, giúp nó dễ dàng lọt vào danh sách gợi ý của những User có cùng sở thích nguyên liệu.

### B. Khai thác Kết nối bậc cao (High-order Connectivity) qua Message Passing
- MF chỉ tối ưu hóa dựa trên liên kết bậc 1 (User A thích Món X).
- Cơ chế **Message Passing** của GNN cho phép thông tin truyền xa hơn trong đồ thị (liên kết bậc 2, bậc 3...):
  - Lượt thích của *User A* truyền năng lượng cho *Món X*.
  - Từ *Món X*, năng lượng truyền sang node *Nguyên liệu B* (ví dụ: Thịt Gà).
  - Từ *Nguyên liệu B*, năng lượng truyền ngược về *Món Y* (dù User A chưa từng tương tác với Món Y).
- Chính nhờ cơ chế "trạm trung chuyển" này, GNN tự động gom nhóm được các User có cùng Gu ẩm thực, giúp chỉ số **NDCG** (xếp hạng gợi ý) tăng vọt lên **0.0783** so với **0.0584** của Baseline.

### C. Khả năng giải thích và Tích hợp NLP (Explainability & Integration)
- Khác với MF (một hộp đen thuần tuý), cấu trúc Đồ thị Đa loại cho phép chúng ta dễ dàng xây dựng một tính năng **Tư vấn bằng ngôn ngữ tự nhiên**. 
- Trong sản phẩm Demo của đồ án, khi người dùng nhập Text (NLP), hệ thống trực tiếp ánh xạ Text thành các node `Ingredient/Tag`, sau đó sử dụng **Cosine Similarity** trên không gian nhúng của GNN để truy xuất món ăn. Đây là quy trình hoàn chỉnh và hiện đại nhất trong các hệ thống Recommender Systems thực tế.

---
**Kết luận:** Việc xây dựng mô hình Heterogeneous GNN kết hợp Collaborative Knowledge Graph không chỉ là một giải pháp phức tạp về mặt thuật toán mà còn chứng minh được hiệu quả thực tế vượt trội so với các kỹ thuật Baseline. Đây hoàn toàn xứng đáng là giải pháp trọng tâm cho đồ án hệ tư vấn AI.
