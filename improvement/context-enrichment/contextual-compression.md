Kỹ thuật được đề cập trong tài liệu là **Nén ngữ cảnh (Contextual Compression)** trong truy xuất tài liệu. Đây là một phương pháp tiên tiến nhằm tối ưu hóa hệ thống RAG (Retrieval-Augmented Generation) bằng cách tập trung vào các thông tin thực sự có giá trị.

Dưới đây là giải thích chi tiết về kỹ thuật này dựa trên các nguồn tin:

### 1. Khái niệm và Mục tiêu
**Nén ngữ cảnh** là kỹ thuật trích xuất và nén những phần phù hợp nhất của tài liệu dựa trên ngữ cảnh của một câu hỏi cụ thể. Thay vì trả về toàn bộ các đoạn văn bản (chunks) thô như các hệ thống truyền thống, kỹ thuật này tinh lọc nội dung để chỉ giữ lại những thông tin cốt lõi nhất.

### 2. Lý do cần sử dụng (Động lực)
Các hệ thống truy xuất tài liệu truyền thống thường trả về toàn bộ khối văn bản, trong đó có thể chứa nhiều thông tin gây nhiễu hoặc không liên quan đến câu hỏi. Việc này không chỉ làm giảm độ chính xác mà còn khiến mô hình ngôn ngữ lớn (LLM) phải xử lý nhiều văn bản thừa, gây lãng phí tài nguyên và giảm hiệu quả,.

### 3. Các thành phần chính của hệ thống
Để thực hiện kỹ thuật này, hệ thống cần các thành phần sau:
*   **Kho lưu trữ vector (Vector store):** Nơi lưu trữ tài liệu đã được mã hóa.
*   **Bộ truy xuất cơ sở (Base retriever):** Dùng để tìm nạp các tài liệu thô ban đầu.
*   **Bộ nén ngữ cảnh dựa trên LLM (LLM-based contextual compressor):** Thường sử dụng các công cụ như `LLMChainExtractor` (trong ví dụ là dùng GPT-4) để phân tích và trích xuất nội dung.
*   **Bộ truy xuất nén ngữ cảnh (Contextual compression retriever):** Sự kết hợp giữa bộ truy xuất cơ sở và bộ nén.
*   **Chuỗi hỏi đáp (Question-answering chain):** Thành phần cuối cùng sử dụng dữ liệu đã nén để trả lời người dùng.

### 4. Quy trình hoạt động chi tiết
1.  **Tiền xử lý:** Tài liệu (ví dụ: file PDF) được xử lý và mã hóa vào kho lưu trữ vector.
2.  **Truy xuất thô:** Khi có câu hỏi, bộ truy xuất cơ sở sẽ tìm các tài liệu có liên quan nhất từ kho lưu trữ.
3.  **Nén và Trích xuất:** Bộ nén LLM sẽ xem xét các tài liệu vừa tìm được và chỉ trích xuất những câu hoặc đoạn văn thực sự trả lời cho câu hỏi đó.
4.  **Tạo câu trả lời:** Chuỗi QA nhận thông tin đã được tinh lọc này để tạo ra câu trả lời chính xác và ngắn gọn.

### 5. Lợi ích của kỹ thuật
*   **Cải thiện tính liên quan:** Hệ thống chỉ trả về thông tin bám sát nhất với truy vấn của người dùng.
*   **Tăng hiệu suất:** Giảm đáng kể lượng văn bản mà LLM cần phải đọc và xử lý, giúp tiết kiệm chi phí và thời gian.
*   **Hiểu ngữ cảnh sâu hơn:** Nhờ sử dụng LLM trong quá trình nén, hệ thống có khả năng hiểu ý định của người dùng tốt hơn để trích xuất thông tin phù hợp.
*   **Tính linh hoạt:** Có thể dễ dàng áp dụng cho nhiều loại tài liệu và câu hỏi khác nhau.

Tóm lại, nén ngữ cảnh mang lại một cách tiếp cận mạnh mẽ để nâng cao chất lượng và hiệu quả của các hệ thống truy xuất thông tin hiện đại.

https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_compression.ipynb