Kỹ thuật được đề cập trong tài liệu bạn cung cấp là **"Tăng cường tài liệu thông qua việc tạo câu hỏi để cải thiện khả năng truy xuất" (Document Augmentation through Question Generation for Enhanced Retrieval)**.

Dưới đây là giải thích chi tiết về cách hoạt động của kỹ thuật này:

### 1. Mục tiêu và Động lực
Mục tiêu chính của kỹ thuật này là nâng cao độ chính xác khi xác định các phần tài liệu có liên quan nhất để trả lời truy vấn của người dùng. Bằng cách làm phong phú các đoạn văn bản bằng những câu hỏi liên quan, hệ thống tăng khả năng tìm thấy các tài liệu phù hợp để làm ngữ cảnh cho mô hình ngôn ngữ lớn (LLM) tạo ra câu trả lời.

### 2. Các bước thực hiện chi tiết
Quy trình này bao gồm các giai đoạn chính sau:

*   **Tiền xử lý tài liệu (Document Preprocessing):**
    *   Chuyển đổi tệp PDF thành chuỗi văn bản (string).
    *   Chia văn bản thành các "tài liệu văn bản" (text documents) có phần chồng lấp để xây dựng ngữ cảnh.
    *   Tiếp tục chia nhỏ mỗi tài liệu đó thành các "đoạn văn bản" (text fragments) có phần chồng lấp để phục vụ cho mục đích truy xuất và tìm kiếm ngữ nghĩa.

*   **Tăng cường bằng câu hỏi (Question Augmentation):**
    *   Sử dụng các mô hình ngôn ngữ của OpenAI để tạo ra các câu hỏi liên quan ở cấp độ tài liệu hoặc đoạn văn bản.
    *   Số lượng câu hỏi được tạo ra có thể điều chỉnh thông qua các hằng số cấu hình (ví dụ: `QUESTIONS_PER_DOCUMENT`).

*   **Tạo kho lưu trữ vector (Vector Store Creation):**
    *   Sử dụng mô hình embedding của OpenAI để tính toán các vector đặc trưng cho các tài liệu và câu hỏi đã được tăng cường.
    *   Lưu trữ các vector này vào một kho lưu trữ vector FAISS.

*   **Truy xuất và Tạo câu trả lời (Retrieval and Generation):**
    *   Khi có truy vấn, hệ thống sẽ tìm kiếm tài liệu FAISS có liên quan nhất. Trong hầu hết các trường hợp, kết quả khớp nhất sẽ là **một câu hỏi được tăng cường** thay vì đoạn văn bản gốc.
    *   Sau khi tìm thấy đoạn/câu hỏi liên quan, hệ thống sẽ tìm ngược lại **tài liệu văn bản gốc (parent document)** chứa nó để lấy toàn bộ ngữ cảnh.
    *   Sử dụng ngữ cảnh đầy đủ này để mô hình ngôn ngữ tạo ra câu trả lời cuối cùng cho người dùng.

### 3. Lợi ích của kỹ thuật
*   **Cải thiện quá trình truy xuất:** Tăng xác suất tìm thấy tài liệu FAISS phù hợp nhất cho một truy vấn cụ thể vì các câu hỏi được tạo ra thường có cấu trúc tương đồng với cách người dùng đặt câu hỏi.
*   **Điều chỉnh ngữ cảnh linh hoạt:** Cho phép dễ dàng thay đổi kích thước cửa sổ ngữ cảnh cho cả tài liệu và các đoạn văn bản nhỏ.
*   **Hiểu ngôn ngữ chất lượng cao:** Tận dụng sức mạnh của các mô hình OpenAI để hiểu và tạo ra các câu hỏi cũng như câu trả lời chính xác.

### 4. Các thành phần kỹ thuật chính
*   **Lớp OpenAIEmbeddingsWrapper:** Cung cấp giao diện nhất quán để tạo các vector embedding.
*   **Hàm `generate_questions`:** Sử dụng các mô hình chat của OpenAI để tạo câu hỏi từ văn bản.
*   **Hàm `process_documents`:** Xử lý logic cốt lõi từ việc chia nhỏ tài liệu, tạo câu hỏi cho đến xây dựng kho lưu trữ vector.

Tóm lại, thay vì chỉ lưu trữ văn bản gốc, kỹ thuật này "dự đoán" trước những gì người dùng có thể hỏi và lưu trữ chính những câu hỏi đó dưới dạng vector để việc đối sánh ngữ nghĩa trở nên hiệu quả hơn.

https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/document_augmentation.ipynb