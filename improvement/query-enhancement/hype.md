Kỹ thuật được đề cập chi tiết trong các nguồn tài liệu bạn cung cấp là **Hypothetical Prompt Embeddings (HyPE)**. Đây là một khung làm việc (framework) mới dành cho các hệ thống **Retrieval-Augmented Generation (RAG)**, nhằm tối ưu hóa quá trình truy xuất thông tin bằng cách giải quyết khoảng cách về phong cách giữa câu hỏi của người dùng và văn bản tài liệu.

Dưới đây là giải thích chi tiết về kỹ thuật này:

### 1. Vấn đề cốt lõi: Khoảng cách Câu hỏi - Câu trả lời
Trong các hệ thống RAG truyền thống (Naive RAG), có một sự không khớp (mismatch) về phong cách: câu hỏi của người dùng thường ở dạng **nghi vấn**, trong khi nội dung tài liệu thường ở dạng **trình bày hoặc khẳng định**. Sự khác biệt này làm giảm khả năng căn chỉnh (alignment) giữa các vector nhúng (embeddings) của câu hỏi và tài liệu, dẫn đến việc bỏ lỡ các thông tin quan trọng.

### 2. Cách tiếp cận của HyPE
Thay vì cố gắng khớp câu hỏi với văn bản tài liệu, HyPE chuyển đổi quá trình truy xuất thành việc **khớp câu hỏi với câu hỏi (question-to-question matching)**.

#### Quy trình thực hiện chia làm hai giai đoạn:

**Giai đoạn Indexing (Ngoại tuyến - Offline):**
*   **Chia nhỏ dữ liệu (Chunking):** Toàn bộ kho văn bản được chia thành các đoạn nhỏ (chunks) có ý nghĩa.
*   **Tạo câu hỏi giả định:** Với mỗi đoạn văn bản, một mô hình ngôn ngữ lớn (LLM) sẽ tạo ra **nhiều câu hỏi giả định** mà đoạn văn bản đó có thể trả lời được.
*   **Nhúng và lưu trữ:** Các câu hỏi này được chuyển thành các vector nhúng (embeddings). Thay vì lưu trữ vector của chính đoạn văn bản, hệ thống lưu trữ các vector của các câu hỏi giả định này và liên kết chúng với đoạn văn bản gốc. Một đoạn văn bản có thể được đại diện bởi nhiều vector câu hỏi khác nhau, giúp mở rộng "phạm vi ngữ nghĩa".

**Giai đoạn Truy xuất (Trực tuyến - Online):**
*   Khi người dùng đặt câu hỏi, hệ thống sẽ nhúng câu hỏi đó và thực hiện tìm kiếm các vector lân cận (ANN search) trong tập hợp các **câu hỏi giả định** đã lưu.
*   Vì cả câu hỏi của người dùng và các vector được lưu trữ đều ở dạng nghi vấn, chúng có xu hướng nằm gần nhau hơn trong không gian vector do sự tương đồng về phong cách (style-based clustering).
*   Sau khi tìm được câu hỏi giả định khớp nhất, hệ thống sẽ lấy đoạn văn bản gốc tương ứng để đưa vào LLM tạo câu trả lời cuối cùng.

### 3. Ưu điểm so với các kỹ thuật khác (như HyDE)
HyPE thường được so sánh với **HyDE (Hypothetical Document Embeddings)**, nhưng có những cải tiến quan trọng:
*   **Không gây trễ (No Latency):** HyDE tạo ra một câu trả lời giả định tại thời điểm người dùng đặt câu hỏi (runtime), điều này làm tăng chi phí tính toán và độ trễ cho mỗi yêu cầu. Ngược lại, HyPE thực hiện việc tạo câu hỏi từ trước trong quá trình lập chỉ mục (indexing), nên tốc độ truy xuất lúc thực tế nhanh như RAG truyền thống.
*   **Độ chính xác cao hơn:** Các thử nghiệm cho thấy HyPE có thể cải thiện **độ chính xác ngữ cảnh (context precision)** lên tới 42 điểm phần trăm và **khả năng thu hồi (recall)** lên tới 45 điểm phần trăm so với các phương pháp tiêu chuẩn.
*   **Giảm ảo giác:** Do việc truy xuất được căn chỉnh tốt hơn, thông tin đưa vào LLM chính xác hơn, giúp giảm tỷ lệ tạo ra các khẳng định sai lệch hoặc không có căn cứ (hallucination).

### 4. Kết quả thực nghiệm
Dựa trên các nguồn tài liệu, HyPE cho thấy hiệu suất vượt trội trên nhiều tập dữ liệu khác nhau (như WikiQA, MS MARCO, RAGBench). Nó đặc biệt hiệu quả trong việc:
*   **Tăng tính trung thực (Faithfulness):** Đảm bảo câu trả lời được tạo ra bám sát vào dữ liệu truy xuất được.
*   **Tận dụng ngữ cảnh:** Giúp LLM sử dụng thông tin từ các đoạn văn bản được truy xuất một cách hiệu quả hơn.

Tóm lại, **HyPE** là một kỹ thuật tối ưu hóa RAG bằng cách "chuẩn bị trước" các câu hỏi tiềm năng cho dữ liệu, giúp hệ thống hiểu và tìm kiếm thông tin theo cách tự nhiên và chính xác hơn mà không làm chậm quá trình phản hồi của người dùng.