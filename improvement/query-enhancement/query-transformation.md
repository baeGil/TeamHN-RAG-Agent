### 1. Viết lại truy vấn (Query Rewriting)
*   **Mục đích:** Kỹ thuật này nhằm diễn đạt lại truy vấn gốc để nó trở nên **cụ thể và chi tiết hơn**, từ đó tăng khả năng tìm thấy các tài liệu thực sự có liên quan.
*   **Cách thức thực hiện:** Hệ thống sử dụng mô hình GPT-4 kết hợp với một mẫu gợi ý (prompt template) tùy chỉnh. Mô hình sẽ lấy truy vấn ban đầu và xây dựng lại nó với nhiều chi tiết hơn.
*   **Ví dụ:** Với truy vấn *"Tác động của biến đổi khí hậu đối với môi trường là gì?"*, kỹ thuật này có thể mở rộng nó để bao gồm các khía cạnh cụ thể như sự thay đổi nhiệt độ và đa dạng sinh học.
*   **Lợi ích:** Cải thiện tính chính xác và độ liên quan của thông tin được truy xuất.

### 2. Gợi ý lùi lại (Step-back Prompting)
*   **Mục đích:** Tạo ra các truy vấn **rộng hơn và khái quát hơn** so với câu hỏi gốc để giúp hệ thống tìm kiếm được các thông tin nền tảng và ngữ cảnh tốt hơn.
*   **Cách thức thực hiện:** Tương tự như kỹ thuật trên, nó sử dụng GPT-4 và các mẫu gợi ý để tạo ra một truy vấn "lùi lại" mang tính tổng quát từ câu hỏi ban đầu.
*   **Ví dụ:** Thay vì hỏi chi tiết về tác động, truy vấn có thể được khái quát hóa thành: *"Các hiệu ứng chung của biến đổi khí hậu là gì?"*.
*   **Lợi ích:** Giúp hệ thống nắm bắt được bối cảnh rộng hơn mà các truy vấn quá chi tiết có thể bỏ lỡ.

### 3. Phân tách thành các truy vấn con (Sub-query Decomposition)
*   **Mục đích:** Chia nhỏ một truy vấn phức tạp thành **2-4 truy vấn con đơn giản hơn** để có thể tìm kiếm thông tin một cách toàn diện từ nhiều khía cạnh khác nhau.
*   **Cách thức thực hiện:** GPT-4 sẽ phân tích câu hỏi phức tạp và bóc tách nó thành các câu hỏi thành phần.
*   **Ví dụ:** Với câu hỏi về tác động của biến đổi khí hậu, kỹ thuật này sẽ chia nó thành các câu hỏi riêng biệt về: đa dạng sinh học, đại dương, các kiểu thời tiết và môi trường trên đất liền.
*   **Lợi ích:** Đảm bảo kết quả tìm kiếm bao phủ được tất cả các khía cạnh của một vấn đề phức tạp, giúp câu trả lời cuối cùng đầy đủ hơn.

### Tổng kết lợi ích của các phương pháp
Các kỹ thuật này mang lại khả năng **linh hoạt cao**, có thể sử dụng độc lập hoặc kết hợp tùy thuộc vào từng trường hợp cụ thể. Chúng đặc biệt có giá trị trong các lĩnh vực có truy vấn phức tạp hoặc đa diện như nghiên cứu khoa học, phân tích pháp lý hoặc các tác vụ tìm kiếm sự thật toàn diện. Tất cả đều được triển khai thông qua các hàm riêng biệt trong mã nguồn, cho phép dễ dàng tích hợp vào các hệ thống RAG hiện có.