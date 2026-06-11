# Tài liệu Hướng dẫn Kiểm thử Conflict & Non-Conflict với Mock Data PDF

Các tài liệu PDF mock đã được tạo tự động và nạp thành công vào cơ sở dữ liệu tri thức của hệ thống. Bạn có thể sử dụng các câu hỏi dưới đây để kiểm thử các trường hợp **xung đột dữ liệu** (Conflict) và **không xung đột** (Non-conflict) thông qua giao diện Chat của hệ thống.

---

## 📂 Danh sách các file Mock PDF đã nạp
1. [alpha_policy_2024.pdf](file:///d:/TeamHN-RAG-Agent/data/mock_conflict_pdfs/files/alpha_policy_2024.pdf): Chính sách công ty Alpha năm 2024 (Trợ cấp ăn trưa: 50.000 VNĐ/ngày).
2. [alpha_policy_2026.pdf](file:///d:/TeamHN-RAG-Agent/data/mock_conflict_pdfs/files/alpha_policy_2026.pdf): Chính sách công ty Alpha năm 2026 (Trợ cấp ăn trưa: 80.000 VNĐ/ngày).
3. [rag_expert_opinions.pdf](file:///d:/TeamHN-RAG-Agent/data/mock_conflict_pdfs/files/rag_expert_opinions.pdf): Báo cáo công nghệ chứa ý kiến trái chiều của Giáo sư A (ủng hộ RAG) và Tiến sĩ B (ủng hộ Fine-tuning).
4. [alpha_recruitment.pdf](file:///d:/TeamHN-RAG-Agent/data/mock_conflict_pdfs/files/alpha_recruitment.pdf): Thông tin tuyển dụng Kỹ sư AI (Yêu cầu: 2 năm kinh nghiệm Python, lương 20M - 40M VNĐ).
5. [alpha_benefits.pdf](file:///d:/TeamHN-RAG-Agent/data/mock_conflict_pdfs/files/alpha_benefits.pdf): Quy chế đãi ngộ (Cấp laptop cao cấp, bảo hiểm PVI 100%, tài trợ học chứng chỉ).
6. [alpha_trip_2025.pdf](file:///d:/TeamHN-RAG-Agent/data/mock_conflict_pdfs/files/alpha_trip_2025.pdf): Thông báo nghỉ mát hè năm 2025 tại **Nha Trang** từ ngày 15/07 - 18/07/2025.

---

## 🧪 Kịch bản kiểm thử (Test Cases)

### 1. Xung đột Thời gian / Phiên bản (Freshness Conflict)
* **Khái niệm:** Dữ liệu thay đổi theo thời gian/phiên bản ở các tài liệu khác nhau. Hệ thống cần ưu tiên nguồn mới nhất hoặc phân tách rõ ràng theo mốc thời gian được hỏi.
* **Câu hỏi thử nghiệm 1:** `Mức trợ cấp ăn trưa của công ty Alpha hiện nay là bao nhiêu?`
  * **Phân loại Conflict kỳ vọng từ DRAG:** `freshness`
  * **Hành vi/Câu trả lời kỳ vọng:** Hệ thống cần nhận diện được chính sách 2026 mới hơn chính sách 2024 và trả lời là **80.000 VNĐ/ngày** (đồng thời có thể đề cập mức cũ năm 2024 là 50.000 VNĐ).
* **Câu hỏi thử nghiệm 2:** `So sánh mức trợ cấp ăn trưa của công ty Alpha vào năm 2024 và năm 2026?`
  * **Phân loại Conflict kỳ vọng từ DRAG:** `freshness` (hoặc `no_conflict` do câu hỏi đã phân tách rõ thời gian).
  * **Hành vi/Câu trả lời kỳ vọng:** Trả lời rõ ràng năm 2024 là 50.000 VNĐ/ngày và năm 2026 là 80.000 VNĐ/ngày.

### 2. Tiền đề sai / Thông tin sai lệch (Misinformation / False Premise)
* **Khái niệm:** Câu hỏi của người dùng mang tiền đề sai lệch so với nguồn tài liệu thực tế, hoặc tài liệu có thông tin mâu thuẫn/sai lệch rõ ràng.
* **Câu hỏi thử nghiệm:** `Tại sao công ty Alpha lại tổ chức đi nghỉ mát hè 2025 ở Phú Quốc?`
  * **Phân loại Conflict kỳ vọng từ DRAG:** `misinformation`
  * **Hành vi/Câu trả lời kỳ vọng:** Hệ thống cần đính chính lại thông tin sai trong câu hỏi: Công ty Alpha tổ chức nghỉ mát ở **Nha Trang** chứ không phải Phú Quốc (theo thông báo nghỉ mát hè 2025).

### 3. Xung đột quan điểm (Conflicting Opinions)
* **Khái niệm:** Các nguồn tài liệu chứa các ý kiến/nhận định chủ quan hoặc kết quả nghiên cứu trái chiều. Hệ thống cần trình bày một cách trung lập các quan điểm thay vì chọn một bên làm sự thật khách quan.
* **Câu hỏi thử nghiệm:** `RAG hay Fine-tuning là phương pháp tốt nhất để xây dựng AI cho doanh nghiệp?`
  * **Phân loại Conflict kỳ vọng từ DRAG:** `conflicting_opinions`
  * **Hành vi/Câu trả lời kỳ vọng:** Trình bày trung lập hai luồng ý kiến: Giáo sư Nguyễn Văn A cho rằng RAG tối ưu hơn do chi phí thấp và cập nhật nhanh; trong khi Tiến sĩ Trần Thị B khẳng định Fine-tuning tốt hơn để mô hình hiểu sâu văn phong và nghiệp vụ doanh nghiệp.

### 4. Thông tin bổ sung (Complementary Information - Không xung đột)
* **Khái niệm:** Các tài liệu cung cấp các thông tin khác nhau nhưng tương thích và bổ trợ cho nhau để trả lời đầy đủ câu hỏi của người dùng.
* **Câu hỏi thử nghiệm:** `Quyền lợi và yêu cầu công việc khi ứng tuyển vị trí Kỹ sư AI tại công ty Alpha là gì?`
  * **Phân loại Conflict kỳ vọng từ DRAG:** `complementary_information`
  * **Hành vi/Câu trả lời kỳ vọng:** Hệ thống tự động gộp thông tin tuyển dụng (lương 20M-40M, 2 năm kinh nghiệm Python/RAG) từ file tuyển dụng và thông tin đãi ngộ (cấp laptop, bảo hiểm PVI, đào tạo) từ file đãi ngộ để đưa ra câu trả lời toàn diện.

### 5. Không xung đột (No Conflict)
* **Khái niệm:** Các nguồn thông tin đồng nhất hoặc không có sự mâu thuẫn nào về mặt sự thật.
* **Câu hỏi thử nghiệm 1:** `Thời gian làm việc chính thức của công ty Alpha là từ mấy giờ đến mấy giờ?`
  * **Phân loại Conflict kỳ vọng từ DRAG:** `no_conflict`
  * **Hành vi/Câu trả lời kỳ vọng:** Trả lời trực tiếp và nhất quán từ cả hai tài liệu chính sách 2024 và 2026: Từ **8:00 đến 17:00**.
* **Câu hỏi thử nghiệm 2:** `Yêu cầu kinh nghiệm đối với vị trí Kỹ sư AI tại Alpha là gì?`
  * **Phân loại Conflict kỳ vọng từ DRAG:** `no_conflict`
  * **Hành vi/Câu trả lời kỳ vọng:** Trả lời trực tiếp: Tối thiểu **2 năm kinh nghiệm** làm việc với Python và hệ sinh thái PyTorch/TensorFlow.
