import os
from pathlib import Path
import fitz

# Output directory for mock PDFs
OUTPUT_DIR = Path(__file__).resolve().parent / "files"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = "C:/Windows/Fonts/arial.ttf"
FONT_NAME = "arial"

def create_pdf(filename, title, content):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_font(fontname=FONT_NAME, fontfile=FONT_PATH)
    
    # Margin of 50 units
    title_rect = fitz.Rect(50, 50, 550, 90)
    body_rect = fitz.Rect(50, 110, 550, 750)
    
    # Write Title
    page.insert_textbox(title_rect, title, fontname=FONT_NAME, fontsize=16, align=0)
    # Write Content
    page.insert_textbox(body_rect, content, fontname=FONT_NAME, fontsize=12, align=0)
    
    pdf_path = OUTPUT_DIR / filename
    doc.save(str(pdf_path))
    doc.close()
    print(f"Created PDF: {pdf_path}")

# 1. Temporal / Version Conflict (freshness) - Outdated Document
create_pdf(
    "alpha_policy_2024.pdf",
    "CHÍNH SÁCH VÀ QUY ĐỊNH CỦA CÔNG TY ALPHA NĂM 2024",
    "1. Trợ cấp ăn trưa:\n"
    "Kể từ ngày 01/01/2024, công ty Alpha áp dụng mức trợ cấp ăn trưa cho toàn bộ nhân viên chính thức là 50.000 VNĐ/ngày làm việc thực tế.\n\n"
    "2. Thời gian làm việc:\n"
    "Thời giờ làm việc tiêu chuẩn của công ty là từ 8:00 đến 17:00 (nghỉ trưa từ 12:00 đến 13:00) từ thứ Hai đến thứ Sáu hằng tuần."
)

# 2. Temporal / Version Conflict (freshness) - Current Document
create_pdf(
    "alpha_policy_2026.pdf",
    "CHÍNH SÁCH VÀ QUY ĐỊNH CỦA CÔNG TY ALPHA NĂM 2026",
    "1. Trợ cấp ăn trưa (CẬP NHẬT MỚI):\n"
    "Kể từ ngày 01/01/2026, công ty Alpha điều chỉnh tăng mức trợ cấp ăn trưa cho nhân viên lên thành 80.000 VNĐ/ngày làm việc thực tế.\n\n"
    "2. Thời gian làm việc:\n"
    "Thời giờ làm việc tiêu chuẩn của công ty Alpha tiếp tục được giữ nguyên từ 8:00 đến 17:00 hằng tuần."
)

# 3. Opinion Conflict (conflicting_opinions)
create_pdf(
    "rag_expert_opinions.pdf",
    "BÁO CÁO CÔNG NGHỆ: Ý KIẾN CHUYÊN GIA VỀ RAG VÀ FINE-TUNING",
    "Hệ thống RAG và Fine-tuning (Tinh chỉnh mô hình) là hai hướng tiếp cận chính để xây dựng AI nội bộ.\n\n"
    "Ý kiến 1: Giáo sư Nguyễn Văn A nhận định rằng: 'RAG là giải pháp tối ưu nhất cho các ứng dụng hỏi đáp doanh nghiệp lớn vì chi phí triển khai thấp, không cần tài nguyên tính toán cao để huấn luyện lại, và quan trọng nhất là có thể cập nhật tri thức mới tức thời bằng cách thay đổi cơ sở dữ liệu nguồn.'\n\n"
    "Ý kiến 2: Ngược lại, Tiến sĩ Trần Thị B khẳng định rằng: 'Fine-tuning (Tinh chỉnh) mới là phương pháp tốt nhất để tối ưu hóa mô hình ngôn ngữ lớn cho doanh nghiệp. Việc tinh chỉnh giúp mô hình tiếp thu sâu sắc văn phong, thuật ngữ chuyên ngành và các quy tắc nghiệp vụ phức tạp của công ty mà kỹ thuật RAG đơn thuần không thể đáp ứng được.'"
)

# 4. Complementary Information - Document 1
create_pdf(
    "alpha_recruitment.pdf",
    "THÔNG BÁO TUYỂN DỤNG KỸ SƯ AI TẠI ALPHA",
    "Công ty Alpha đang tìm kiếm 03 Kỹ sư AI tài năng gia nhập đội ngũ.\n\n"
    "Yêu cầu công việc:\n"
    "- Ứng viên có tối thiểu 2 năm kinh nghiệm làm việc với ngôn ngữ lập trình Python và hệ sinh thái PyTorch/TensorFlow.\n"
    "- Có kinh nghiệm xây dựng hệ thống RAG hoặc xử lý ngôn ngữ tự nhiên.\n\n"
    "Mức lương:\n"
    "- Mức lương khởi điểm dao động từ 20.000.000 VNĐ đến 40.000.000 VNĐ tùy thuộc vào năng lực và kinh nghiệm thực tế."
)

# 5. Complementary Information - Document 2
create_pdf(
    "alpha_benefits.pdf",
    "QUYỀN LỢI VÀ CHẾ ĐỘ ĐÃI NGỘ DÀNH CHO NHÂN VIÊN ALPHA",
    "Khi trở thành nhân viên chính thức tại công ty Alpha, bạn sẽ nhận được các chế độ đãi ngộ sau:\n\n"
    "- Được cấp laptop cấu hình cao (MacBook Pro hoặc ThinkPad Workstation) để làm việc.\n"
    "- Bảo hiểm sức khỏe cao cấp toàn diện từ đối tác PVI (công ty chi trả 100% chi phí mua bảo hiểm).\n"
    "- Cơ hội tham gia các khóa đào tạo chuyên môn, chứng chỉ quốc tế do công ty tài trợ toàn phần."
)

# 6. Misinformation / False Premise - Source Document
create_pdf(
    "alpha_trip_2025.pdf",
    "THÔNG BÁO VỀ KẾ HOẠCH NGHỈ MÁT HÈ NĂM 2025",
    "Ban Giám đốc công ty Alpha thông báo kế hoạch tổ chức kỳ nghỉ mát hè năm 2025 cho toàn thể nhân viên như sau:\n\n"
    "- Địa điểm nghỉ dưỡng: Thành phố Nha Trang, tỉnh Khánh Hòa.\n"
    "- Thời gian tổ chức: Từ ngày 15/07/2025 đến hết ngày 18/07/2025.\n"
    "- Đối tượng tham gia: Toàn bộ cán bộ nhân viên đã ký hợp đồng lao động chính thức tính đến tháng 6/2025."
)

print("\nAll mock PDFs generated successfully in the 'files' subdirectory!")
