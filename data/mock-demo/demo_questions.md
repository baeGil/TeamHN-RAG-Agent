# Câu hỏi demo conflict

## Demo 1 - Mâu thuẫn ngày tháng

Hỏi:

```text
SEA Games 31 khai mạc tại Mỹ Đình vào ngày nào?
```

Kỳ vọng:

- UI hiện panel kiểm tra mâu thuẫn nguồn.
- Conflict giữa nguồn nói ngày 12/5/2022 và nguồn nói ngày 21/5/2022.
- Câu trả lời nên nêu rõ tài liệu đang mâu thuẫn, không chốt một ngày duy nhất nếu không có lý do ưu tiên nguồn.

## Demo 2 - Mâu thuẫn số liệu tài chính

Hỏi:

```text
Doanh thu năm tài khóa 2025 của Công ty A là bao nhiêu?
```

Kỳ vọng:

- UI phát hiện conflict giữa 1.500 tỷ đồng và 1.200 tỷ đồng.
- User có thể bấm chọn nguồn báo cáo quản trị hoặc báo cáo kiểm toán trong panel conflict.

## Demo 3 - Mâu thuẫn quy định

Hỏi:

```text
Học sinh được học thêm tối đa bao nhiêu buổi mỗi tuần cho mỗi môn học?
```

Kỳ vọng:

- UI phát hiện conflict giữa 3 buổi/tuần và 2 buổi/tuần.
- Câu trả lời nên nhắc có quy chế cũ và quy định mới.

## Demo 4 - Không có conflict

Hỏi:

```text
Hệ thống chấm công của Công ty B hoạt động như thế nào?
```

Kỳ vọng:

- UI hiện đã đối chiếu các cặp và không phát hiện mâu thuẫn.
- Câu trả lời tổng hợp thông tin QR, báo cáo, đồng bộ dữ liệu.

## Demo 5 - Câu hỏi tổng hợp để show nhiều conflict

Hỏi:

```text
Hãy tổng hợp các thông tin quan trọng trong bộ tài liệu demo và chỉ rõ nếu có nguồn nào mâu thuẫn.
```

Kỳ vọng:

- Có thể hiện nhiều cặp conflict cùng lúc.
- Phù hợp để demo panel UI, source chips và citation drawer.
