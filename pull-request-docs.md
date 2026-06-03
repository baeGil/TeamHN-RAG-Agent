# Tài liệu Pull Request

## Tiêu đề

Cập nhật giao diện xem PDF trong danh sách tài liệu

## Tóm tắt thay đổi

- Bỏ nút biểu tượng con mắt trong danh sách tài liệu PDF.
- Cho phép người dùng nhấn trực tiếp vào tên tài liệu PDF để mở file.
- Khi rê chuột hoặc focus bằng bàn phím vào tên tài liệu PDF, tên sẽ được gạch chân để thể hiện đây là nội dung có thể nhấn.
- Giữ nguyên nút xoá tài liệu và cách hiển thị URL/văn bản.

## Lý do thay đổi

Giao diện cũ dùng biểu tượng con mắt để xem PDF, nhưng tín hiệu hành động chưa thật rõ ràng với người dùng. Việc biến tên tài liệu PDF thành liên kết trực tiếp giúp thao tác tự nhiên hơn: người dùng nhìn thấy tên tài liệu và có thể nhấn vào chính tên đó để mở file.

Hiệu ứng gạch chân khi hover/focus giúp người dùng nhận biết đây là vùng có thể tương tác mà không cần thêm biểu tượng phụ.

## Phạm vi chỉnh sửa

- `frontend/src/components/Sidebar.tsx`
  - Thay phần tên PDF từ thẻ hiển thị tĩnh thành nút dạng text.
  - Gọi `window.open(api.documentPdfUrl(d.id), "_blank", "noopener,noreferrer")` khi nhấn vào tên PDF.
  - Loại bỏ nút con mắt trong `doc-actions`.

- `frontend/src/styles.css`
  - Thêm style cho `.doc-title-link`.
  - Thêm underline khi hover/focus.
  - Thêm focus outline để hỗ trợ thao tác bằng bàn phím.

## Kiểm thử

Đã chạy:

```bash
npm run build
```

Kết quả: build thành công.

Lưu ý: Vite có cảnh báo bundle lớn hơn 500 kB sau khi minify. Cảnh báo này không liên quan đến thay đổi giao diện xem PDF.

## Checklist

- [x] Bỏ biểu tượng con mắt khỏi tài liệu PDF.
- [x] Tên tài liệu PDF có thể nhấn để mở PDF.
- [x] Hover/focus vào tên PDF có gạch chân.
- [x] Không thay đổi API backend.
- [x] Không ảnh hưởng thao tác xoá tài liệu.
- [x] Build frontend thành công.
