---
name: user-preferences
---

# User preferences

- Người dùng làm R&D VLA, cần hiểu dataset format, processing và ingestion để tạo dữ liệu
  training-ready.
- Kết quả đọc hiểu nên được đóng gói thành báo cáo Markdown có thể tìm và đọc lại, vì giá trị
  nằm ở knowledge tích lũy chứ không chỉ câu trả lời trong chat.
- Agent được chủ động chạy code, tải và xóa dữ liệu/artifact trong workspace. Lý do là giảm
  vòng lặp xin phép cho các thao tác nghiên cứu thông thường.
- Vẫn hỏi trước commit/push, Git init, thao tác ngoài workspace hoặc thay đổi hệ thống bên
  ngoài.
- Knowledge, plans, workflows và skills của workspace phải ở `.agents/`, không ở global hay
  thư mục tool-specific.
- Khi tổ chức tài liệu/paper theo yêu cầu rõ, giữ đúng reading order/taxonomy người dùng đưa;
  tên nhóm nên ngắn, English nếu đó là convention đã chọn.

Xem correction cụ thể tại [[corrections]].
