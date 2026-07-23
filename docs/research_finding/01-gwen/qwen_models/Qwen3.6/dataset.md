# Qwen3.6-35B-A3B — Bộ dữ liệu

## 1. Kết luận về mức độ công khai

Qwen3.6 là mô hình mở trọng số nhưng chưa công khai quy trình huấn luyện. Model card không cung cấp danh sách đầy đủ tên bộ dữ liệu, số mẫu, số token, tỷ lệ trộn, phương pháp lọc hay loại bỏ dữ liệu trùng lặp.

## 2. Các nhóm dữ liệu có thể xác nhận ở cấp miền

| Nhóm | Nội dung | Mức độ |
|---|---|---|
| Nền tảng | Văn bản, mã nguồn, toán học, đa ngôn ngữ | Nền tảng kế thừa |
| Đa phương thức | Cặp ảnh–văn bản, video–văn bản, tài liệu | Được công bố qua loại mô hình/năng lực |
| Lập trình | Kho mã, lỗi, bản vá, diff, frontend | Suy ra từ trọng tâm lập trình có tính tác tử |
| Sử dụng công cụ | Thiết bị đầu cuối, trình duyệt, hệ thống tập tin, lệnh gọi MCP/công cụ | Miền năng lực/sau huấn luyện |
| Môi trường RL | Thực thi mã, trình duyệt, tác vụ tác tử | Được mô tả ở cấp môi trường |
| Dấu vết suy luận | Suy luận lịch sử và quỹ đạo nhiều lượt | Gắn với Thinking Preservation |
| Dữ liệu tổng hợp/bộ xác minh | Đầu ra của mô hình giáo viên, kiểm thử, phản hồi từ công cụ | Có thể có, nhưng công thức xây dựng chưa được công khai |

Không nên ghi tên bộ dữ liệu cụ thể nếu nguồn không xác nhận. Cách diễn đạt an toàn là Qwen3.6 mở rộng hoặc tái trọng số dữ liệu hậu huấn luyện cho các tác vụ lập trình và tác tử trên nền tảng đa phương thức Qwen3.5.

## 3. Bộ dữ liệu và đánh giá không đồng nhất

Quỹ đạo kho mã/bản vá/công cụ là dạng dữ liệu hợp lý cho hậu huấn luyện, nhưng các benchmark như SWE-bench, Terminal-Bench, MCPMark hoặc QwenWebBench không chứng minh rằng chúng được dùng làm tập huấn luyện.

## 4. So với Qwen3.5

| Khía cạnh | Qwen3.5 | Qwen3.6 |
|---|---|---|
| Phương thức nền tảng | Văn bản, mã nguồn, toán học, ảnh/video, tài liệu | Kế thừa |
| Trọng tâm dữ liệu | Tổng hợp đa phương thức + tác tử | Tác tử lập trình, giao diện người dùng, kho mã, quy trình sử dụng công cụ |
| Lịch sử suy luận | Chưa có công bố tương đương | Quỹ đạo/hành vi suy luận lịch sử |
| Tên/tỷ lệ bộ dữ liệu | Chưa công khai đầy đủ | Chưa công khai đầy đủ |
