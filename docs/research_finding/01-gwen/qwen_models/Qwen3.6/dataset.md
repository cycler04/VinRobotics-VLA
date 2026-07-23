# Qwen3.6-35B-A3B — Bộ dữ liệu

## 1. Kết luận về mức độ công khai

Qwen3.6 là open-weight nhưng chưa phải open-training. Model card không đưa danh sách đầy đủ tên dataset, số mẫu, số token, tỷ lệ trộn, filtering hay deduplication.

## 2. Các nhóm dữ liệu có thể xác nhận ở mức domain

| Nhóm | Nội dung | Mức độ |
|---|---|---|
| Foundation | Text, code, math, multilingual | Nền tảng kế thừa |
| Multimodal | Image-text, video-text, documents | Công bố qua model type/capability |
| Coding | Repository, bug, patch, diff, frontend | Suy ra từ agentic coding focus |
| Sử dụng công cụ | Thiết bị đầu cuối, trình duyệt, hệ thống tập tin, lệnh gọi MCP/công cụ | Miền năng lực/sau huấn luyện |
| RL environments | Code execution, browser, agent tasks | Mô tả ở mức environment |
| Thinking traces | Historical reasoning và multi-turn trajectories | Gắn với Thinking Preservation |
| Synthetic/verifier data | Teacher outputs, tests, tool feedback | Có thể có, nhưng recipe chưa public |

Không nên ghi tên dataset cụ thể nếu nguồn không xác nhận. Cách diễn đạt an toàn là Qwen3.6 mở rộng hoặc tái trọng số post-training data cho coding và agentic tasks trên nền multimodal Qwen3.5.

## 3. Dataset và evaluation không đồng nhất

Repository/patch/tool trajectory là dạng dữ liệu hợp lý cho post-training, nhưng benchmark như SWE-bench, Terminal-Bench, MCPMark hoặc QwenWebBench không chứng minh chúng được dùng làm training set.

## 4. So với Qwen3.5

| Khía cạnh | Qwen3.5 | Qwen3.6 |
|---|---|---|
| Foundation modalities | Text, code, math, image/video, documents | Kế thừa |
| Nhấn mạnh dữ liệu | Tổng hợp đa phương thức + đại lý | Agent mã hóa, giao diện người dùng, kho lưu trữ, quy trình làm việc của công cụ |
| Thinking history | Chưa có công bố tương đương | Historical thinking trajectories/behavior |
| Dataset names/ratios | Chưa public đầy đủ | Chưa public đầy đủ |
