---
name: inspect-vla-dataset
description: Inspect, validate, and assess VLA datasets and converted artifacts. Use when the user asks to "inspect dataset", "đọc format", "xem schema", "kiểm tra RLDS/LeRobot", "convert sample", "ingest dataset", "dataset này training-ready chưa", "so sánh input/output", or needs evidence about episode boundaries, action/state semantics, timestamps, modalities, conversion loss, and loader readiness.
---

# Inspect VLA dataset

Đọc `.agents/workflows/01_inspect_dataset.md`, `.agents/02_architecture.md` và
`.agents/04_commands.md` trước.

## Thực hiện

1. Xác định source path/format và giữ source immutable.
2. Đo size/file count, rồi inspect 1–2 episode trước; mặc định không decode image.
3. Lập schema map cho boundary, action/state, timestamp, image, language và provenance.
4. Convert sang output riêng khi cần; không ghi đè artifact có giá trị.
5. Validate source/output và ghi exact commands.
6. Nếu người dùng cần lưu kết quả, chuyển sang skill `write-research-report`.

Với artifact lớn, dùng `.agents/agents/01_dataset_artifact_reader.md` làm vai trò subagent để
rút thống kê thay vì nạp dữ liệu thô vào context.

Không gọi output training-ready nếu chưa chứng minh một loader đích đọc được batch và giữ
đúng semantics. Tách rõ verified, inferred, unknown và planned.
