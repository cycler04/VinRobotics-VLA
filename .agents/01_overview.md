# Tổng quan workspace

## Mục tiêu

Workspace phục vụ R&D về Vision-Language-Action (VLA): đọc paper, thử nghiệm có
bằng chứng, tổng hợp phát hiện và viết báo cáo. Code đang chạy được chỉ là một
phần nhỏ của phạm vi đó: bộ công cụ đọc, kiểm tra và chuyển đổi dataset VLA.

Ưu tiên hiện tại:

1. Kết luận nghiên cứu phải truy vết được tới paper, code hoặc log thử nghiệm.
2. Thử nhỏ, đo được và tái lập được trước khi chạy dataset hay workload lớn.
3. Báo cáo phải phân biệt dữ kiện, diễn giải và giả thuyết.

Thông tin về cách cộng tác và giai đoạn dự án nằm duy nhất tại
[memory/MEMORY.md](memory/MEMORY.md).

## Trạng thái thật

- Đây là Git repository hợp lệ, branch chính hiện là `main`; không tự commit hoặc
  push.
- Workspace đang chứa thay đổi nghiên cứu chưa commit. Luôn xem `git status
  --short` trước khi sửa và không ghi đè file đang dở.
- Kho tài liệu gồm paper, ghi chú và research finding về VLA, Qwen và real-time
  action chunking.
- Package `vla-data-tools` hiện hỗ trợ:
  `LeRobot v2/v3 | RLDS -> canonical episode v0.1 -> inspect | HDF5 | Parquet`.
- `third_party/01_real-time-chunking-kinetix/` là Git submodule/vendor có quy ước
  và môi trường riêng.

Luồng code và layout output được mô tả tại
[02_architecture.md](02_architecture.md). Lệnh đã xác minh và trạng thái môi
trường nằm tại [04_commands.md](04_commands.md).

## Khoảng cách giữa tên gọi và thực tế

Tên repo và roadmap nghiên cứu rộng hơn code hiện có. Chưa có model
training/inference, robot execution, training loader, writer LeRobot,
HDF5/Parquet reader hay round-trip hoàn chỉnh. `notes/sprint1_task.md` là roadmap,
không phải danh sách capability đã hoàn thành.

`Roadmap.md` hiện là tài liệu 3D Gaussian Splatting/Kaggle chưa được track và
không khớp trực tiếp pipeline VLA hiện tại. Không dùng nó làm kiến trúc mặc định
nếu chưa xác nhận phạm vi của tác vụ.

## Các bẫy nguy hiểm

1. Dataset có thể rất lớn. `LeRobotReader.episodes(max_episodes=...)` vẫn đọc
   toàn bộ Parquet vào RAM trước khi giới hạn episode; Parquet writer cũng tích
   toàn bộ frame trong RAM. Luôn đo dung lượng và smoke test nhỏ trước.
2. Full extra `.[rlds,dev]` kéo TensorFlow/CUDA nặng. Với test LeRobot/HDF5,
   không mặc định cài RLDS nếu không cần.
3. Tài liệu có chỗ hứa nhiều hơn code: guide trỏ tới
   `docs/dataset_converter_report.md` không tồn tại và nói có output thật trong
   `output/`, nhưng output bị ignore và không có trong checkout.
4. RLDS có thể dùng timestamp tổng hợp. Dataset không có profile mặc định 10 Hz;
   đây là giả định, không phải ground truth.
5. `scripts/download_vla_sample.sh` gọi thẳng `.venv/bin/python`; chưa có `.venv`
   thì lệnh thất bại ngay.
6. Docstring của `src/local_video_server.py` ghi sai đường dẫn chạy là
   `scripts/local_video_server.py`.

## Adapter

Hiện chỉ có adapter Codex. Chưa cần chừa adapter cho agent khác theo lựa chọn của
người dùng. Nếu thêm agent mới, agent đó phải tự xác minh file neo, memory,
workflow, subagent và permission của chính nó; không sao chép giả định từ Codex.

