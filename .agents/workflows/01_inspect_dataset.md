# Workflow: inspect và ingest dataset VLA

Mục tiêu là tạo bằng chứng rằng một source có thể được đọc đúng semantics và chuyển thành
artifact mà training loader có thể dùng, không chỉ tạo được file output.

## 1. Chốt phạm vi và bảo toàn source

- Xác định format, path, revision/license và sample cần đọc.
- Đo disk/file count; không scan hoặc decode toàn bộ khi sample nhỏ đủ trả lời câu hỏi.
- Không sửa source dataset. Tạo output ở `output/<dataset>/` hoặc thư mục mới.
- Nếu artifact lớn, giao phần đọc/rút thống kê cho agent mô tả ở
  `../agents/01_dataset_artifact_reader.md`.

## 2. Lập schema map trước khi convert

Ghi rõ cho mỗi trường:

- episode/frame boundary và index;
- camera/image/video reference;
- state/action shape, dtype và semantics;
- timestamp/control rate/synchronization;
- language, success, terminal flags, robot/task metadata;
- field thiếu, assumption và loss khi map sang canonical.

Không dùng shape để suy action semantics. Nếu source không cung cấp frame/unit/gripper
convention, đánh dấu unknown.

## 3. Smoke inspect

Chạy `--max-episodes 1` hoặc `2`, mặc định `--decode-images false`. Ghi command, exit code và
tóm tắt output. Kiểm tra NaN/Inf, timestamp monotonic, độ dài các tensor, first/last/terminal
flags và language coverage.

Với LeRobot, nhớ rằng `max_episodes` hiện được áp sau khi metadata Parquet đã được đọc/group;
đừng xem đây là bảo đảm memory-safe cho dataset lớn.

## 4. Convert sang đích riêng

Chọn HDF5 để debug episode hoặc Parquet để query frame metadata. Chỉ decode image khi câu hỏi
cần pixel. Writer hiện chưa atomic/resumable; không ghi đè output quan trọng khi chưa có cách
tái tạo.

## 5. Validate sau convert

Đối chiếu tối thiểu:

- episode count và step count từng episode;
- action/state shape, dtype, NaN/Inf;
- timestamp và flags;
- language/task/success/provenance;
- image reference hoặc decoded sample;
- loss/assumption đã biết.

“Training-ready” chỉ được kết luận khi có loader đích đọc được artifact và một batch sample
giữ đúng schema/semantics. Hiện repo chưa có training-loader test chuẩn, nên báo cáo mặc định
phải ghi đây là phần chưa verify.

## 6. Ghi báo cáo

Dùng [02_write_research_report.md](02_write_research_report.md). Tách rõ `Verified`,
`Inferred`, `Unknown` và `Planned`; kèm path/lệnh đủ để chạy lại.
