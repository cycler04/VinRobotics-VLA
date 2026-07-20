# Tổng quan workspace

## Mục tiêu hiện tại

Đây là workspace R&D về dữ liệu cho Vision-Language-Action (VLA). Trọng tâm là hiểu định
dạng dataset, cách xử lý và ingest nhiều nguồn, rồi chứng minh dữ liệu đầu ra đủ rõ schema,
semantics và chất lượng để dùng cho training.

Không mô tả workspace này như một hệ thống train hoặc inference robot hoàn chỉnh. Thành
phần chạy được hiện tại là package Python `vla-data-tools` dùng để đọc, inspect, validate và
convert sample dataset.

## Trạng thái thật

Đang chạy được:

- đọc LeRobot v2/v3 metadata và RLDS/TFDS sample;
- chuyển source episode về `CanonicalEpisode`;
- inspect số episode/step, shape, timestamp, language coverage và validation error;
- ghi HDF5 hoặc Parquet vào `output/<dataset>/`;
- tải các sample LeRobot, DROID và OXE bằng script;
- 4 unit test hiện có đều pass.

Mới là kế hoạch hoặc chưa đủ bằng chứng:

- writer `canonical -> LeRobot` và loader training bằng PyTorch;
- round-trip RLDS/LeRobot giữ nguyên semantics;
- ingestion streaming/chunked đủ scale cho dataset lớn;
- resume/atomic output, manifest đầy đủ, benchmark throughput;
- pipeline train, evaluation, inference và robot execution.

## Khoảng cách giữa tên gọi và thực tế

- `VinRobotics` và các tài liệu VLA bao quát cả pipeline robot, nhưng code chỉ tập trung vào
  Sprint 1 data ingestion/conversion.
- Cụm “training-ready” là mục tiêu cần chứng minh bằng loader, schema và validation; một file
  HDF5/Parquet được ghi thành công chưa tự động là training-ready.
- `notes/sprint1_task.md` là roadmap cũ: checklist chưa phản ánh phần đã implement, còn một
  số ví dụ dùng `python -m tools.inspect` và `--repo-id` không tồn tại trong CLI hiện tại.
- `src/local_video_server.py` và `docs/code_guilders/local_labeler_guild.md` thuộc một luồng
  Caption QA khác. Repo không có editor hoặc `download_videos.py`; không xem chúng là phần
  của VLA data pipeline.

## Bẫy và nợ kỹ thuật

- `.git/` rỗng nên `git status`, `git log` và `git check-ignore` đều không dùng được.
- `notes/wifi.txt` chứa access detail cục bộ; file đã được ignore và không được trích nội dung
  vào báo cáo, log hay prompt.
- `requirements.txt` cài TensorFlow CUDA và pytest vô điều kiện, trong khi `pyproject.toml`
  tách chúng thành extras; cài theo requirements có thể tạo environment rất lớn.
- `LeRobotReader` gom metadata Parquet trước khi áp `max_episodes`; Parquet writer gom rows
  trước khi ghi. Cần ước lượng RAM/disk trước full run.
- Writer ghi trực tiếp vào đích, chưa atomic/resumable. Không ghi đè artifact có giá trị nếu
  chưa giữ source hoặc có cách tái tạo.
- Test chưa bao phủ RLDS fixture thật, downloader, CLI end-to-end, video server hoặc writer
  LeRobot.

Kế hoạch nghiên cứu tiếp theo nằm ở [plans/01_dataset_to_training_ready.md](plans/01_dataset_to_training_ready.md).
