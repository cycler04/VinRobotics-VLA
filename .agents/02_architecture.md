# Kiến trúc

## Hai mảng của workspace

### Nghiên cứu

- `docs/papers/`: paper và danh sách liên kết nguồn.
- `docs/research_finding/`: phân tích, key finding và báo cáo theo chủ đề.
- `notes/`: ghi chú vận hành, task và roadmap cũ; không mặc định là spec hiện
  hành.
- `third_party/`: code vendor/submodule phục vụ tái lập paper.

Luồng nghiên cứu chuẩn được tách thành ba workflow:

```text
paper/source -> research note -> experiment evidence -> synthesis/report
```

Chi tiết nằm trong [workflows/](workflows/).

### VLA data tools

```text
LeRobot v2/v3 ─┐
               ├─> CanonicalEpisode v0.1 ─> inspect/validate
RLDS/TFDS ─────┘                           ├─> HDF5
                                           └─> Parquet + .npy image assets
```

- CLI: `src/vla_data_tools/__main__.py`
- Schema và validator: `canonical.py`
- Readers: `lerobot.py`, `rlds.py`
- Inspector: `inspect.py`
- Writers: `writers.py`
- Tests: `tests/test_vla_data_tools.py`

Entry point đóng gói là `vla-data-tools`; entry point không cần install là
`PYTHONPATH=src python -m vla_data_tools`.

## Contract canonical hiện có

Mỗi episode giữ ID, dataset, timestamp, state/action raw, cờ
first/last/terminal, robot/task, image hoặc reference, action/state spec và
source metadata. Validator kiểm tra length, dtype, giá trị hữu hạn, timestamp và
shape cơ bản.

Action/state semantics có thể vẫn là `unknown`. Hai vector cùng shape không có
nghĩa là cùng semantics; không chuẩn hóa hay ghép dataset khi chưa xác minh
frame, unit và control convention.

## Layout dữ liệu

- Input local: `dataset/` — không track.
- Output conversion: `output/<dataset>/<dataset>.hdf5|parquet` — không track.
- Ảnh decode cho Parquet:
  `output/<dataset>/<stem>_assets/episode_<id>/<camera>.npy`.
- Artifact/log/checkpoint lớn phải ở các thư mục đã ignore, không đặt lẫn với
  tài liệu nguồn.

## Ranh giới

- `src/local_video_server.py` là tiện ích Caption QA độc lập.
- Submodule Kinetix tự quản dependencies và lệnh bằng `uv`; không áp conventions
  của package gốc vào code vendor nếu tác vụ không yêu cầu sửa vendor.
- Chưa có đường đọc ngược HDF5/Parquet, writer LeRobot hoặc round-trip; mọi sơ đồ
  có các nhánh đó chỉ là thiết kế.

