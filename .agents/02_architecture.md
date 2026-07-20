# Kiến trúc và luồng dữ liệu

## Luồng đang chạy

```text
dataset/LeRobot ──> LeRobotReader ──┐
                                    ├─> CanonicalEpisode ──> inspect/validate
dataset/RLDS ─────> RLDSReader ─────┘                    ├─> HDF5
                                                         └─> Parquet + assets
```

Các điểm vào chính:

| Thành phần | Nguồn sự thật | Vai trò |
|---|---|---|
| CLI | `src/vla_data_tools/__main__.py` | Parse `inspect` và `convert`, chọn reader/writer |
| Canonical contract | `src/vla_data_tools/canonical.py` | Episode trung gian và validation cơ bản |
| LeRobot reader | `src/vla_data_tools/lerobot.py` | Đọc metadata/data Parquet v2/v3, giữ video reference |
| RLDS reader | `src/vla_data_tools/rlds.py` | Đọc TFDS/TFRecord episode, tùy chọn decode image |
| Inspector | `src/vla_data_tools/inspect.py` | Tổng hợp shape, step, language, timestamp và lỗi |
| Writers | `src/vla_data_tools/writers.py` | Ghi HDF5 hoặc Parquet; image decode tách ra `.npy` |
| Downloader | `scripts/download_vla_samples.py` | Tải sample LeRobot/DROID/OXE và manifest |

## Canonical contract

`CanonicalEpisode` là ranh giới giữa reader và writer. Khi thêm format, ưu tiên viết một
adapter vào/ra contract này thay vì converter trực tiếp giữa từng cặp format.

Các semantics không được suy ra chỉ từ shape:

- action absolute hay delta, joint hay end-effector;
- coordinate frame, unit, rotation representation và gripper convention;
- state fields và thứ tự chiều;
- timestamp/control rate và cách đồng bộ camera;
- task language, success, terminal flags và provenance.

Nếu source không cung cấp đủ, ghi `unknown` hoặc assumption kèm bằng chứng; không điền một
giá trị có vẻ hợp lý rồi trình bày như fact.

## Layout dữ liệu và output

```text
dataset/                     # source local, immutable, gitignored
  <dataset>/
output/                      # derived artifact, gitignored
  <dataset>/
    <dataset>.hdf5
    <dataset>.parquet
    <dataset>_assets/        # .npy khi có decoded image
docs/                        # báo cáo đã tổng hợp, có thể đọc lại
notes/                       # nguồn, task và ghi chú thô
```

Không sửa source dataset tại chỗ. Output mới phải vào `output/` hoặc một đích riêng do người
dùng chỉ định. Báo cáo không chứa blob/bảng dữ liệu lớn; chỉ chứa thống kê, sample nhỏ và
đường dẫn bằng chứng.

## Thiết kế chưa được implement

Luồng mong muốn tiếp theo là:

```text
source -> canonical -> validated shards + manifest -> training loader -> round-trip check
```

Đây là hướng trong [plans/01_dataset_to_training_ready.md](plans/01_dataset_to_training_ready.md),
không phải mô tả năng lực runtime hiện tại.
