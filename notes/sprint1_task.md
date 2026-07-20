

# VLA Data Ingestion & Format Conversion Roadmap

## 1. Mục tiêu

Xây được data-ingestion pipeline và bộ format-conversion utilities cho VLA pretraining. Kết quả không chỉ là tải được dữ liệu, mà là có thể đọc, kiểm tra, chuẩn hoá và dùng lại trajectory từ nhiều ecosystem cho training.

### Definition of done

- [ ] Hands-on download và parse tối thiểu ba nguồn dữ liệu đại diện.
- [ ] Hiểu và ghi lại schema của Open X-Embodiment/OXE, DROID và LeRobot.
- [ ] Có canonical internal schema cho robot episode.
- [ ] Convert được `RLDS -> internal -> LeRobot v2`.
- [ ] Convert được `LeRobot -> internal -> HDF5/Parquet`.
- [ ] Có validator và round-trip tests cho episode boundary, timestamps, state/action và metadata.
- [ ] Có CLI, manifest, logging và khả năng giới hạn số episode để chạy sample trước.

## 2. Kiến thức format cần nắm

### RLDS

RLDS (Reinforcement Learning Datasets) là schema dữ liệu theo `episode -> steps`, thường được phát hành thông qua TensorFlow Datasets (TFDS) và file TFRecord. Nó **không phải Apache Arrow**.

Một step điển hình:

```python
{
    "observation": {
        "image": ...,
        "state": ...,
    },
    "action": ...,
    "language_instruction": ...,
    "is_first": False,
    "is_last": False,
    "is_terminal": False,
}
```

RLDS phù hợp cho OXE/RT-X, DROID và nhiều training setup VLA. Tuy nhiên các dataset vẫn có thể khác nhau về camera, state, action, coordinate frame và control frequency.

### LeRobot v2

LeRobot là ecosystem PyTorch-native trên Hugging Face. LeRobot v2 thường tổ chức dữ liệu frame-level trong Parquet, video hoặc image assets cho camera, và metadata theo episode/task.

LeRobot v3 đã đổi sang file-based: nhiều episode có thể nằm trong một Parquet/MP4, episode boundaries được tra bằng metadata. Giai đoạn đầu nên target **v2**, nhưng internal format cần có episode index để có thể nâng lên v3 sau này.

### HDF5 và Parquet internal

- **HDF5:** tốt cho debug, xem một episode, tương thích nhiều codebase imitation learning.
- **Parquet + MP4/images:** tốt hơn khi scale lớn và query metadata. Không nên lưu RGB decode nguyên khối trong Parquet; giữ video/image assets riêng, Parquet giữ frame index hoặc reference.

## 3. Dataset thực hành

Không tải full OXE/DROID ngay. Hãy chỉ tải vài episode/shard để xây parser và validator; sau đó mới scale.

| Dataset                    | Format chính                            | Mục đích thực hành                                                           | Ưu tiên |
| -------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------- | --------- |
| OXE subset, ví dụ Bridge | RLDS/TFDS/TFRecord                       | Hiểu multi-embodiment, nested episode/step schema, language/action heterogeneity | 1         |
| DROID sample               | RLDS                                     | Multi-camera, robot state/action thực tế, calibration và metadata              | 2         |
| Một LeRobot dataset nhỏ  | LeRobot v2                               | Parquet/video layout và PyTorch-ready loading                                    | 3         |
| AgiBot World task subset   | Dataset-native, có tool chuyển LeRobot | Bài tập mở rộng large-scale ingestion                                         | Sau cùng |

Ghi chú: DROID full rất lớn, khoảng 1.7 TB; đừng biến việc download full thành blocker.

## 4. Kiến trúc cần xây

Không nên viết converter trực tiếp cho từng cặp format. Sử dụng một canonical schema làm trung gian:

```text
RLDS ───────┐
LeRobot ────┼──> Canonical Episode ──> HDF5 / Parquet / LeRobot / training loader
Raw dataset ┘
```

Điều này giúp thêm format mới chỉ cần một adapter đọc hoặc ghi thay vì viết mọi cặp converter.

## 5. Canonical internal schema

```python
episode = {
    "episode_id": "...",
    "dataset_name": "...",
    "robot": {
        "name": "...",
        "embodiment": "...",
        "control_frequency_hz": 10,
    },
    "task": {
        "language_instruction": "...",
        "success": True,
    },
    "steps": {
        "timestamp": float32,             # [T]
        "observation.images.front": ...,  # image array hoặc video reference
        "observation.images.wrist": ...,
        "observation.state": float32,     # [T, Ds]
        "action.raw": float32,            # [T, Da]
        "action.normalized": float32,     # optional
        "is_first": bool,
        "is_last": bool,
        "is_terminal": bool,
    },
    "action_spec": {
        "representation": "delta_ee_pose_gripper",
        "frame": "robot_base",
        "rotation": "axis_angle",
        "units": {"position": "m", "rotation": "rad"},
        "gripper_semantics": "open_close",
    },
}
```

### Nguyên tắc quan trọng

- Giữ `action.raw` nguyên bản.
- Lưu `action_spec` và `state_spec` rõ ràng: action/state nghĩa là gì, absolute hay delta, coordinate frame, unit và gripper convention.
- Không giả định hai vector có cùng số chiều là tương đương về nghĩa.
- Giữ timestamps gốc, camera calibration và metadata nguồn khi dataset có cung cấp.
- Các modality nâng cao như depth, stereo, force-torque và asynchronous stream là extension fields, không làm block phiên bản đầu.

## 6. Cấu trúc repo đề xuất

```text
vla-data-tools/
  adapters/
    rlds_reader.py
    lerobot_reader.py
    hdf5_reader.py
  schemas/
    canonical_episode.py
    action_spec.py
  converters/
    rlds_to_internal.py
    lerobot_to_internal.py
    internal_to_lerobot.py
    internal_to_hdf5.py
  validators/
    validate_episode.py
    compare_roundtrip.py
  tools/
    inspect.py
  notebooks/
    01_inspect_oxe.ipynb
    02_inspect_droid.ipynb
    03_inspect_lerobot.ipynb
```

## 7. Thứ tự implement

1. Xây canonical schema và validator cơ bản.
2. Viết dataset inspector.
3. Viết `LeRobot -> canonical` vì dễ debug bằng PyTorch.
4. Viết `RLDS -> canonical` để học TFDS/TFRecord và nested features.
5. Viết `canonical -> HDF5` để tạo output dễ inspect.
6. Viết `canonical -> LeRobot v2` để tạo output training-ready.
7. Chỉ sau đó mới viết `canonical -> RLDS`, vì cần TFDS builder/writer tương thích.

Mọi converter ban đầu cần hỗ trợ sample mode:

```bash
convert --max-episodes 10 --decode-images false
convert --max-episodes 2 --decode-images true
```

## 8. Dataset inspector bắt buộc

Mỗi format phải có lệnh inspect, ví dụ:

```bash
python -m tools.inspect --format rlds --path data/droid_sample
python -m tools.inspect --format lerobot --repo-id lerobot/example_dataset
```

Output tối thiểu:

```text
episodes: 100
steps: min=80, median=214, max=502
control rate: 10 Hz
cameras: exterior_left, exterior_right, wrist
image shape: 480 x 640 x 3
state: [T, 7], float32
action: [T, 7], float32
instruction coverage: 94%
action convention: documented / unknown / verified
```

Ngoài việc hiểu schema, inspector cần phát hiện NaN/Inf, shape không nhất quán, missing modality và timestamp không tăng dần.

## 9. Validation và round-trip tests

Sau mỗi conversion, kiểm tra:

- [ ] Số episode không đổi.
- [ ] Số step từng episode không đổi.
- [ ] `is_first`, `is_last`, `is_terminal` ở đúng vị trí.
- [ ] Timestamps tăng dần và sai số trong tolerance.
- [ ] State/action shape, dtype, NaN/Inf.
- [ ] Language instruction, success và provenance metadata không mất.
- [ ] Một số frame RGB sample giống sau decode.
- [ ] `action_spec` và `state_spec` còn nguyên.

Round-trip tối thiểu:

```text
RLDS sample
  -> canonical
  -> LeRobot v2
  -> canonical
  -> compare
```

Action, state và metadata phải match trong tolerance. RGB có thể không byte-identical nếu encode lại bằng MP4 lossy compression.

## 10. Lộ trình 4 tuần

### Tuần 1 — Ecosystem và inspection

- [ ] Đọc docs/paper OXE, DROID, LeRobot.
- [ ] Download sample nhỏ của ba dataset.
- [ ] Hoàn thành inspector cho từng format.
- [ ] Viết schema-comparison report.
- [ ] Chốt canonical schema v0.1, `action_spec`, `state_spec`.

### Tuần 2 — RLDS

- [ ] Parse OXE subset và DROID sample.
- [ ] Implement `RLDS -> canonical -> HDF5`.
- [ ] Viết validator, test 5–10 episodes.
- [ ] Nắm `tf.data`, TFRecord, nested feature và TFDS builder.

### Tuần 3 — LeRobot

- [ ] Parse một LeRobot dataset nhỏ.
- [ ] Implement `LeRobot -> canonical`.
- [ ] Implement `canonical -> LeRobot v2`.
- [ ] Hỗ trợ image/video references, metadata và episode index.
- [ ] Load output bằng PyTorch `DataLoader`.

### Tuần 4 — Productionize

- [ ] Round-trip tests cho RLDS và LeRobot.
- [ ] CLI, manifest, chunking, resume và logging.
- [ ] Benchmark đọc/ghi và dung lượng: HDF5 vs Parquet + MP4.
- [ ] Viết guide: thêm một dataset adapter mới.
- [ ] Nếu còn thời gian, ingest một AgiBot task subset.

## 11. Survey checklist

Với mỗi dataset, trả lời các câu hỏi sau thay vì chỉ ghi tên/size:

| Câu hỏi              | Nội dung phải ghi                                         |
| ---------------------- | ----------------------------------------------------------- |
| Đơn vị dữ liệu?   | frame, step, episode, trajectory hay segment                |
| Modalities?            | RGB, depth, proprioception, language, force, calibration    |
| Action semantics?      | joint/Cartesian, absolute/delta, frame, units, gripper      |
| Đồng bộ thời gian? | cùng tần số, nearest-frame hay asynchronous streams      |
| Training loader?       | `tf.data`, PyTorch Dataset, video streaming               |
| License và splits?    | research/commercial, train/validation/test/task split       |
| Loss khi convert?      | depth, calibration, raw video, action semantics, timestamps |

## 12. Tiêu chí hoàn thành cuối cùng

Bạn hoàn thành mục tiêu khi chứng minh được câu này bằng code và test:

> Có thể lấy vài episode RLDS từ OXE/DROID, giữ nguyên episode semantics, action/state/language/camera metadata, chuyển sang internal format và LeRobot v2, rồi load output bằng PyTorch cho training.

Phiên bản v1 chỉ cần có contract chung: **RGB + language + state + action + episode metadata**. Đừng cố làm universal converter cho mọi sensor/action representation ngay từ đầu.

## 13. Nguồn chính thức để bắt đầu

- [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/)
- [OpenVLA training repository](https://github.com/openvla/openvla)
- [DROID dataset documentation](https://droid-dataset.github.io/droid/the-droid-dataset)
- [LeRobot Dataset v3 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [LeRobot migration: v2.1 vs v3.0](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)
- [AgiBot World repository](https://github.com/OpenDriveLab/AgiBot-World)
