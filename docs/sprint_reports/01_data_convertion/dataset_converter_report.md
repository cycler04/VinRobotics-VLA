# Dataset converter: contract, evidence, and current limits

Ngày kiểm chứng: 2026-07-16.

Câu hỏi chính của báo cáo này là: converter hiện chuyển đổi dữ liệu nào, giữ hoặc biến đổi
thông tin gì, và output đã đủ an toàn để gọi là training-ready hay chưa? Phạm vi chỉ gồm
package `vla_data_tools`; downloader, labeler và pipeline training không thuộc báo cáo.

## Kết luận ngắn

**Verified:** converter đang chạy theo mô hình adapter:

```text
LeRobot v2/v3 ──> LeRobotReader ──┐
                                  ├─> CanonicalEpisode v0.1 ──> HDF5
RLDS/TFDS ──────> RLDSReader ─────┘                         └─> Parquet + .npy assets
```

Nó giữ được episode boundary, state/action dạng vector, terminal flags, task text ở mức
episode và provenance đủ để audit nhiều quyết định mapping. Tuy nhiên, một số dữ liệu được
chuẩn hóa hoặc tổng hợp: timestamp/state/action bị ép sang `float32`, `is_first`/`is_last`
được dựng lại, timestamp RLDS có thể được sinh từ control rate, và language/success có thể
bị rút gọn.

**Verified:** converter chưa hiểu đầy đủ action/state semantics. Unit, coordinate frame và
absolute-vs-delta phần lớn vẫn là `unknown` hoặc `source_defined`. Validation hiện chứng minh
tính nhất quán cấu trúc, không chứng minh semantic equivalence, khả năng round-trip hay khả
năng nạp trực tiếp vào training loader. Vì vậy output hiện là intermediate artifact có thể
inspect và preprocess tiếp, chưa phải bằng chứng rằng dataset đã training-ready.

Các lệnh vận hành được giữ riêng trong
[VLA_DATA_TOOLS_GUIDE.md](code_guilders/VLA_DATA_TOOLS_GUIDE.md) để báo cáo này không lặp
runbook.

## Contract và mapping thực tế

`CanonicalEpisode` là ranh giới duy nhất giữa reader và writer. Contract chứa timestamp,
state/action, ba boundary/terminal flags, task/robot metadata, decoded images hoặc image
references, action/state spec và source metadata
([canonical.py](../src/vla_data_tools/canonical.py#L9-L35)). CLI chỉ hỗ trợ hai input
`lerobot|rlds` và hai output `hdf5|parquet`
([__main__.py](../src/vla_data_tools/__main__.py#L34-L80)).

| Thành phần         | Mapping đã xác minh                                                                                                                                                                                     | Conversion loss hoặc giới hạn                                                                                                                                                                                                                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Episode boundary     | Row LeRobot được group theo`episode_index`; RLDS giữ mỗi phần tử `episode -> steps` thành một episode. Canonical luôn đặt đúng một `is_first` ở đầu và một `is_last` ở cuối. | Source`is_first`/`is_last` của RLDS chỉ còn trong `source_metadata.source_flags`; canonical flags không phải bản sao nguyên trạng.                                                                                                                                                  |
| Timestamp            | LeRobot lấy`timestamp` của frame. RLDS dùng timestamp numeric nếu tìm thấy, nếu không sinh `step_index / control_rate_hz`.                                                                     | Timestamp được ép`float32`. Với RLDS không khai báo rate, 10 Hz là assumption mặc định; có thể override bằng CLI.                                                                                                                                                                 |
| State                | LeRobot lấy`observation.state`. RLDS ưu tiên các field state phổ biến; DROID ghép joint và gripper; fallback ghép các vector numeric nhỏ.                                                     | Flatten và ép`float32`; fallback phụ thuộc field order. Unit, frame và ý nghĩa từng chiều không được chuẩn hóa.                                                                                                                                                                  |
| Action               | LeRobot lấy`action`; RLDS lấy vector `action`; cả hai được ép `float32`.                                                                                                                      | LeRobot để representation/frame/unit là`unknown`; RLDS thường là `source_defined`. Chưa có phép kiểm tra semantic equivalence.                                                                                                                                                      |
| Terminal và success | LeRobot map`next.done` sang `is_terminal`, success là `any(next.success)`. RLDS giữ `is_terminal`.                                                                                               | Success theo step của LeRobot bị collapse về một boolean episode; RLDS đặt success là`None`.                                                                                                                                                                                             |
| Language             | LeRobot resolve task theo`task_index`, fallback sang episode metadata. RLDS giữ instruction không rỗng đầu tiên trong episode.                                                                     | Không giữ chuỗi language thay đổi theo step như một modality riêng.                                                                                                                                                                                                                       |
| Image                | LeRobot giữ đường dẫn video và time range nếu metadata có. RLDS có thể decode thành tensor`[T,H,W,C]` hoặc giữ reference khi `--decode-images false`.                                     | LeRobot không kiểm file/reference decode được. RLDS reference hiện chỉ trỏ tới thư mục TFRecord, không chứa shard/record/step locator, nên chưa chứng minh phục hồi trực tiếp đúng frame. Camera decoded thiếu ở bất kỳ step nào sẽ không được stack vào output. |
| Provenance           | Source format/version, episode metadata, feature specs, rate source và source flags được serialize vào metadata.                                                                                      | Không có code revision, source checksum hay manifest output để chứng minh provenance end-to-end.                                                                                                                                                                                             |

Các mapping LeRobot nằm ở
[lerobot.py](../src/vla_data_tools/lerobot.py#L121-L191); heuristic, timestamp và mapping
RLDS nằm ở [rlds.py](../src/vla_data_tools/rlds.py#L52-L92) và
[rlds.py](../src/vla_data_tools/rlds.py#L150-L299).

## Validation và output layout

Validation kiểm các tensor có cùng số step, episode không rỗng, state/action có shape
`[T,D]`, timestamp tăng nghiêm ngặt và finite, boundary flags hợp lệ, decoded image có shape
`[T,H,W,C]`. Khác state/action shape hoặc camera set giữa các episode được inspector báo lỗi
([canonical.py](../src/vla_data_tools/canonical.py#L48-L100),
[inspect.py](../src/vla_data_tools/inspect.py#L12-L74)).

Điều validation **không** kiểm gồm unit/frame/action convention, ý nghĩa từng state dimension,
success semantics, image-to-action synchronization và khả năng load output. `action_spec` có
giá trị `unknown` hoặc `source_defined` vẫn là dictionary không rỗng, nên không tạo warning.
Vì vậy `validation_errors: []` chỉ có nghĩa là contract cấu trúc đã qua kiểm tra.

HDF5 ghi tuần tự theo episode với layout chính:

```text
/episodes/<episode_id>/steps/
  timestamp
  observation/state
  observation/images/<camera>  # chỉ khi đã decode
  action/raw
  is_first | is_last | is_terminal
```

Metadata episode nằm trong HDF5 attributes dạng JSON; numeric datasets dùng gzip. Parquet
ghi một row mỗi frame với `episode_id`, `frame_index`, timestamp, state/action, flags và
`episode_metadata_json`. Metadata JSON bị lặp trên từng frame. Decoded images của Parquet
được tách sang `<stem>_assets/episode_<id>/<camera>.npy`, còn Parquet chỉ giữ reference
([writers.py](../src/vla_data_tools/writers.py#L19-L100)).

## Bằng chứng runtime

Ba lệnh `inspect --max-episodes 2 --decode-images false` được chạy trên sample local ngày
2026-07-16 và đều trả status 0:

| Source             | Steps min–max | Rate median | State / action | Camera | Language coverage |
| ------------------ | -------------: | ----------: | -------------- | -----: | ----------------: |
| LeRobot PushT      |       118–161 |       10 Hz | 2D / 2D        |      1 |              100% |
| DROID partial RLDS |       240–512 |       15 Hz | 8D / 7D        |      3 |                0% |
| OXE UTokyo PR2     |       130–132 |       10 Hz | 7D / 8D        |      1 |              100% |

Các artifact HDF5 và Parquet hiện có trên disk cho cùng ba source có episode/step counts và
boundary counts nhất quán:

| Artifact pair                                                              | Episodes | Frame/step rows | Timestamp provenance                  |
| -------------------------------------------------------------------------- | -------: | --------------: | ------------------------------------- |
| `output/lerobot_pusht/*`                                                 |      200 |          24,892 | Timestamp source LeRobot              |
| `output/droid_200/*`                                                     |      200 |          59,930 | Synthesized, profile 15 Hz            |
| `output/utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds/*` |      200 |          27,416 | Synthesized, default assumption 10 Hz |

Ở mỗi HDF5 và Parquet trên, số `is_first=true` và `is_last=true` đều bằng số episode.
`is_terminal` không đồng nghĩa với `is_last`: PushT HDF5 có 400 terminal flags cho 200
episode, và năm episode source được đối chiếu đều có `next.done=true` ở hai frame cuối.
Snapshot này xác minh layout và consistency của artifact đang có, nhưng artifact không lưu
code revision; do đó nó không tự chứng minh mọi file được sinh từ chính source tree hiện tại.

`.venv/bin/pytest -q` chạy ngày 2026-07-16 cho kết quả `4 passed`. Test hiện chứng
minh LeRobot v2/v3 synthetic fixtures, episode limit, validation/inspection cơ bản, HDF5 và
Parquet layout, decoded `.npy` asset và default output path
([test_vla_data_tools.py](../tests/test_vla_data_tools.py)). **Unknown/chưa được test:**
`RLDSReader`, partial-shard path, CLI end-to-end, failure recovery, large-data behavior và
round-trip.

## Giới hạn vận hành

- **Verified:** `LeRobotReader` đọc toàn bộ data Parquet thành Python rows và group toàn
  dataset trước khi áp `max_episodes`; giới hạn này không chặn I/O/RAM ban đầu.
- **Verified:** `RLDSReader` materialize toàn bộ steps của từng episode; inspector materialize
  toàn bộ episode được chọn; Parquet writer tích toàn bộ frame rows trong RAM. Chưa có
  benchmark để đặt ngưỡng dataset an toàn.
- **Verified:** writer ghi trực tiếp vào destination. HDF5 mở mode `w`; Parquet có thể tạo
  `.npy` assets trước khi file Parquet hoàn tất. Không có atomic rename, resume, checksum,
  manifest hoặc quarantine cho episode lỗi.
- **Verified:** chưa có output reader hoặc test canonical -> output -> canonical. Chưa có
  writer LeRobot hay training loader.
- **Inferred:** HDF5 phù hợp smoke conversion lớn hơn Parquet vì writer ghi từng episode,
  nhưng reader vẫn có thể là nút thắt RAM; cần đo peak RSS trước khi chạy full dataset.
- **Planned:** validated shards, manifest, training loader và round-trip check là hướng tiếp
  theo trong
  [01_dataset_to_training_ready.md](../.agents/plans/01_dataset_to_training_ready.md), không
  phải runtime capability hiện tại.

## Cách tái lập

Chạy từ workspace root:

```bash
.venv/bin/pytest -q

.venv/bin/python -m vla_data_tools inspect \
  --format lerobot --path dataset/lerobot_pusht --max-episodes 2

.venv/bin/python -m vla_data_tools inspect \
  --format rlds --path dataset/droid_200 \
  --max-episodes 2 --decode-images false

.venv/bin/python -m vla_data_tools inspect \
  --format rlds \
  --path dataset/oxe/utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds \
  --max-episodes 2 --decode-images false
```

Trước conversion lớn, đo `du -sh dataset output`, dùng sample nhỏ, và coi output là derived
artifact có thể tái tạo thay vì ghi đè artifact có giá trị.
