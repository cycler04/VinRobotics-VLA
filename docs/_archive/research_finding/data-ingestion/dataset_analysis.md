# Phân tích EgoDex, EgoVerse và XP10M

## Câu hỏi và phạm vi

Báo cáo này chỉ phân tích ba source trong
`dataset/corpus_sample_bundle`: quy mô, format, modality, đặc trưng riêng,
license, chất lượng quan sát được và sample thật. Thiết kế adapter, canonical
schema, training view và runtime nằm ở
[báo cáo ingestion/runtime](ingestion_runtime.md).

Nguồn được đọc ngày 2026-07-27:

- [README của bundle](../../../dataset/corpus_sample_bundle/README.md);
- [schema đi kèm bundle](../../../dataset/corpus_sample_bundle/corpus/SCHEMA.md);
- manifest, canonical Parquet, raw-source peeks và sáu video trong bundle;
- [companion data-flow](corpus-data-flow.html).

Các con số production không được đo lại vì production storage không nằm trong
workspace. Báo cáo tách rõ số được source công bố và số đo trực tiếp từ sample.

## Kết luận ngắn

Ba dataset cùng là egocentric human manipulation nhưng khác nhau đáng kể:

| Source | Capture | Raw format | Hand signal | Điểm khác biệt chính |
| --- | --- | --- | --- | --- |
| EgoDex | Apple Vision Pro | LeRobot v3 + packed AV1 | ARKit 24-joint SE(3) | finger joint có cả rotation |
| EgoVerse | Aria-style internal rig | Zarr + HEVC | mocap 21×3 | có hai hand-coordinate family |
| XP10M | HOMIE head rig | HDF5 + H.264 | predicted MANO/21×3 | IMU/full body, validity thấp, non-commercial |

EgoDex giàu nhất về finger transform; EgoVerse gần representation `kp21` nhất;
XP10M giàu sensor/context nhất nhưng hand là prediction và license hạn chế.
Không dataset nào chứa robot command/action trực tiếp trong Layer-1 sample:
đây là human state/geometry, video và language.

## 1. Quy mô

### Production được công bố

Bundled README công bố 380.401 clip, xấp xỉ 3.000 giờ:

| Source | Clip | Native FPS |
| --- | ---: | ---: |
| EgoDex | 318.055 | 30 |
| EgoVerse | 49.650 | 30 |
| XP10M | 12.696 | 20 |
| **Tổng** | **380.401** | — |

Companion report ghi đã đo ba production manifest:

| Source | Clip | Giờ | Frame | Release bytes |
| --- | ---: | ---: | ---: | ---: |
| EgoDex v06 | 318.055 | 732,42 | 79.101.635 | 245 GB |
| EgoVerse v06 | 49.650 | 987,03 | 106.598.737 | 76 GB |
| XP10M v06 | 12.696 | 980,03 | 70.562.112 | 47 GB |
| **Tổng** | **380.401** | **2.699,48** | **256.262.484** | **368 GB** |

Hai nguồn mô tả không hoàn toàn khớp: `≈3.000 h` so với `2.699,48 h`.
Production manifests/gate reports không nằm trong bundle nên cần đo lại release
thật trước khi lập ngân sách storage hoặc compute.

Clip count cũng không đại diện cho thời gian: EgoDex chiếm phần lớn số clip,
nhưng EgoVerse và XP10M mỗi source có nhiều giờ hơn.

### Sample bundle đo được

| Source | Clip | Canonical frame | Duration | Narrative row |
| --- | ---: | ---: | ---: | ---: |
| EgoDex | 2 | 480 | 16,0 s | 10 |
| EgoVerse | 2 | 480 | 16,0 s | 8 |
| XP10M | 2 | 492 | 24,6 s | 12 |
| **Tổng** | **6** | **1.452** | **56,6 s** | **30** |

Bundle chỉ bằng khoảng 0,00158% production clip count và 0,000524% claim
3.000 giờ. Nó là integration fixture, không phải statistical sample.

Toàn bundle gồm 31 file, 120.752.152 byte:

- canonical corpus: 17 file, 2.514.027 byte;
- raw-source peeks: 6 file, 105.916.877 byte;
- video: 6 file, 12.308.854 byte;
- shared docs/loader: 5 file, 42.922 byte.

Sáu sample video đều là H.264. Mỗi video có đúng một frame container nhiều hơn
annotation span:

| Source | Resolution | Annotation/video frame |
| --- | --- | --- |
| EgoDex | 1920×1080 | 240/241 cho mỗi clip |
| EgoVerse | 640×480 | 240/241 cho mỗi clip |
| XP10M | 512×512 | 258/259 và 234/235 |

## 2. EgoDex

### Format và modality

EgoDex là Apple Vision Pro recording cho tabletop manipulation. Raw export theo
LeRobot v3:

- video production là AV1, packed nhiều episode mỗi file;
- episode metadata giữ data/video file index, timestamp span và global dataset
  row range;
- frame data riêng chứa camera/head, wrist và 24 ARKit finger joint mỗi tay;
- canonical sample dùng 30 FPS, 1920×1080 và metric scale 1,0.

Canonical `hand_format` là `arkit_joints_se3_wrist_relative`: mỗi joint lưu
transform `3×4`, tức 24 × 12 = 288 float mỗi tay. Đây là source duy nhất trong
ba source giữ rotation cho từng finger joint.

Hai sample clip có cả hai tay valid trên toàn bộ 480 frame. Narrative sidecar
có 8 row từ `qwen36_narrative_v1` và 2 row `joystick_v1`.

### Raw peek

`raw_sources/egodex/episodes_meta_add_remove_lid.parquet`:

- 104.865.770 byte;
- 2.565 episode và 687.221 frame;
- episode dài 15–840 frame;
- chỉ là task subset `add_remove_lid`, không phải toàn EgoDex.

Episode 179:

```json
{
  "episode_index": 179,
  "length": 240,
  "data_file_index": 2,
  "dataset_range": [47324, 47564],
  "video_file_index": 2,
  "video_time_s": [420.6333333333334, 428.6333333333334],
  "task": "Remove the lid from the cup placed on a wooden table..."
}
```

### Sample canonical thật

```yaml
clip_id: egodex/train.add_remove_lid.ep000179/shot-0
fps: 30
resolution: 1920x1080
n_frames: 240
duration_s: 8.0
hand_format: arkit_joints_se3_wrist_relative
valid_left_right: 240 / 240
split_group: add_remove_lid
world_scale: 1.0
streams: []
```

Frame 120, timestamp 4,0 s:

```yaml
left_wrist_world_m: [-0.12394, 0.24193, 0.32832]
right_wrist_world_m: [0.13490, 0.16832, 0.29551]
valid_left_right: 1 / 1
finger_vector_length_each_hand: 288
overlapping_language:
  left: hold cup
  right: remove lid from third cup and place on table
```

### Đặc trưng và loss khi tạo model view

- Native ARKit cung cấp metric head/hand tracking, không cần reconstruct từ RGB.
- Representation giàu hơn `kp21` vì có per-joint rotation.
- Nếu chuyển sang 21×3, rotation của finger joint và joint ngoài taxonomy 21
  keypoint bị bỏ.
- Production AV1 cần FFmpeg/PyAV; OpenCV decode không phải lựa chọn an toàn.

## 3. EgoVerse

### Format và modality

EgoVerse là internal Aria-style recording. Mỗi raw episode là Zarr store:

- `obs_head_pose`: float64 `[7]`;
- left/right wrist hoặc EEF pose: float64 `[7]`;
- left/right keypoint: float64 `[63]`;
- narration JSON;
- camera-front HEVC video.

Raw note ghi nhận hai storage family: keypoint đã ở world frame và keypoint
head-fixed với axis permutation theo episode. Shape giống nhau nhưng semantics
không giống nhau.

Canonical format là `egoverse_kp21x3_wrist_relative_cam`: 21×3 offset, tính
theo mét, wrist-relative và cùng trục với camera.

### Raw peek

```yaml
episode_id: 692eab3de2322e3b092b5f5c
frames: 189
fps: 30
duration_s: 6.274
embodiment: human_bimanual
video: HEVC, 640x360x3
head_pose: float64[7]
left/right_wrist_pose: float64[7]
left/right_keypoints: float64[63]
annotations: JSON annotation_v1
```

Hai video trong bundle là H.264 640×480. Raw peek và sample vì vậy đại diện hai
media family khác nhau; không có một resolution/intrinsics duy nhất cho toàn
EgoVerse.

### Sample canonical thật

```yaml
clip_id: egoverse/2026-05-02-16-17-47-195828/shot-0
fps: 30
resolution: 640x480
n_frames: 240
task: freeform_return_small_objects_to_bins
hand_format: egoverse_kp21x3_wrist_relative_cam
valid_left_right: 240 / 240
gravity_method: aria_slam
narrative_models:
  egoverse_annotation_v1: 1
  qwen3vl_hands_v1: 1
  joystick_v1: 1
```

Frame 120, timestamp 4,0 s:

```yaml
left_wrist_world_m: [-0.10068, 0.15108, 0.31249]
right_wrist_world_m: [0.30524, 0.10280, 0.36676]
valid_left_right: 1 / 1
finger_vector_length_each_hand: 63
native_narration: Put the black hair tie into the right transparent container
```

### Đặc trưng và giới hạn

- Đây là source gần model-side `kp21` nhất.
- Hai sample clip có cả hai tay valid toàn bộ 480 frame.
- Có native narration và hand-decomposed narration.
- Raw attrs không đủ để kiểm độc lập branch detection hoặc transform của hai
  hand-coordinate family.
- Camera family khác nhau khiến hard-code intrinsics hoặc image shape dễ tạo
  projection error dù tensor vẫn đúng shape.

## 4. XP10M

### Format, modality và license

XP10M là Ropedia Xperience-10M, manual labor ngoài tự nhiên. Raw HDF5 chứa:

- SLAM body pose và point cloud;
- stereo camera calibration;
- predicted MANO/21-joint hand;
- full-body mocap;
- IMU;
- hierarchical caption;
- video/media metadata.

Manifest ghi gated, approved **NON-COMMERCIAL**; sample ghi CC-BY-NC-4.0.
License phải đi cùng mọi derived artifact và model lineage.

### Raw HDF5 peek

| Field | Shape | Dtype/ý nghĩa |
| --- | --- | --- |
| `slam/trans_xyz` | `(234,3)` | float64 body translation |
| `slam/quat_wxyz` | `(234,4)` | float64 body rotation |
| `hand_mocap/*_joints_3d` | `(234,21,3)` | float32 |
| `hand_mocap/*_translation` | `(234,3)` | float32 wrist |
| `full_body_mocap/keypoints` | `(234,52,3)` | float32 |
| `imu/accel_xyz`, `gyro_xyz` | `(2325,3)` | float64, khoảng 200 Hz |
| `video/frame_number` | `(234,)` | int64 |

Frame raw đầu:

```json
{
  "device_timestamp": "86771504779500",
  "slam_trans_xyz": [-0.102295, -0.08294, 0.201295],
  "slam_quat_wxyz": [-0.141947, -0.199378, 0.806984, -0.537473],
  "left_wrist": [-0.000941, 0.182361, 0.312733],
  "right_wrist": [0.092294, 0.058142, 0.334940]
}
```

Caption raw có `Main Task`, `Sub Task`, `Current Action`, sampled frame,
objects và interaction. Sample có main task “Organizing small cardboard pieces
on a table” nhưng action đầu lại là “Pack phone into bag”, cho thấy raw
annotation có thể tự mâu thuẫn.

### Sample canonical thật

```yaml
clip_id: xp10m/a9dfa4f3-853f-4a8f-9041-0dfd72fdeb21_ep22/shot-0
fps: 20
resolution: 512x512
n_frames: 234
duration_s: 11.7
hand_format: xp10m_kp21x3_wrist_relative_cam
valid_left_right: 195 / 202
gravity_method: xp10m_slam
streams: [head_body_pose]
license: NON-COMMERCIAL
```

Clip ep9, frame 129, timestamp 6,45 s:

```yaml
left_wrist_world_m: [0.05714, -0.13763, 0.24055]
right_wrist_world_m: [0.13190, -0.11467, 0.41322]
valid_left_right: 1 / 1
finger_vector_length_each_hand: 63
language_at_frame: stationary
```

Clip ep9 có `head_body_pose` 258 row ở 20 Hz, format
`pose_w2b_quat_wxyz_trans_v1`.

### Đặc trưng và conversion loss

- Video 20 FPS, 512×512; khác rate/aspect ratio hai source còn lại.
- Head/body pose phải đi cùng stereo calibration.
- Hand là prediction, không phải native tracking/mocap.
- Raw note nói left hand dùng mirrored-right HaMeR convention.
- Validity thấp: tổng sample left `453/492`, right `388/492`.
- Raw có IMU, full body, point cloud, MANO parameters và caption hierarchy.
  Canonical sample chỉ giữ core head/hand, `head_body_pose` và joystick language.
- Cả 12 XP10M narrative row trong sample đều là `joystick_v1`; không có
  `xp10m_subtask_v1` hoặc `xp10m_action_v1`.

## 5. So sánh feature và scale

| Thuộc tính | EgoDex | EgoVerse | XP10M |
| --- | --- | --- | --- |
| Native FPS | 30 | 30 | 20 |
| Sample resolution | 1920×1080 | 640×480 | 512×512 |
| Finger vector/tay | 288 | 63 | 63 |
| Finger semantics | 24 joint SE(3) | kp21 wrist-relative cam | kp21 wrist-relative cam |
| Hand source | ARKit tracking | mocap | predicted MANO/HaMeR |
| Head source | ARKit | Aria SLAM | body SLAM + calibration |
| Sample validity | 100%/100% | 100%/100% | 92,1%/78,9% L/R |
| Extra sensor trong canonical | không | không | `head_body_pose` |
| Language trong sample | hand/ego + joystick | native/hand + joystick | joystick only |
| License | Apple dataset terms | internal terms | non-commercial |

Canonical sample lưu position theo mét và rotation matrix; world frame chỉ có
nghĩa trong một clip. Hai vector cùng shape giữa các source không chứng minh
chúng đã cùng coordinate semantics.

## 6. Mức độ bằng chứng

### Verified

- Bundle có 6 clip, 1.452 canonical frame, 56,6 s và 30 narrative row.
- Sáu annotation, sáu narrative, hai XP10M stream và sáu video đều đọc được.
- Frame liên tục `0..n_frames-1`, timestamp bằng `frame/fps`, validity count
  khớp manifest.
- Raw peeks có schema/shape và sample values được ghi ở trên.
- XP10M projection smoke test chiếu 21 điểm lên đúng vùng bàn tay phải ở frame
  129.

### Unknown

- Production scale chưa được đo lại từ release thật.
- Sáu clip không đủ suy ra phân phối task, motion hoặc quality production.
- EgoDex/EgoVerse chưa được projection smoke test trực quan.
- EgoVerse branch detection không thể tái lập chỉ từ attrs-only raw peek.
- XP10M caption loss quan sát trong hai clip chưa chứng minh toàn release cũng
  chỉ còn joystick.

## Lệnh khảo sát

```bash
find dataset/corpus_sample_bundle -type f -printf '%p\t%s bytes\n'
du -sh dataset/corpus_sample_bundle

uv run --no-project --with pyarrow --with h5py \
  python3 <metadata-inspection>

ffprobe -v error -select_streams v:0 \
  -show_entries format=filename,duration,size:stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames \
  -of json dataset/corpus_sample_bundle/videos/<source>/<clip>.mp4

uv run --no-project --with numpy --with pyarrow \
  --with opencv-python-headless \
  python3 dataset/corpus_sample_bundle/load_example.py
```

Metadata inspection dùng `pyarrow 25.0.0`, `h5py 3.14.0`. Loader chỉ decode
một XP10M frame; không decode toàn bộ video.
