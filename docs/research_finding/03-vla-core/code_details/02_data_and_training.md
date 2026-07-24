# Dữ liệu, action contract và training

## Ranh giới dữ liệu

Snapshot không tự đọc một dataset chuẩn độc lập. `CorpusPretrainDataset` là adapter mỏng quanh
`corpus.labels.pretrain_loader.Layer1PretrainSampler`, một package thuộc repo `data_corpus`
không có trong workspace. Ba path release mặc định còn trỏ tuyệt đối tới `/mnt/SSD4/...`.

Vì vậy, các phần sau được xác minh trong snapshot:

- cách window do sampler trả về được pack thành tensor;
- cách decode frame, tạo text, collate và tính loss;
- các hyperparameter được train loop truyền vào model.

Các phần chưa xác minh:

- schema chính xác và unit của từng field upstream;
- cách `n_windows()` và `sample()` chọn timestamp/window;
- split train/val có thực sự group-disjoint;
- action có được normalize trước `pack_actions()` hay không;
- số lượng `4.48M windows` và license từng source.

## Action contract run-1

Một action chunk có `16` step ở `10 Hz`, tức horizon danh nghĩa `1,6 s`. Mỗi step có 153
chiều:

| Slice | Kích thước | Semantics theo code | Mask |
| --- | ---: | --- | --- |
| `0:3` | 3 | `head_d_pos` | luôn 1 |
| `3:9` | 6 | hai cột đầu của `head_d_rot` | luôn 1 |
| `9:12` | 3 | left `pos_cam` | left `valid` |
| `12:18` | 6 | left `rot_cam` dạng 6D | left `valid` |
| `18:81` | 63 | left `kp21`, flatten `21 × 3` | left `valid` |
| `81:84` | 3 | right `pos_cam` | right `valid` |
| `84:90` | 6 | right `rot_cam` dạng 6D | right `valid` |
| `90:153` | 63 | right `kp21`, flatten `21 × 3` | right `valid` |

`rot_to_6d()` lấy hai **cột** đầu của rotation matrix. Nếu hand là `None`, toàn bộ action và
mask của hand đó giữ bằng 0. Flow loss nhân mask theo từng phần tử rồi chia cho tổng phần tử
hợp lệ, nên hand invalid không ảnh hưởng loss.

Code không ghi:

- frame tọa độ và unit của `head_d_pos`;
- `head_d_rot` là delta ở frame nào;
- scale/unit của camera-space hand position và keypoint;
- convention camera axes;
- cách biến output 6D trở lại rotation matrix;
- normalization/denormalization statistics.

Những semantics này phải lấy từ `data_corpus/ACTION_SPEC.md` hoặc artifact gốc trước khi
output được dùng để điều khiển robot.

## Window và text supervision

Dataset enumerate window của mỗi clip với:

```text
step = max(1, int(window_stride_s * fps))
window_stride_s mặc định = 2.0
```

Danh sách `(clip_idx, window_start)` được shuffle một lần bằng seed cố định. Mỗi `__getitem__`:

1. gọi sampler ngoài để lấy một window;
2. pack action và mask;
3. gọi FFmpeg để decode đúng một frame;
4. nối tối đa hai narrative đầu và joystick state thành `text`.

FFmpeg được spawn lại cho mỗi sample và filter theo frame index. Cách này tương thích AV1
theo chủ đích code nhưng có nguy cơ thành bottleneck I/O/CPU; snapshot không có benchmark,
cache hoặc persistent decoder.

Text có dạng:

```text
<gen:MODEL> Task context: NARRATIVE
Locomotion: JOYSTICK_OR_STATIONARY
```

Collator truyền toàn bộ chuỗi này làm `task`, rồi lấy dòng đầu làm `narrative_target`. Kết quả
assistant target xuất hiện nguyên văn bên trong user prompt. **Inferred:** đây là target
leakage cho narrative LM objective, trừ khi upstream chủ ý dùng loss này chỉ như reconstruction
sanity signal. Không có comment hoặc test chứng minh ý định đó.

## Source balancing

`SourceTemperatureSampler` nhóm index theo `source`, sau đó lấy:

$$
p(s) = \frac{n_s^\tau}{\sum_j n_j^\tau}
$$

- `tau=1`: gần phân phối window tự nhiên;
- `tau=0`: uniform theo source;
- mặc định `tau=0.5`: làm phẳng chênh lệch giữa các source.

Sau khi chọn source, sampler chọn ngẫu nhiên đều một index trong source đó, có replacement.
Một epoch sinh đúng `len(dataset)` index, nên một số window có thể lặp và một số không được
thấy. RNG là `numpy.RandomState` nằm trong sampler; code không lưu state của sampler trong
checkpoint.

## Flow-matching objective

Với ground-truth action $a$, noise $\epsilon$ và timestep $t$:

$$
x_t = (1-t)\epsilon + ta
$$

$$
v^\* = a - \epsilon
$$

Action head dự đoán $\hat{v}(x_t, t, c)$ với conditioning $c$ từ vision/narrative/proprio.
Loss action là masked mean squared error:

$$
\mathcal{L}_{action}
=
\frac{\sum m \odot (\hat{v} - v^\*)^2}
     {\sum m + 10^{-8}}
$$

Timestep được lấy từ `s ~ Beta(1.5, 1.0)`, rồi `t = (1-s) × 0.999`. MSE được ép sang FP32
để ổn định dù model dùng BF16 mặc định.

Total loss được viết là:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{action}
+ 0.1\mathcal{L}_{narrative}
$$

Tuy nhiên train loop tạo config với inner language model và vision model đều frozen.
Narrative loss chỉ phụ thuộc output Qwen, không đi qua action head. `VLAModel.__init__()` không
freeze rõ outer `self.qwen.lm_head`; tùy `lm_head` có parameter riêng hay tie với embedding,
loss này có thể chỉ cập nhật output head hoặc không cập nhật parameter nào. Vì chưa load được
model trong workspace, trạng thái này là **Unknown**. Muốn gọi đây là dual training đúng nghĩa
cần log trainable parameter theo component và xác minh gradient runtime.

Ngoài ra, comment `train_narrative` trong config gọi action term là “L1”, nhưng
implementation dùng MSE. Code loss là nguồn sự thật ở đây.

## Train loop hiện có

`pretrain.py` cung cấp:

- AdamW, learning rate mặc định `1e-4`, weight decay `0.01`;
- batch `8`, gradient accumulation `4`;
- clip gradient norm ở `1.0`;
- temperature sampler `tau=0.5`;
- `--overfit N` để giới hạn danh sách window;
- log loss và throughput;
- save `model.state_dict()` theo chu kỳ và ở cuối.

Nó chưa cung cấp:

- validation dataloader/evaluation metric;
- learning-rate scheduler hoặc warmup;
- optimizer, scheduler, scaler, RNG và dataloader state trong checkpoint;
- resume logic;
- mixed-precision context hoặc distributed wrapper;
- experiment metadata gồm model config, source revision và dependency versions;
- best-checkpoint selection hoặc early stopping.

File `config.json` cạnh checkpoint chỉ lưu CLI args, không serialize `VLAConfig`, release
content, model revision hoặc corpus schema version.

## Inference và action output

Sampling bắt đầu từ Gaussian noise và dùng explicit Euler với `num_steps=4`, `dt=1/4`,
timestep lần lượt `0`, `0.25`, `0.5`, `0.75`. Nó không evaluate velocity tại `t=1`.

Output được mô tả là normalized action. Snapshot không có:

- inverse normalization;
- temporal execution policy cho 16-step chunk;
- action safety/clipping;
- camera/control synchronization;
- closed-loop receding-horizon runtime;
- robot-specific command adapter.

Vì vậy model output hiện là tensor prediction contract, chưa phải lệnh robot có thể thực thi.
