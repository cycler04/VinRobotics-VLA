# Module `src/model.py`

## Vai trò

Module định nghĩa toàn bộ policy behavior cloning: config, positional embedding, lịch
trọng số prefix, MLP-Mixer, flow-matching loss và ba cách sinh action chunk.

Contract mặc định:

```text
observation:  [batch, obs_dim]
action/noise: [batch, H=8, action_dim]
flow time:    [batch] hoặc [batch, H]
velocity:     [batch, H=8, action_dim]
```

`ModelConfig` dùng channel 256, hidden channel 512, hidden token 64, bốn Mixer block và
chunk tám action
([`model.py:11-18`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py#L11-L18)).

## Thành phần

### `posemb_sincos`

Biến scalar flow time thành sinusoidal embedding chẵn chiều. Period trải log-uniform từ
`4e-3` đến `4.0` khi được gọi trong policy. Hàm từ chối embedding dimension lẻ.

### `get_prefix_weights`

Tạo vector trọng số dài `H` để đo/guidance độ khớp với chunk cũ:

- `ones`: một cho mọi vị trí trước `end`;
- `zeros`: hard prefix, một trước `start`, còn lại bằng không;
- `linear`: giảm tuyến tính từ vùng cố định sang vùng tự do;
- `exp`: biến đổi lịch tuyến tính thành suy giảm mạnh hơn.

`end` thắng `start`: nếu `end < start`, code hạ `start` xuống `end`
([`model.py:40-63`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py#L40-L63)).

### `MLPMixerBlock`

Mỗi block có hai residual branch:

1. token mixing trên trục thời gian của chunk;
2. channel mixing trên feature channel.

Cả hai dùng LayerNorm không affine và AdaLN zero-init tạo `scale`, `shift`, `gate` từ
flow-time embedding. Zero-init gate làm block ban đầu gần identity
([`model.py:66-100`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py#L66-L100)).

### `FlowPolicy.__call__`

Policy lặp cùng observation tại mọi token, nối nó với noisy action token, project lên
channel dimension, chạy Mixer stack rồi project về `action_dim`. Model không có image
encoder, language encoder hay causal mask
([`model.py:103-158`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py#L103-L158)).

## Các sampler

### `action`: flow sampling chuẩn

Khởi tạo Gaussian noise và tích phân Euler từ `t=0` đến `1` trong `num_steps`:

```text
x <- x + (1 / num_steps) * v_theta(obs, x, t)
```

Evaluation mặc định dùng năm flow step
([`model.py:160-171`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py#L160-L171)).

### `realtime_action`: RTC

Có hai nhánh loại trừ nhau:

- `simulated_delay is None`: inference-time RTC. Mỗi flow step tính denoised estimate,
  VJP của denoiser và pseudo-inverse correction để kéo chunk mới về chunk cũ theo soft
  prefix weights. Guidance bị chặn bởi `max_guidance_weight`.
- `simulated_delay is not None`: training-time RTC. Các vị trí trước `inference_delay`
  được thay trực tiếp bằng prefix sạch, time của chúng đặt thành `1`, rồi chạy forward
  policy bình thường.

Vì nhánh training-time không đọc `prefix_attention_schedule` hay `max_guidance_weight`,
hai cấu hình `realtime` và `hard_masking` trong evaluator trở thành cùng một phép sinh khi
load model có `simulated_delay`
([`model.py:219-265`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py#L219-L265)).

### `bid_action`: sample-and-rank

Sinh `n_samples` chunk, tính backward loss so với chunk trước trên vùng overlap, rồi chọn
chunk loss thấp nhất. Nếu có `weak_policy` và `bid_k`, code thêm forward contrast giữa top-k
sample của strong và weak policy. Mặc định `bid_k=None`, nên evaluator chỉ dùng backward
rejection sampling và không cần weak checkpoint
([`model.py:173-217`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py#L173-L217)).

## Loss

Loss chuẩn sample:

```text
noise ~ N(0, I)
t ~ Uniform(0, 1)
x_t = (1 - t) noise + t action
target velocity = action - noise
loss = mean((v_theta(obs, x_t, t) - target velocity)^2)
```

Khi `simulated_delay=N`, code sample delay nguyên trong `[0, N)` với xác suất tăng theo
`exp(delay)`, đặt prefix tương ứng thành action sạch tại `t=1`, và chỉ tính loss trên postfix.
Vì cận trên loại trừ, `simulated_delay=5` train delay `0..4`, không phải delay 5
([`model.py:267-289`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py#L267-L289)).

## Giới hạn

- Euler solver có số bước cố định; không có adaptive ODE solver.
- Checkpoint không tự chứa `ModelConfig`; load sai chunk size hoặc hidden dimension sẽ không
  tái tạo đúng graph.
- Inference-time RTC cần VJP trong từng flow step nên đắt hơn `action`.
- Các `assert` shape là contract runtime chính; module không có unit test trong repo.
