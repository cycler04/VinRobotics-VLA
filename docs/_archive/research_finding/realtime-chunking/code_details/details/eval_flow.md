# Module `src/eval_flow.py`

## Vai trò

Module có hai tầng:

- `eval(...)`: rollout một policy trên một level với một cấu hình method/delay/horizon;
- `main(...)`: load checkpoint mọi level, shard evaluation và quét toàn bộ grid thí nghiệm.

## Cấu hình

`EvalConfig` mặc định:

| Field | Mặc định | Ý nghĩa |
|---|---:|---|
| `step` | `-1` | checkpoint epoch cuối |
| `weak_step` | `None` | checkpoint weak policy cho BID đầy đủ |
| `num_evals` | `2048` | rollout song song mỗi level/config |
| `num_flow_steps` | `5` | Euler step khi sample |
| `inference_delay` | `0` | số action cũ chạy trong lúc inference |
| `execute_horizon` | `1` | số action thực thi mỗi vòng |
| `method` | naive | sampler |
| `model` | `ModelConfig()` | graph dùng để restore checkpoint |

Ba config method nằm tại
[`eval_flow.py:24-52`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py#L24-L52).

## Vòng lặp action chunk

Environment evaluation dùng:

```text
Kinetix symbolic continuous
-> NoisyActionWrapper
-> AutoReplayWrapper
-> LogWrapper
-> BatchEnvWrapper(num_evals)
```

Không có observation/action history hoặc dense reward wrapper. Đây khớp observation gốc
được lưu cho flow policy, không phải augmented observation dùng bởi expert.

Mỗi vòng:

1. policy sinh `next_action_chunk`;
2. thực thi `inference_delay` action đầu từ chunk cũ;
3. thực thi phần còn lại tới `execute_horizon` từ chunk mới;
4. bỏ `execute_horizon` token đầu của chunk mới và pad zero cuối để căn frame cho vòng sau.

```text
old chunk: [old actions chạy trong delay | phần còn lại]
new chunk: [prefix tương ứng             | postfix mới]
execute:   [old prefix                   | new tới horizon]
next old:  [new sau horizon                         | zero pad]
```

Logic nằm tại
[`eval_flow.py:71-143`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py#L71-L143).

Method dispatch:

- naive -> `FlowPolicy.action`;
- realtime -> `FlowPolicy.realtime_action`;
- BID -> `FlowPolicy.bid_action`;
- hard masking cũng gọi realtime sampler nhưng đổi schedule thành `zeros`.

Script assert `execute_horizon >= inference_delay`. Grid `main` còn đảm bảo
`execute_horizon <= H - inference_delay`.

## Metric

Rollout chạy đủ `ceil(max_timesteps / execute_horizon)` chunk iteration. Sau khi flatten
time, code lấy metric tại `done` đầu tiên của mỗi rollout và trung bình qua `num_evals`:

- `returned_episode_returns`;
- `returned_episode_lengths`;
- `returned_episode_solved`.

Video của rollout đầu tiên được render và trả bởi `eval`, nhưng `main` bỏ video
([`eval_flow.py:145-168`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py#L145-L168)).

## Grid của `main`

Với `H=8`, script quét:

```text
delay 0: horizon 1..8
delay 1: horizon 1..7
delay 2: horizon 2..6
delay 3: horizon 3..5
delay 4: horizon 4
```

Tổng cộng 24 cặp delay/horizon × 4 method × 12 level = 1.152 row CSV. Mỗi row là trung
bình của 2.048 rollout mặc định. Các method là `naive`, `realtime`, `bid`,
`hard_masking`
([`eval_flow.py:248-305`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py#L248-L305)).

Output:

```text
<output-dir>/results.csv
```

Các column gồm metric, `delay`, `method`, `level`, `execute_horizon`.

## Restore và sharding

`main` sort thư mục epoch theo số, chọn `step`, tạo graph từ `config.model`, rồi thay pure
state dict. Vì vậy CLI model config phải khớp checkpoint. Evaluation `vmap` theo level,
`shard_map` level qua device mesh; số device phải chia trục level
([`eval_flow.py:191-246`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py#L191-L246)).

## Giới hạn và code smell

- Đây là mô phỏng delay theo số controller step, không đo wall-clock inference.
- Không có flag thu hẹp grid method/delay/horizon; smoke test vẫn chạy toàn bộ grid.
- Carry `n` được shift qua mỗi iteration nhưng không tham gia quyết định hoặc metric.
- Code render video cho từng `eval` rồi bỏ, gây compute không tạo artifact.
- `weak_step` chỉ hữu ích khi cấu hình BID có `bid_k`; BID mặc định không dùng weak policy.
- Nếu model training-time RTC được load, schedule `exp` và `zeros` đều bị bỏ qua trong
  `realtime_action`; hai row method có thể tương đương về thuật toán.
