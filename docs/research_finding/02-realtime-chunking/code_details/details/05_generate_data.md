# Module `src/generate_data.py`

## Vai trò

Module load nhiều expert seed cho từng level, chọn ngẫu nhiên một expert hợp lệ cho mỗi
environment/episode, rollout song song và ghi demonstration dạng NPZ.

## Input

`run_path` phải có layout do `train_expert.py` tạo:

```text
<run_path>/seed_<n>/<numeric-update>/
├── stats/worlds_l_<level>.json
└── policies/worlds_l_<level>.pkl
```

Mỗi seed chọn một checkpoint như sau:

1. đọc tất cả thư mục update có tên số;
2. lấy những checkpoint có solve rate `>= 0.65`;
3. chọn ngẫu nhiên một checkpoint đạt ngưỡng;
4. nếu không có checkpoint đạt ngưỡng, lấy argmax nhưng đánh mask `False`.

Do đó code không luôn “load best checkpoint” như README diễn đạt; nó chọn ngẫu nhiên trong
tập đạt ngưỡng. Checkpoint argmax của seed không đạt ngưỡng được load nhưng không được chọn
cho rollout
([`generate_data.py:95-127`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/generate_data.py#L95-L127)).

## Environment và policy mixture

Wrapper stack:

```text
Kinetix symbolic continuous
-> NoisyActionWrapper
-> ObsHistoryWrapper(history=4)
-> ActionHistoryWrapper
-> AutoReplayWrapper
-> LogWrapper
-> BatchEnvWrapper
```

Expert nhận observation có history giống lúc train. Nhưng field `Data.obs` được lấy bằng
`ObsHistoryWrapper.get_original_obs`, nên dataset lưu observation symbolic gốc, không lưu
history stack hoặc action được nối vào expert input
([`generate_data.py:79-93`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/generate_data.py#L79-L93),
[`generate_data.py:175-188`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/generate_data.py#L175-L188)).

Mỗi environment giữ một `policy_idx`. Index chỉ đổi tại `done`, nên một episode không trộn
expert ở giữa. Nếu đặt `action_sample_std`, code thay std học được bằng hằng số trước khi
sample.

**Semantics quan trọng:** NPZ lưu action do expert ra lệnh. `NoisyActionWrapper` thêm noise
sau đó trong environment step, nên action thực thi trong physics khác action lưu. Dataset
không ghi noise realization.

## Data contract

`Data` có sáu field:

| Field | Ý nghĩa | Shape trước khi tách level |
|---|---|---|
| `obs` | observation symbolic gốc trước action | `[level, step, env, obs_dim]` |
| `action` | commanded expert action | `[level, step, env, action_dim]` |
| `done` | episode kết thúc sau action | `[level, step, env]` |
| `solved` | solved của episode vừa return | `[level, step, env]` |
| `return_` | return của episode vừa return | `[level, step, env]` |
| `length` | length của episode vừa return | `[level, step, env]` |

`num_steps` là cận dưới. Số step thực được làm tròn lên bội của
`num_envs * batch_size`
([`generate_data.py:24-60`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/generate_data.py#L24-L60),
[`generate_data.py:71-77`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/generate_data.py#L71-L77)).

Sau collection, module chuyển batch result về CPU, nối các batch theo step và ghi:

```text
<run-path>/data/worlds_l_<level>.npz
```

Mỗi NPZ vẫn giữ hai trục `[step, env, ...]`; nó không tách file theo episode
([`generate_data.py:192-223`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/generate_data.py#L192-L223)).

## Giới hạn và failure mode

- Nếu không seed nào của một level đạt threshold, `good_policy_mask.sum()==0`; logic
  `randint(..., maxval=0)` không có fallback hợp lệ.
- Danh sách thư mục update không được sort. Random choice có seed nên có thể lặp lại trên
  cùng filesystem listing, nhưng mapping index không được chuẩn hóa giữa filesystem.
- Toàn bộ các batch được giữ trong list `data` rồi mới stack; collection lớn cần nhiều host
  RAM.
- NPZ không ghi level config, expert ID theo transition, noise thực thi, schema version hay
  provenance checkpoint.
