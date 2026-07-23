# Cài đặt và cách dùng

## Điều kiện chạy

Chạy các lệnh từ:

```bash
cd third_party/01_real-time-chunking-kinetix
```

Repo khai báo Python `>=3.11`, JAX CUDA 12 `0.4.35`, NumPy `1.26.4`, Tyro, Einops,
Pandas, `tqdm-loggable` và dependency Kinetix dạng editable local
([`pyproject.toml:5-28`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/pyproject.toml#L5-L28)).
Kinetix lại kéo Flax, Optax, WandB, ImageIO, TensorFlow Probability, renderer và nhiều
dependency GPU khác. Vì vậy `uv sync` là cài đặt nặng, không phải smoke test nhẹ.

```bash
git submodule update --init
uv sync
```

Trong checkout hiện tại, submodule Kinetix đã có ở commit
`cf7453ea103fa0b77348af1a39f689c658161613`. Workspace chưa xác minh một lượt `uv sync`
hoàn chỉnh hoặc khả năng chạy trên GPU hiện có.

## Pipeline chuẩn

### 1. Train expert

```bash
uv run src/train_expert.py
```

Mặc định chạy 12 level × 8 seed × `65,536,000` environment step cho mỗi cặp level/seed.
Đây là workload lớn; output nằm trong `logs-expert/<wandb-run-name>/`.

Smoke test được đề xuất, chưa xác minh runtime:

```bash
uv run src/train_expert.py \
  --config.level-paths worlds/l/grasp_easy.json \
  --config.num-seeds 1 \
  --config.num-updates 1 \
  --config.num-steps 8 \
  --config.num-envs 8 \
  --config.num-minibatches 1 \
  --config.num-epochs 1 \
  --config.log-interval 1
```

### 2. Sinh demonstration

```bash
uv run src/generate_data.py \
  --config.run-path ./logs-expert/<wandb-run-name>
```

Mặc định thu ít nhất một triệu transition cho mỗi level, làm tròn lên bội của
`num_envs * batch_size`, rồi ghi vào `<run-path>/data/`.

Smoke test được đề xuất:

```bash
uv run src/generate_data.py \
  --config.run-path ./logs-expert/<wandb-run-name> \
  --config.level-paths worlds/l/grasp_easy.json \
  --config.num-envs 8 \
  --config.batch-size 8 \
  --config.num-steps 64
```

### 3. Train flow policy

```bash
uv run src/train_flow.py \
  --config.run-path ./logs-expert/<wandb-run-name>
```

Mặc định train 32 epoch. Mỗi epoch còn chạy evaluation cho mọi execute horizon từ 1 đến 8
với `2048` rollout/level theo `EvalConfig` mặc định, nên giảm `eval.num-evals` khi smoke test:

```bash
uv run src/train_flow.py \
  --config.run-path ./logs-expert/<wandb-run-name> \
  --config.level-paths worlds/l/grasp_easy.json \
  --config.num-epochs 1 \
  --config.batch-size 32 \
  --config.eval.num-evals 8
```

### 4. Đánh giá

```bash
uv run src/eval_flow.py \
  --run-path ./logs-bc/<wandb-run-name> \
  --output-dir ./eval_output
```

Script luôn quét toàn bộ delay `0..4`, horizon hợp lệ và bốn method. Có thể giảm
`--config.num-evals`, nhưng code không có flag chọn một delay/method duy nhất:

```bash
uv run src/eval_flow.py \
  --run-path ./logs-bc/<wandb-run-name> \
  --config.num-evals 8 \
  --output-dir ./eval_output-smoke
```

README upstream ghi `--config.run-path`, nhưng `run_path` là tham số trực tiếp của
`eval_flow.main`, không nằm trong `EvalConfig`; theo chữ ký Tyro, flag đúng là `--run-path`
([`eval_flow.py:171-190`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py#L171-L190)).

### 5. Render level

```bash
uv run src/render_levels.py
```

Module này không nhận CLI option và luôn ghi vào `rendered_levels/`.

## Training-time RTC

Theo README, flow policy chuẩn được fine-tune bằng cách:

1. đổi `ModelConfig.simulated_delay` thành `5`;
2. load checkpoint epoch `24`;
3. train thêm 8 epoch.

Tyro cho phép thể hiện trực tiếp bằng CLI mà không sửa source:

```bash
uv run src/train_flow.py \
  --config.run-path <expert-run-path> \
  --config.load-dir <bc-run-path>/24 \
  --config.num-epochs 8 \
  --config.eval.model.simulated-delay 5
```

Lệnh trên là diễn giải từ dataclass và hướng dẫn upstream; chưa được chạy trong workspace.
README gốc mô tả cùng recipe tại
[`README:37-42`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/README.md#L37-L42).

## Bẫy vận hành

- Tất cả path mặc định tương đối với current working directory; chạy từ workspace root sẽ
  không tìm thấy `worlds/l/*.json`.
- `wandb.init()` được gọi vô điều kiện trong hai script train. Cần cấu hình WandB phù hợp,
  ví dụ offline mode nếu không muốn ghi ra dịch vụ ngoài.
- Top-level code import trực tiếp Flax, Optax, ImageIO, WandB và TensorFlow Probability
  nhưng không khai báo chúng là direct dependency; môi trường hiện dựa vào dependency
  transitive/lock của Kinetix.
- Expert assets upstream khoảng 60 GiB; không được tải trong lần nghiên cứu này
  ([`README:14-17`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/README.md#L14-L17)).
- `eval_flow.py` khai báo `output_dir: str | None`, nhưng luôn gọi `Path(output_dir)`;
  truyền `None` sẽ lỗi thay vì tắt output
  ([`eval_flow.py:303-305`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py#L303-L305)).
