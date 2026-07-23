# Cài đặt và cách dùng Kinetix

## Cài đặt

Kinetix đang là submodule vendored của repo RTC. Chạy từ repo RTC:

```bash
cd /home/dung/Workspace/VinRobotics/third_party/01_real-time-chunking-kinetix
git submodule update --init
uv sync
```

Hoặc nếu dùng Kinetix độc lập, upstream hướng dẫn:

```bash
cd third_party/kinetix
pip install -e .
```

Package yêu cầu Python `>=3.10` và khai báo JAX CUDA 12 cùng các dependency Jax2D, JaxGL,
Flax, Optax, Gymnax, JaxUED, Hydra, WandB, Pygame và ImageIO
([`pyproject.toml`](../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/pyproject.toml)).
Việc cài đặt chưa được hoàn tất trong workspace vì tải CUDA wheel rất lớn.

## Chạy ví dụ level có sẵn

Từ root Kinetix:

```bash
cd third_party/01_real-time-chunking-kinetix/third_party/kinetix
python3 examples/example_premade_level_replay.py
```

Ví dụ load `worlds/l/grasp_easy.json`, tạo pixel/continuous/replay environment, sample một
action, step một lần và mở Matplotlib
([`example_premade_level_replay.py`](../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/examples/example_premade_level_replay.py)).

## Chạy level ngẫu nhiên

```bash
python3 examples/example_random_level_replay.py
```

Ví dụ dùng `sample_kinetix_level`, sau đó reset/step/render giống level có sẵn
([`example_random_level_replay.py`](../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/examples/example_random_level_replay.py)).

## API tối thiểu

```python
import jax

from kinetix.environment.env import make_kinetix_env_from_args
from kinetix.environment.env_state import EnvParams
from kinetix.util.saving import load_from_json_file

level, static_params, env_params = load_from_json_file(
    "worlds/l/grasp_easy.json"
)
env = make_kinetix_env_from_args(
    obs_type="symbolic",
    action_type="continuous",
    reset_type="replay",
    static_env_params=static_params,
)

rng = jax.random.key(0)
rng, reset_key, action_key, step_key = jax.random.split(rng, 4)
obs, state = env.reset_to_level(reset_key, level, env_params)
action = env.action_space(env_params).sample(action_key)
obs, state, reward, done, info = env.step(
    step_key, state, action, env_params
)
```

## Train PPO

```bash
python3 experiments/ppo.py
```

Mặc định dùng entity observation, multi-discrete action, small environment và random level.
Một số biến thể:

```bash
# Random level, medium capacity
python3 experiments/ppo.py train_levels=random env_size=m

# Một handmade level
python3 experiments/ppo.py \
  train_levels=s \
  train_levels.train_levels_list='["s/h4_thrust_aim.json"]'

# Toàn bộ large holdout levels
python3 experiments/ppo.py \
  train_levels=l \
  env_size=l \
  eval_env_size=l
```

`env_size` phải đủ chứa level. Path trong `train_levels_list` tương đối với `worlds/`.

## Train PLR hoặc SFL

```bash
python3 experiments/plr.py
python3 experiments/sfl.py
```

- PLR duy trì replay buffer level và luân phiên domain-randomization, replay, mutation tùy
  config.
- SFL sample nhiều level, đo learnability rồi ưu tiên tập level có tiềm năng học cao.

Hai script đều là workload nghiên cứu lớn, dùng WandB và không có smoke test được xác minh
trong workspace.

## Hydra overrides

Config có cấu trúc nhóm:

```text
env, env_size, learning, misc, eval,
eval_env_size, train_levels, model, ued
```

Override dùng cú pháp Hydra:

```bash
python3 experiments/ppo.py \
  env=symbolic \
  model.transformer_depth=8 \
  learning.total_timesteps=100000
```

Các key cụ thể được giải thích trong
[`docs/configs.md`](../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/docs/configs.md).

## Mở editor

```bash
python3 kinetix/editor.py
```

Editor native dùng Pygame và Hydra `configs/editor.yaml`. Upstream khuyên dùng KinetixJS
editor thay vì editor native. World JSON export có thể load bằng `load_from_json_file`.

## Bẫy vận hành

- Các command giả định current directory là root Kinetix.
- Experiments gọi WandB theo config; dùng `misc.wandb_mode=offline` hoặc tắt WandB nếu
  không muốn tác động dịch vụ ngoài.
- Pixel rendering cần nhiều memory hơn symbolic/entity.
- `StaticEnvParams` thay đổi shape và kích hoạt JAX recompile.
- Không coi ví dụ một-step là xác minh training pipeline; báo cáo này chưa chạy runtime.
