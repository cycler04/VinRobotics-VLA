# Config, editor và examples

## Hydra config tree

Các entry config:

```text
configs/ppo.yaml
configs/plr.yaml
configs/sfl.yaml
configs/editor.yaml
```

Chúng compose các group:

| Group | Vai trò |
|---|---|
| `env` | environment name, dense reward, frame skip |
| `env_size` | capacity s/m/l/custom |
| `learning` | PPO, recurrent, SFL/UED rollout settings |
| `misc` | WandB, checkpoint, video và log |
| `eval` | level, attempt và frequency |
| `eval_env_size` | static capacity khi evaluation |
| `train_levels` | random hoặc danh sách handmade |
| `model` | MLP/Transformer/recurrent architecture |
| `ued` | PLR, SFL hoặc ACCEL behavior |

Hydra merge được chuẩn hóa thành dict phẳng hơn bởi `normalise_config`. Tên group rộng hơn
runtime: một key có trong YAML chưa chắc mọi experiment đọc nó
([`docs/configs.md`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/docs/configs.md),
[`config.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/util/config.py)).

## Editor

`kinetix/editor.py` là Pygame editor khoảng 2.600 dòng. Nó:

- tạo level mới hoặc load JSON;
- thêm/chọn/sửa shape, joint, thruster và role;
- chuyển edit/play mode;
- điều khiển motor/thruster bằng keyboard;
- import/export world;
- dùng renderer và physics engine trực tiếp.

Entry point dùng Hydra `configs/editor.yaml`
([`editor.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/editor.py)).

Upstream coi native editor là ít polished hơn KinetixJS. Assets PNG trong
`kinetix/assets/` chỉ phục vụ UI/rendering, không phải training data.

## Examples

| File | Minh họa |
|---|---|
| `example_premade_level_replay.py` | Load JSON, replay một level, step và render |
| `example_random_level_replay.py` | Sample UED level, step và render |

Cả hai unwrap state bằng chuỗi `.env_state.env_state.env_state` để render. Đây là chi tiết
phụ thuộc đúng wrapper stack; code mới nên dùng helper unwrap rõ ràng thay vì hard-code độ
sâu wrapper.

## Worlds và docs

- `worlds/s`, `worlds/m`, `worlds/l` là handmade/evaluation level theo capacity.
- `docs/README.md` giải thích action/observation/reset và custom-world flow.
- `docs/configs.md` là catalog Hydra control surface.
- `images/` là minh họa README/docs.

## Usage boundary với RTC

Top-level RTC scripts không đọc Hydra config của Kinetix và không dùng editor/experiment.
Chúng tạo environment bằng code và dùng Tyro config riêng. Không sửa `configs/` với kỳ vọng
nó thay đổi RTC training.

