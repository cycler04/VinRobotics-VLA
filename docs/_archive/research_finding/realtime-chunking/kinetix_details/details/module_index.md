# Index toàn bộ module Kinetix

## Package `kinetix`

| File | Vai trò |
|---|---|
| `kinetix/__init__.py` | Package marker, không export API |
| `kinetix/editor.py` | Native Pygame level editor |

## `kinetix/environment`

| File | Vai trò |
|---|---|
| `environment/__init__.py` | Package marker |
| `environment/env_state.py` | `EnvState`, `EnvParams`, `StaticEnvParams` |
| `environment/env.py` | Environment factory, action conversion, reward và physics step |
| `environment/wrappers.py` | Reset/replay, Gymnax adapter, batch, dense reward và logging |
| `environment/utils.py` | Permute concrete/PCG state và remap index |

## `kinetix/environment/ued`

| File | Vai trò |
|---|---|
| `ued_state.py` | `UEDParams` |
| `distributions.py` | Random level distribution và no-op filtering |
| `mutators.py` | Primitive mutation shape/joint/thruster/gravity |
| `util.py` | Geometry/role/state helpers cho generator |
| `ued.py` | Compose sampler, mutation và reset function cho experiment |

## `kinetix/pcg`

| File | Vai trò |
|---|---|
| `pcg/__init__.py` | Package marker |
| `pcg_state.py` | Range/mask template `PCGState` |
| `pcg.py` | Sample template và convert concrete state |

## `kinetix/render`

| File | Vai trò |
|---|---|
| `render/__init__.py` | Package marker |
| `renderer_pixels.py` | JaxGL pixel renderer và RL pixel observation |
| `renderer_symbolic_common.py` | Shared entity feature extraction |
| `renderer_symbolic_flat.py` | Flatten feature thành vector |
| `renderer_symbolic_entity.py` | Structured entity tensors và attention mask |
| `textures.py` | Load package texture assets |

## `kinetix/models`

| File | Vai trò |
|---|---|
| `models/__init__.py` | Network factory theo environment/config |
| `action_spaces.py` | Hybrid và multi-discrete Distrax distribution |
| `actor_critic.py` | GRU, CNN/symbolic encoder và shared actor-critic |
| `transformer_model.py` | Entity Transformer actor-critic |
| `rel_multi_head.py` | Custom relative multi-head attention utilities, không ở factory path chính |

## `kinetix/util`

| File | Vai trò |
|---|---|
| `util/__init__.py` | Package marker |
| `config.py` | Hydra normalization, params, WandB và log helpers |
| `learning.py` | Eval, rollout, GAE và PPO update dùng chung |
| `saving.py` | World/PCG/checkpoint serialization và WandB artifact |
| `timing.py` | Helper đo thời gian một function |

## Entry points ngoài package

| File | Vai trò |
|---|---|
| `experiments/ppo.py` | PPO baseline trên random hoặc handmade level |
| `experiments/plr.py` | Prioritized Level Replay/DR/ACCEL experiment |
| `experiments/sfl.py` | Sampling for Learnability experiment |
| `examples/example_premade_level_replay.py` | Minimal JSON level replay |
| `examples/example_random_level_replay.py` | Minimal random UED level replay |

## Resource không phải Python

| Path | Vai trò |
|---|---|
| `configs/` | 37 Hydra YAML |
| `worlds/` | 74 handmade world JSON: 10 small, 24 medium, 40 large |
| `kinetix/assets/` | Texture/icon cho renderer/editor |
| `images/` | README và documentation media |
| `docs/` | Usage/config documentation upstream |

## Call graph rút gọn

```text
experiments/*
├── util/config
├── environment/env + wrappers
│   ├── Jax2D physics
│   ├── render/*
│   └── pcg/* + environment/ued/*
├── models/*
├── util/learning
└── util/saving

examples/*
├── environment/env
├── environment/ued/distributions hoặc util/saving
└── render/renderer_pixels

editor
├── environment + PCG/UED
├── renderer
└── saving
```

## Ranh giới

Index này mô tả file có trong checkout, không tuyên bố mọi module đều nằm trên runtime path.
Các marker `__init__.py`, custom relative attention, hybrid action và một số legacy helper
không được entry point chính gọi trực tiếp.

