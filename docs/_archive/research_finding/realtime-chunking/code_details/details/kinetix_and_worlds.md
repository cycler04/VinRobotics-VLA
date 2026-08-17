# Dependency Kinetix và `worlds/l`

## Phạm vi sử dụng

Repo vendored toàn bộ Kinetix, nhưng pipeline RTC chỉ gọi bốn nhóm API:

Giải thích đầy đủ dependency nằm tại
[báo cáo chi tiết Kinetix](../../kinetix_details/overview.md). File này chỉ giữ phần giao
nhau trực tiếp với RTC.

| Import | API được dùng | Vai trò |
|---|---|---|
| `kinetix.environment.env` | `make_kinetix_env_from_name` | tạo symbolic continuous environment |
| `kinetix.environment.env_state` | `EnvState`, `EnvParams`, `StaticEnvParams` | contract state và config |
| `kinetix.environment.wrappers` | base wrapper, `AutoReplayWrapper`, `LogWrapper` | reset cùng level và metric episode |
| `kinetix.util.saving` | `load_from_json_file` | deserialize world JSON |
| `kinetix.render.renderer_pixels` | `make_render_pixels` | render frame/video |

Các package Kinetix về UED, PCG, model RL có trong submodule nhưng không được top-level RTC
code import. Các Hydra YAML vendored cũng không điều khiển sáu script RTC; chúng dùng
dataclass + Tyro.

## Environment contract

Mọi module tạo:

```python
make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", ...)
```

Factory map tên này tới `KinetixSymbolicContinuousActions`
([`env.py:207-223`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/environment/env.py)).
`EnvParams` mặc định có `max_timesteps=256`; `StaticEnvParams` định nghĩa capacity vật thể,
frame skip và số binding
([`env_state.py:26-43`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/environment/env_state.py)).

RTC code cố định capacity:

```text
12 polygon
4 circle
6 joint
2 thruster
4 motor binding
2 thruster binding
frame_skip = 2
```

Đây là capacity tensor tĩnh, không phải mọi slot đều active trong mọi level.

## World JSON

Có 12 file:

```text
car_launch              cartpole_thrust
catapult                catcher_v3
chain_lander            grasp_easy
h17_unicycle            hard_lunar_lander
mjc_half_cheetah        mjc_swimmer
mjc_walker              trampoline
```

Mỗi JSON có ba phần:

- `env_state`: polygon, circle, joint, thruster, gravity và collision state;
- `env_params`: physics/runtime params, gồm `max_timesteps=256`;
- `static_env_params`: capacity, binding count, frame skip và render shape.

`saving.load_from_json_file` parse JSON rồi trả tuple
`(EnvState, StaticEnvParams, EnvParams)`
([`saving.py:307-336`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/util/saving.py),
[`saving.py:534-536`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/util/saving.py)).

Số slot active đo trực tiếp từ JSON:

| Level | Polygon | Circle | Joint | Thruster |
|---|---:|---:|---:|---:|
| `car_launch` | 10 | 2 | 3 | 0 |
| `cartpole_thrust` | 9 | 0 | 1 | 2 |
| `catapult` | 8 | 2 | 1 | 0 |
| `catcher_v3` | 12 | 2 | 3 | 2 |
| `chain_lander` | 10 | 1 | 4 | 2 |
| `grasp_easy` | 12 | 2 | 6 | 0 |
| `h17_unicycle` | 6 | 2 | 2 | 0 |
| `hard_lunar_lander` | 8 | 0 | 2 | 2 |
| `mjc_half_cheetah` | 12 | 0 | 6 | 0 |
| `mjc_swimmer` | 8 | 0 | 2 | 0 |
| `mjc_walker` | 12 | 0 | 6 | 0 |
| `trampoline` | 8 | 2 | 2 | 0 |

So với 74 world trong submodule Kinetix, sáu file top-level byte-identical, hai tên tồn tại
nhưng nội dung khác và bốn file chỉ có ở top-level. **Unknown:** repo không ghi provenance
hoặc lý do của các biến thể này, nên không nên thay thế `worlds/l` bằng bộ world vendored.

## Wrapper semantics từ Kinetix

- `AutoReplayWrapper` lưu level ban đầu trong state và reset về đúng level đó khi episode
  kết thúc.
- `LogWrapper` duy trì return/length, đặt `returned_episode_solved = info["GoalR"]` và
  `returned_episode = done`
  ([`wrappers.py:268-309`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/environment/wrappers.py)).
- Renderer nhận `EnvState`, dùng `screen_dim/downscale`, rồi trả pixel buffer; RTC code tự
  transpose và flip để ghi ảnh/video.

## Dependency và khả năng tái lập

Submodule được ghim trong checkout tại commit `cf7453e`, nhưng `pyproject.toml` của Kinetix
có dependency Git `jaxued @ ...@main`. `uv.lock` có thể khóa resolution hiện tại, song việc
resolve lại ngoài lock sẽ phụ thuộc upstream mutable branch. Kinetix cũng khai báo
`jax[cuda12_pip]`, trong khi repo cha khai báo `jax[cuda12]`; cài đặt đòi CUDA wheel lớn.

**Unknown:** báo cáo chưa xác minh simulator chạy đúng trên CPU-only, GPU hiện tại hoặc
Python 3.13. Chỉ source contract và checkout revision đã được kiểm tra.
