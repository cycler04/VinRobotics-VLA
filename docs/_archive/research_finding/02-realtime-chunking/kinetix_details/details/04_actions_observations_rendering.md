# Action, observation và rendering

## Action spaces

### Continuous

Agent phát vector dài:

```text
num_motor_bindings + num_thruster_bindings
```

Motor bị clip `[-1, 1]`, thruster `[0, 1]`. Binding mở rộng một action sang mọi joint hoặc
thruster có cùng binding. `motor_auto=True` ép motor tương ứng thành `+1`.

**Discrepancy:** Gymnax `Box` công bố low `-1` cho cả thruster, nhưng processing clip
thruster âm thành `0`
([`env.py:73`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/environment/env.py),
[`env.py:310`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/environment/env.py)).

### Discrete

Một categorical action:

```text
2 * motor_bindings + 1 no-op + thruster_bindings
```

Mỗi motor có forward/backward full-power; mỗi thruster có một on action. Chỉ một binding
được điều khiển chủ động tại mỗi step.

### Multi-discrete

Environment nhận vector categorical dài `motor_bindings + thruster_bindings`:

- motor: `{off, +1, -1}`;
- thruster: `{off, +1}`.

Actor phát logits phẳng dài `3*motor_bindings + 2*thruster_bindings`;
`MultiDiscreteActionDistribution` chia logits thành từng categorical rồi sample vector trên
([`action_spaces.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/models/action_spaces.py)).

### Hybrid

Model layer có distribution ghép categorical + Gaussian, nhưng `env.py` không đăng ký
Hybrid environment. Đây là capability rời rạc/chưa nối end-to-end trong checkout này.

## Observation spaces

### Symbolic-flat

`make_render_symbolic` tạo một vector phẳng từ feature polygon, circle, joint, thruster và
gravity. Slot inactive được zero; ba polygon wall/ceiling index `1,2,3` bị bỏ. Output được
clip `[-10, 10]` và thay NaN
([`renderer_symbolic_flat.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/render/renderer_symbolic_flat.py)).

Ưu điểm là nhanh; nhược điểm là dimension phụ thuộc environment capacity và không
permutation-invariant.

### Symbolic-entity

`EntityObservation` giữ tensor riêng cho polygon, circle, joint, thruster, active masks,
index liên kết và attention mask bốn channel:

1. fully connected shapes;
2. multi-hop/collision-permitted relation;
3. one-hop joint relation;
4. current collision manifold.

Format này phục vụ `ActorCriticTransformer` và giữ cấu trúc graph rõ hơn vector phẳng
([`renderer_symbolic_entity.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/render/renderer_symbolic_entity.py)).

### Pixels

`make_render_pixels` dùng JaxGL để rasterize object, joint và thruster. RL wrapper chia pixel
cho `255` và trả:

```text
PixelsObservation(
    image=<float image>,
    global_info=[gravity_y / 10]
)
```

([`renderer_pixels.py:274`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/render/renderer_pixels.py)).

### Blind

Chỉ trả one-hot của timestep dài `max_timesteps + 1`. Nó hữu ích làm baseline không quan sát
state vật lý.

## Feature construction

`renderer_symbolic_common.py` là nguồn chung cho feature shape/joint/thruster. Nó chuẩn hóa
vị trí, velocity, rotation, vật lý, role, binding và mask trước khi flat/entity renderer
đóng gói
([`renderer_symbolic_common.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/render/renderer_symbolic_common.py)).

`textures.py` load các PNG package assets cho renderer. `rel_multi_head.py` chứa một
attention implementation riêng có relative shift/mask helper, nhưng Transformer chính
hiện import Flax `MultiHeadDotProductAttention`, không import class custom này.

## Giới hạn

- Pixel observation là partially observable đối với density, restitution và thuộc tính
  không nhìn thấy.
- Flat observation không chuyển trực tiếp giữa size `s/m/l`.
- Entity observation nhanh hơn pixel nhưng attention mask tăng theo bình phương số shape.
- `PixelObservations.observation_space` mô tả `Box`, còn giá trị thực là
  `PixelsObservation` dataclass gồm image + global info; đây là schema mismatch cần kiểm
  runtime consumer.

