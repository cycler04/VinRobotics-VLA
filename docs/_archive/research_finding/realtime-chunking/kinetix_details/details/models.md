# Model và action distribution

## Factory

`make_network_from_config` chọn network dựa trên `env_name`:

| Observation | Network |
|---|---|
| Pixels | `ActorCriticPixelsRNN` |
| Symbolic hoặc Blind | `ActorCriticSymbolicRNN` |
| Entity | `ActorCriticTransformer` |

Action mode được suy từ tên environment. Factory lấy action dimension từ Gymnax space,
đọc config recurrence/MLP/Transformer rồi tạo Flax Linen module
([`models/__init__.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/models/__init__.py)).

## Shared actor-critic

`GeneralActorCriticRNN` nhận embedding observation, tùy chọn đưa qua GRU 256 chiều, rồi tách:

- actor MLP -> action distribution;
- critic MLP -> scalar value.

`ScannedRNN` reset hidden state theo `done`. Discrete dùng categorical, continuous dùng
diagonal Gaussian, multi-discrete dùng nhiều categorical
([`actor_critic.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/models/actor_critic.py)).

Environment mới là nơi clip/convert action; Gaussian continuous policy không tự squash
action trước khi tính log probability.

## Pixel và symbolic encoder

- Pixel model dùng hai Conv layer rồi nối `global_info`.
- Symbolic model dùng một Dense projection.
- Cả hai chuyển embedding vào shared recurrent actor-critic.

**Code mismatch:** pixel code truy cập `obs.image` và `obs.global_info`, phù hợp
`PixelsObservation`; `PixelObservations.observation_space` lại trả một image-only `Box`.

## Entity Transformer

`ActorCriticTransformer`:

1. encode circle, polygon, joint và thruster riêng;
2. nối shape tokens và mask inactive;
3. tùy chọn thêm dummy aggregation token;
4. chạy nhiều gated Transformer layer;
5. cập nhật shape embedding bằng joint/thruster relation;
6. aggregate dummy, mean hoặc cả hai;
7. đưa embedding vào shared actor-critic.

Attention mask có bốn relation channel và được repeat theo head. `full_attention_mask=True`
bỏ relation-specific structure, chỉ giữ inactive mask
([`transformer_model.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/models/transformer_model.py)).

## Distribution

`MultiDiscreteActionDistribution` chia flat logits thành từng categorical, rồi cộng log-prob
và entropy. `HybridActionDistribution` cộng log-prob/entropy của categorical và Gaussian
([`action_spaces.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/models/action_spaces.py)).

## `rel_multi_head.py`

Module chứa bản attention tương đối mở rộng từ Flax, cùng helper causal/padding mask. Trong
checkout này `transformer_model.py` import trực tiếp Flax attention, nên implementation
custom không nằm trên model factory path chính
([`rel_multi_head.py`](../../../../../third_party/01_real-time-chunking-kinetix/third_party/kinetix/kinetix/models/rel_multi_head.py)).

## Giới hạn

- GRU hidden size hard-code 256 dù helper có argument `hidden_size`.
- `add_generator_embedding=True` trong shared actor-critic raise `NotImplementedError`.
- Hybrid distribution không nối tới environment factory.
- Checkpoint phụ thuộc Hydra config để dựng đúng graph; format không tự mô tả architecture.
- Không có unit test model trong vendored repo.

