# Qwen-VL và Qwen-VLA khác nhau ở đâu?

> **Câu hỏi:** Qwen-VLA khác Qwen-VL như thế nào, và Qwen-VLA thực sự kế thừa
> những gì từ Qwen-VL?
>
> **Phạm vi:** Qwen2-VL, Qwen2.5-VL và Qwen-VLA. “Adopt” chỉ được dùng cho
> component hoặc capability mà paper Qwen-VLA xác nhận; quan hệ suy ra từ tên
> model được ghi riêng. Ngày đọc: 2026-07-20.

## Câu trả lời ngắn

**Verified.** Qwen-VL là vision-language model: nó biến image/video thành visual
tokens, trộn chúng với text tokens và dùng LLM để sinh text, tọa độ, timestamp
hoặc function call đã được serialize thành token. Ngay cả khi Qwen2/2.5-VL được
gọi là visual agent, executor bên ngoài vẫn phải chuyển những token hành động đó
thành thao tác trong môi trường. [Qwen2-VL, §2.1–2.2, pp.3–7][qwen2-vl]
[Qwen2.5-VL, §2.1–2.3, pp.3–10][qwen25-vl]

**Verified.** Qwen-VLA giữ một pretrained Qwen vision-language backbone để nhìn,
ground ngôn ngữ và reasoning, nhưng thêm một **DiT flow-matching action expert**
riêng để sinh action chunk liên tục. Output mới là tensor số thực dùng cho robot,
waypoint hoặc trajectory, không phải action được viết thành vocabulary token.
[Qwen-VLA, §2.1–2.5, pp.3–6][qwen-vla]

**Điểm dễ hiểu sai:** Qwen-VLA paper không nói model được build trực tiếp trên
Qwen2-VL, Qwen2.5-VL hay Qwen3-VL. Lineage implementation duy nhất được xác nhận
là **Qwen3.5-4B vision-language backbone + action expert khoảng 1.15B tham số**.
[Qwen-VLA, §1–2.2, pp.2–4][qwen-vla]

## Khác biệt kiến trúc cốt lõi

```text
Qwen-VL
image/video ─> ViT ─> visual merger ─> visual tokens ─┐
text ─────────────────────────────────────────────────┼─> VLM ─> text tokens
                                                     └────────> box / timestamp /
                                                                function-call tokens

Qwen-VLA
image/video ─> ViT + spatial merge ─┐
text + embodiment prompt ───────────┼─> Qwen3.5 VLM ─┬─> text tokens
                                    │ hidden states  │
noisy action chunk + timestep ──────┴───────────────> DiT action expert
                                                        │ flow integration
                                                        v
                                                continuous action chunk
```

| Khía cạnh              | Qwen2/2.5-VL                                                                      | Qwen-VLA                                                                               |
| ------------------------ | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Mục tiêu               | Hiểu image/video và trả lời bằng language/structured tokens                  | Biến hiểu biết đa phương thức thành action hoặc trajectory có thể thực thi |
| Backbone                 | ViT + visual merger + autoregressive Qwen LLM                                     | Pretrained Qwen3.5-4B VLM                                                              |
| Output path              | LM head, next-token prediction                                                    | Giữ text path và thêm DiT action path                                               |
| Action representation    | Tên action, argument hoặc tọa độ được serialize thành text/function call | Tensor số thực liên tục`Y ∈ R^(H×K)`                                           |
| Action generation        | Autoregressive token decoding                                                     | Conditional flow matching, bắt đầu từ noise và dùng vài Euler steps             |
| Robot-specific context   | Action schema/tool definition nằm ngoài model                                   | Text prompt nêu robot, số arm, control frequency và prediction horizon              |
| Training signal          | Text-token loss, kể cả box/point/timestamp/function call                        | Masked flow loss cho action + next-token loss cho VL data                              |
| Closed-loop optimization | Agent loop do hệ ngoài điều phối                                             | SFT trên trajectory, sau đó PPO với simulator success reward                       |

Khác biệt quan trọng nhất là **output space**. Qwen-VL dự đoán phần tử trong
vocabulary; Qwen-VLA dự đoán một distribution trên chuỗi vector điều khiển liên
tục. Vì vậy Qwen-VLA không chỉ là Qwen-VL được fine-tune thêm robotics data.

## Qwen-VLA xác nhận đã lấy gì từ Qwen-VL stack?

### 1. Pretrained multimodal backbone

Qwen-VLA dùng Qwen3.5-4B cho perception và high-level reasoning. Visual tokens
được tạo bởi ViT có spatial merging rồi interleave trực tiếp với text tokens
trong một transformer theo early vision-language fusion. Backbone dùng gated
linear attention ở phần lớn layer và grouped-query softmax attention theo chu
kỳ để xử lý multimodal sequence dài. [Qwen-VLA, §2.2, p.4][qwen-vla]

Đây là phần “nhìn và hiểu” được giữ lại. Paper quy cho backbone các năng lực:

- fine-grained visual perception;
- referential grounding và spatial reasoning;
- hiểu image, video và instruction trong cùng context;
- multilingual instruction following;
- structured reasoning và text generation.

Các năng lực này tạo hidden states làm điều kiện cho action expert. DiT không tự
đọc pixel; nó nhận representation đã được VLM grounding với instruction.
[Qwen-VLA, §1–2.2, pp.2–4][qwen-vla]

### 2. Visual-token và language-token interface

Qwen2-VL thiết lập pattern `ViT → 2×2 token compression → visual tokens → Qwen LLM`, kèm dynamic resolution và M-RoPE cho text/image/video. Qwen2.5-VL tiếp tục
pattern này với redesigned ViT, two-layer MLP merger, native resolution, window
attention và absolute-time alignment cho video. [Qwen2-VL, §2.1,][qwen2-vl] [Qwen2.5-VL, §2.1, pp.3–5][qwen25-vl]

Qwen-VLA xác nhận giữ **pattern ở mức khái niệm**: ViT, spatial merging, visual
tokens interleaved với text và một unified multimodal transformer. Tuy nhiên,
paper không xác nhận rằng nó copy nguyên ViT checkpoint, patch size, merger,
dynamic-resolution policy hay M-RoPE implementation của Qwen2.5-VL. Những chi
tiết đó không nên ghi là direct reuse nếu chưa có source từ Qwen3.5 hoặc code.

### 3. Language grounding và prompt làm control interface

Qwen-VLA tận dụng interface text có sẵn của VLM để đưa embodiment semantics vào
model. Mỗi sample có prompt mô tả robot, single/dual arm, phần thân tùy chọn,
control frequency, action-chunk length và task instruction. Hidden states của
prompt đi cùng hidden states thị giác vào DiT, giúp một action expert dùng chung
cho nhiều platform. [Qwen-VLA, §2.3–2.4, pp.4–5][qwen-vla]

Điều này kế thừa khả năng instruction following của VL, nhưng mục tiêu mới là
condition một motor distribution chứ không chỉ sinh câu trả lời. “Unified” ở
đây cũng không có nghĩa mọi robot dùng cùng physical semantics: mỗi dataset vẫn
giữ delta/absolute EEF, joint, rotation và gripper convention riêng; tensor chỉ
được thống nhất về shape bằng zero padding và validity mask.

### 4. VL data và next-token objective

Qwen-VLA không bỏ language head sau khi thêm action expert. Trong joint training,
auxiliary VL data vẫn dùng standard next-token loss để giữ perception, grounding
và reasoning, đồng thời hạn chế catastrophic forgetting. Pretraining mixture có
general VL data 3.4%, 2D spatial grounding 2.5%, autonomous-driving VQA 2.4% và
fine-grained action captions 0.2%, bên cạnh robot/human/navigation trajectories.
[Qwen-VLA, §2.5, pp.5–6; §3.2, p.7][qwen-vla]

Ablation của paper báo co-training VL+VLA tăng RoboCasa 4.9 điểm và RoboTwin 4.6
điểm so với VLA-only trong setting được thử. Đây là bằng chứng rằng VL
supervision không chỉ để giữ khả năng trả lời text mà còn hỗ trợ policy learning;
nó chưa chứng minh mức tăng tương tự trên mọi robot hoặc task.
[Qwen-VLA, §5.2.2, pp.23–24][qwen-vla]

## Phần mới chỉ có ở Qwen-VLA

### Continuous flow-matching action expert

Action expert là single-stream DiT khoảng 1.15B tham số, gồm 16 DiT blocks. Nó
concatenate VLM hidden states với noisy action chunk, dùng joint self-attention,
AdaLN timestep conditioning và multi-section RoPE. Training học velocity field;
inference tích phân từ noise về action bằng vài Euler steps. Cơ chế này phù hợp
với action distribution đa mode và high-frequency hơn một LM head sinh token.
[Qwen-VLA, §2.2 và §2.5, pp.4–5][qwen-vla]

### Unified tensor interface, không ép cùng semantics

Mỗi target có shape cố định `H×K`. Control channel thực nằm ở prefix, phần còn
lại zero-pad; mask loại padded channel và padded timestep khỏi loss. Interface
này bao phủ:

- manipulation: delta EEF, Euler/quaternion rotation, absolute joint, gripper,
  dexterous-hand joints;
- navigation: waypoint `(Δx, Δy, Δθ)`;
- trajectory prediction: future spatial coordinates;
- egocentric human motion: wrist/hand pose trajectory.

Action được normalize theo từng dataset bằng percentile 1/99 rồi clip về
`[-1,1]`. Do đó model có một decoder dùng chung, nhưng semantics và normalization
vẫn phụ thuộc dataset/embodiment. [Qwen-VLA, §2.4, pp.4–5; §3.2.1,][qwen-vla]

### Progressive action training

Qwen-VLA phải giải quyết việc VLM đã pretrained nhưng DiT còn random:

1. **T2A:** freeze VLM, không đưa image, train DiT từ instruction và embodiment
   prompt để học language-conditioned action prior.
2. **CPT:** unfreeze cả VLM và DiT, joint train để ground action vào visual
   observation.
3. **SFT:** một branch multi-task cho VQA/grounding/manipulation/navigation và
   một branch teleoperation cho robot thật.
4. **RL:** từ multi-task SFT, dùng PPO và binary success reward trong
   SimplerEnv để tối ưu closed-loop success.

[Qwen-VLA, §3.1, pp.6–7; §4, pp.13–15][qwen-vla]

## Những gì chưa thể gọi là “adopted”

**Unknown.** Qwen-VLA paper không nêu tokenizer, vocabulary, exact ViT
configuration, patch size, spatial-merger dimensions hay Qwen3.5 context length.
Nó cũng không nói dùng trực tiếp Qwen2.5-VL weights.

**Not established by the source.** Dynamic resolution, Qwen2.5-VL window
attention, synchronized absolute-time M-RoPE, two-frame video tubes và
coordinate serialization là các đặc tính của những đời VL trước. Chúng có thể
nằm trong lineage Qwen3.5, nhưng Qwen-VLA paper không cung cấp chain bằng chứng
để khẳng định từng module được giữ nguyên.

**Verified limitation.** Default Qwen-VLA không nhận proprioceptive robot state;
nó dựa vào camera, instruction và embodiment prompt. Paper chỉ thấy lợi ích nhỏ
khi thử thêm state trên RoboTwin-2.0. Action data vẫn nhỏ hơn VL data, joint
optimization có thể làm giảm nhẹ pure-VL/navigation performance, và đánh giá
hiện tại chủ yếu short-horizon. Memory, failure recovery, world modeling, force
và tactile feedback vẫn là future work. [Qwen-VLA, §5.2.4, pp.25–26; §7,][qwen-vla]

## Kết luận

Có thể mô tả ngắn gọn như sau:

```text
Qwen-VL  = visual perception + language grounding + token generation
Qwen-VLA = Qwen3.5 VLM backbone
           + embodiment-conditioned continuous action expert
           + trajectory data, action loss và closed-loop policy training
```

Phần được kế thừa là **cognitive stack**: visual encoder/fusion, pretrained
multimodal representation, language instruction interface, grounding, reasoning
và text objective. Phần tạo nên VLA là **motor stack**: real-valued action space,
DiT flow policy, embodiment-aware conditioning và progressive action training.

## Nguồn

- Wang et al. *Qwen2-VL: Enhancing Vision-Language Model's Perception of the
  World at Any Resolution*. arXiv:2409.12191, 2024. [Local PDF][qwen2-vl] ·
  [arXiv](https://arxiv.org/abs/2409.12191)
- Bai et al. *Qwen2.5-VL Technical Report*. arXiv:2502.13923, 2025.
  [Local PDF][qwen25-vl] · [arXiv](https://arxiv.org/abs/2502.13923)
- Qwen Team. *Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks,
  Environments, and Robot Embodiments*. arXiv:2605.30280v2, 2026.
  [Local PDF][qwen-vla] · [arXiv](https://arxiv.org/abs/2605.30280)

[qwen2-vl]: ../gwen-overview/qwen2_vl_2409.12191.pdf
[qwen25-vl]: ../gwen-overview/qwen2.5_vl_2502.13923.pdf
[qwen-vla]: ../vla-specific/qwen_vla_2605.30280.pdf
