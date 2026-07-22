# Qwen3.5 — Training stages

> Qwen công khai pipeline ở mức khái quát, không công khai toàn bộ recipe. Bảng dưới đây phân biệt `✓` xác nhận, `△` diễn giải hợp lý từ công bố, `?` chưa công bố.

| Giai đoạn             | Nội dung                                                                | Trạng thái |
| ----------------------- | ------------------------------------------------------------------------ | ------------ |
| Foundation pre-training | Text, code, math, multilingual và multimodal next-token prediction      | ✓/△        |
| Early-fusion multimodal | Image-text, video-text, documents, chart/OCR/UI cùng chuỗi token       | ✓           |
| Long-context            | Continued training cho tài liệu, video, repository và trajectory dài | △           |
| SFT                     | Instruction, hội thoại, reasoning, code, multimodal và tool-use       | ✓/△        |
| Reasoning RL            | Rollout, verifier/environment reward, policy optimization                | ✓/△        |
| Agentic post-training   | Multi-step plan → tool call → observation → answer                    | ✓/△        |
| Specialized behavior    | Thinking/non-thinking, format, multilingual, tool reliability            | △           |

## 1. Foundation và multimodal pre-training

```text
Text / code / math / multilingual
Image-text / video-text / documents
                ↓
Unified token sequence (text + visual tokens)
                ↓
            causal LM loss
                ↓
       Qwen3.5 foundation/base model
```

Objective cơ bản là `L = -Σ log p(x_t | x_<t)`. Qwen nhấn mạnh early-fusion multimodal training và hiệu suất gần text-only training. Chưa có số chính thức đầy đủ về tổng token, tỷ lệ modality, curriculum, resolution và video hours.

## 2. Long-context continued training

Pipeline có thể trình bày như sau:

```text
Short-context foundation model
              ↓
Long-context continued pre-training
              ↓
Long-document / video / repository / trajectory examples
              ↓
Long-context instruction tuning
```

Đây là diễn giải từ năng lực công bố, không phải recipe từng bước đã được Qwen xác nhận. Cần phân biệt model nhận được 256K/1M input với việc model luôn truy xuất và suy luận tốt ở độ dài đó.

## 3. Supervised post-training

```text
Base model
   ↓
Instruction-response + reasoning traces
   ↓
Multimodal conversations + code tasks
   ↓
Tool-use trajectories
   ↓
Supervised fine-tuning → useful assistant/agent
```

Loss thường vẫn là token-level cross-entropy, có thể chỉ tính trên response (`L_SFT = -Σ_{t∈response} log p(y_t | x,y_<t)`). Dataset, tỷ lệ trộn và masking chính xác chưa được công bố đầy đủ.

## 4. Reasoning RL và agentic training

```Shell
Prompt / environment state
          ↓
Model rollout: reasoning → action/tool call → observation
          ↓
Verifier / environment / judge
          ↓
Reward
          ↓
Sequence-level policy optimization
          ↓
Model tạo trajectory ổn định hơn
```

Reward có thể liên quan đến unit tests, mathematical checkers, tool success, browser completion, format hoặc environment state. Qwen nói đến hàng triệu agent environments và task distribution tăng dần độ phức tạp. Tuy nhiên tên thuật toán, reward function, số rollout và toàn bộ environment suite cho từng checkpoint chưa được công bố.

GSPO từng được Qwen giới thiệu cho RL sequence-level, nhưng không được tự động kết luận rằng mọi checkpoint Qwen3.5 đều dùng đúng cùng recipe nếu model card không xác nhận.

## 5. Agent loop sau post-training

```text
User request → plan → tool call → observation
                  ↑                 │
                  └── update plan ──┘
                                  ↓
                             final answer
```

Điều này giải thích vì sao evaluation nhấn mạnh coding agent, browser agent, repository task, planning và tool use hơn là chỉ hỏi–đáp một lượt.

## 6. Những gì không thể kết luận

- Không có corpus đầy đủ, số token và mixing ratio.
- Không có filtering/deduplication pipeline hoàn chỉnh.
- Không có optimizer, schedule, global batch và compute budget cho mọi model.
- Không có danh sách đầy đủ reward model, environment và reward function.

![TODO: pipeline training stages](Image/qwen35_training_pipeline.png)
![TODO: agentic RL loop](Image/qwen35_agentic_rl_loop.png)
