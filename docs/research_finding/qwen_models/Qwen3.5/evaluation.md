# Qwen3.5 — Evaluation

> Điểm dưới đây là nhóm benchmark được model card/blog công bố hoặc nhắc tới. Khi trình bày phải gọi là **reported results**; không mặc nhiên coi là independent reproduction. Setting, prompt, thinking mode, tool access và số sample có thể ảnh hưởng lớn đến điểm.

## 1. Ma trận năng lực

| Nhóm              | Benchmark tiêu biểu                                                   | Đo lường                                                  |
| ------------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| Knowledge          | MMLU-Pro, MMLU-Redux, SuperGPQA, C-Eval                                 | Kiến thức đa lĩnh vực, English/Chinese, câu hỏi khó  |
| STEM/reasoning     | GPQA, HLE, AIME, HMMT, IMOAnswerBench                                   | Scientific reasoning, expert QA, competition math            |
| Code               | LiveCodeBench                                                           | Code trên bài tương đối mới                           |
| Coding agent       | SWE-bench Verified, SWE-bench Multilingual/Pro, Terminal-Bench, NL2Repo | Sửa repository, chạy test, terminal và tạo patch         |
| General agent/tool | TAU, Tool Decathlon, MCPMark, MCP-Atlas, WideSearch, DeepPlanning       | Chọn tool, arguments, planning, search, environment         |
| Vision-language    | MMMU, MMMU-Pro, MathVista, RealWorldQA, MMBench, SimpleVQA              | Visual reasoning, chart/diagram, VQA, real-world recognition |
| Hallucination      | HallusionBench                                                          | Multimodal hallucination                                     |

## 2. Coding agent khác code generation truyền thống

```text
Đọc repository → tìm bug → hiểu dependency → sửa nhiều file
      → chạy test → quan sát lỗi → lặp lại → xuất patch hợp lệ
```

SWE-bench/Terminal-Bench/NL2Repo đánh giá chuỗi hành động và khả năng hoàn thành task, không chỉ exact-match một function như HumanEval.

## 3. Vision-language evaluation

```text
Image / chart / document / real scene
                ↓
          visual understanding
                ↓
      answer / math / explanation
```

MMMU và MMMU-Pro thiên về multimodal academic reasoning; MathVista về visual math; RealWorldQA về nhận biết thế giới thực; HallusionBench kiểm tra xu hướng bịa thông tin từ hình ảnh.

## 4. Qwen3.6-35B-A3B — bảng số liệu được công bố

Các số sau là **reported results của Qwen3.6-35B-A3B**, hữu ích để minh họa hướng nâng cấp agent/coding của dòng sau:

| Nhóm           | Benchmark          | Score |
| --------------- | ------------------ | ----: |
| Coding agent    | SWE-bench Verified |  73.4 |
| Coding agent    | Terminal-Bench 2.0 |  51.5 |
| Coding agent    | NL2Repo            |  29.4 |
| General agent   | DeepPlanning       |  25.9 |
| Tool use        | MCPMark            |  37.0 |
| Knowledge       | MMLU-Pro           |  85.2 |
| STEM            | GPQA               |  86.0 |
| Code            | LiveCodeBench v6   |  80.4 |
| Math            | AIME26             |  92.7 |
| Multimodal      | MMMU               |  81.7 |
| Multimodal math | MathVista mini     |  86.4 |
| Visual QA       | RealWorldQA        |  85.3 |
| Hallucination   | HallusionBench     |  69.8 |

Không nên trộn các điểm này với Qwen3.5 khi so sánh nếu checkpoint, prompt và evaluation protocol khác nhau.

## 5. Cách đọc kết quả khi thuyết trình

- Xác định model/checkpoint và phiên bản benchmark.
- Ghi rõ thinking enabled/disabled, tool access và agent scaffold.
- Phân biệt zero-shot, few-shot, self-consistency và multi-turn.
- Ghi “reported by Qwen” nếu chưa có reproduction độc lập.
- Không suy ra training dataset từ việc một benchmark xuất hiện trong model card.

![TODO: bản đồ benchmark theo capability](Image/qwen35_evaluation_map.png)
![TODO: coding-agent evaluation loop](Image/qwen35_coding_agent_eval.png)
