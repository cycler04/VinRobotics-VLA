# Qwen3.6-35B-A3B — Evaluation

> Các điểm dưới đây là **reported results** từ model card. Cần phân biệt benchmark public, benchmark có scaffold/judge riêng và benchmark nội bộ của Qwen.

## 1. Benchmark dùng để đo gì?

| Nhóm | Benchmark | Ý nghĩa |
|---|---|---|
| Coding agent | SWE-bench Verified/Multilingual/Pro | Sửa issue trong repository và vượt tests |
| Coding agent | Terminal-Bench 2.0 | Hoàn thành task qua terminal |
| Coding agent | SkillsBench, NL2Repo | Kỹ năng thực tế và thay đổi cấp repository |
| Frontend agent | QwenWebBench | Web app, game, SVG, visualization, animation, 3D; auto-render + multimodal judge |
| Agent/tool | MCPMark, MCP-Atlas, Tool Decathlon | Chọn tool, arguments và xử lý observation |
| Planning | DeepPlanning | Giữ mục tiêu qua kế hoạch nhiều bước |
| Search | WideSearch | Search, browse và tổng hợp kết quả |
| Knowledge | MMLU-Pro, MMLU-Redux, SuperGPQA, C-Eval | Kiến thức tổng quát và khoa học chuyên sâu |
| Reasoning/code | GPQA, HLE, LiveCodeBench, HMMT, IMOAnswerBench, AIME26 | Science, toán và code mới |
| Vision | MMMU, MMMU-Pro, MathVista, RealWorldQA, MMBench, SimpleVQA, HallusionBench | Multimodal reasoning, VQA và hallucination |
| Document/video/spatial | OmniDocBench, OCRBench, VideoMME, VideoMMMU, MLVU, RefCOCO, ODInW13 | OCR, video và spatial intelligence |

## 2. Qwen3.5 → Qwen3.6: kết quả chính

| Benchmark | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Delta |
|---|---:|---:|---:|
| SWE-bench Verified | 70.0 | 73.4 | +3.4 |
| SWE-bench Multilingual | 60.3 | 67.2 | +6.9 |
| SWE-bench Pro | 44.6 | 49.5 | +4.9 |
| Terminal-Bench 2.0 | 40.5 | 51.5 | +11.0 |
| MCPMark | 27.0 | 37.0 | +10.0 |
| SkillsBench Avg5 | 4.4 | 28.7 | +24.3 |
| NL2Repo | 20.5 | 29.4 | +8.9 |
| DeepPlanning | 22.8 | 25.9 | +3.1 |
| MMLU-Pro | 85.3 | 85.2 | -0.1 |
| GPQA | 84.2 | 86.0 | +1.8 |
| HLE | 22.4 | 21.4 | -1.0 |
| LiveCodeBench v6 | 74.6 | 80.4 | +5.8 |
| AIME26 | 91.0 | 92.7 | +1.7 |
| MMMU | 81.4 | 81.7 | +0.3 |
| MathVista mini | 86.2 | 86.4 | +0.2 |
| RealWorldQA | 84.1 | 85.3 | +1.2 |
| HallusionBench | 67.9 | 69.8 | +1.9 |

## 3. Interpretation

Kết quả cho thấy cải thiện lớn nhất nằm ở agentic coding và tool use: Terminal-Bench, MCPMark, SkillsBench, NL2Repo và SWE-bench đều tăng. Knowledge gần như phẳng; HLE giảm nhẹ; vision-language tăng nhẹ. Đây là pattern phù hợp với nhận định Qwen3.6 là targeted post-training update thay vì architectural leap.

## 4. Cảnh báo về tính so sánh

- SWE-bench dùng internal agent scaffold, bash/file-edit tools, 200K context và sampling cụ thể.
- Terminal-Bench dùng Harbor/Terminus-2, timeout 3 giờ, trung bình 5 runs.
- SkillsBench chỉ dùng 78 self-contained tasks, loại API-dependent tasks, trung bình 5 runs.
- QwenClawBench và QwenWebBench là benchmark nội bộ; QwenWebBench dùng auto-render và multimodal judge.
- Một số benchmark dùng judge bên ngoài hoặc scaffold riêng; không so sánh score nếu protocol khác nhau.
- Không có paper peer-reviewed độc lập cho Qwen3.6; các số liệu chưa phải independent reproduction.

## 5. Kết luận so sánh

Qwen3.6 mạnh hơn Qwen3.5 rõ nhất ở coding agent, frontend workflow, repository-level reasoning, tool calling và multi-turn planning. Nó không chứng minh một backbone mới; kiến trúc gần như giữ nguyên, còn phần thay đổi chính là post-training, agent scaffold và Thinking Preservation.
