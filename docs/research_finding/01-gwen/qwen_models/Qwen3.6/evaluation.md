# Qwen3.6-35B-A3B — Evaluation

> Các điểm dưới đây là **reported results** từ model card chính thức. Cần phân biệt benchmark public, benchmark có scaffold/judge riêng và benchmark nội bộ của Qwen.

## 1. Bảng phân nhóm năng lực

| Nhóm năng lực | Benchmark | Năng lực chính |
|---|---|---|
| Coding agent | SWE-bench Verified/Multilingual/Pro, Terminal-Bench 2.0, SkillsBench, NL2Repo | Sửa repository, chạy test, thao tác terminal và thực hiện thay đổi nhiều file |
| Frontend agent | QwenWebBench | Tạo web app, game, SVG, visualization, animation và 3D |
| Agent/tool use | MCPMark, MCP-Atlas, Tool Decathlon, BFCL, TAU2-Bench | Chọn tool, điền arguments, xử lý observation và hoàn thành workflow |
| Planning | DeepPlanning | Lập kế hoạch và duy trì mục tiêu qua nhiều bước |
| Search agent | WideSearch, BrowseComp, BrowseComp-zh, HLE with tool, Seal-0 | Tìm kiếm, browse và tổng hợp evidence |
| Knowledge | MMLU-Pro, MMLU-Redux, SuperGPQA, C-Eval | Kiến thức tổng quát, chuyên gia và khoa học chuyên sâu |
| STEM/reasoning | GPQA, HLE, HLE-Verified, HMMT, IMOAnswerBench, AIME26 | Scientific reasoning và toán thi đấu |
| Code generation | LiveCodeBench v6, SecCodeBench | Sinh code mới và code security |
| Vision-language | MMMU, MMMU-Pro, MathVista, RealWorldQA, MMBench, SimpleVQA | Multimodal reasoning và visual question answering |
| Hallucination | HallusionBench | Nhận biết thông tin có hoặc không có trong ảnh |
| Document/OCR | OmniDocBench1.5, CharXiv, MMLongBench-Doc, CC-OCR, AI2D, OCRBench | Đọc tài liệu, biểu đồ, OCR và document understanding |
| Video | VideoMME, VideoMMMU, MLVU, MVBench, LVBench, MMVU | Hiểu nội dung và temporal reasoning trong video |
| Spatial/visual agent | RefCOCO, ODInW13, CountBench, ScreenSpot Pro, OSWorld-Verified, AndroidWorld | Grounding, đếm, nhận biết vị trí và thao tác giao diện |

## 2. Benchmark chi tiết và cách tính

| Benchmark | Đánh giá cụ thể | Cách tính chính |
|---|---|---|
| SWE-bench Verified | Sửa issue thực tế trong repository | Tỷ lệ patch pass toàn bộ test xác minh |
| SWE-bench Multilingual/Pro | Sửa repository đa ngôn ngữ hoặc task khó hơn | Success rate theo test của repository |
| Terminal-Bench 2.0 | Hoàn thành task trong môi trường terminal | Tỷ lệ task đạt trạng thái cuối hoặc pass verifier/test |
| SkillsBench | Các kỹ năng coding thực tế, self-contained | Điểm trung bình tỷ lệ task hoàn thành qua test |
| NL2Repo | Tạo/sửa repository từ mô tả tự nhiên | Build, unit test và tiêu chí task quyết định success |
| QwenWebBench | Tạo frontend, game, SVG, visualization, animation và 3D | Auto-render kết hợp multimodal judge chấm output |
| MCPMark | Sử dụng MCP server/tool | Đánh giá tool selection, arguments, execution và task success |
| MCP-Atlas | Workflow phối hợp nhiều MCP tool | Tỷ lệ hoàn thành workflow nhiều bước |
| Tool Decathlon | Chọn và sử dụng nhiều loại tool | Đúng tool, đúng tham số và đạt kết quả cuối |
| BFCL-V4 | Function calling | Tỷ lệ gọi đúng function và arguments theo ground truth |
| TAU2-Bench | Tool use trong hội thoại theo policy | Tỷ lệ hoàn thành task trong môi trường mô phỏng |
| DeepPlanning | Lập kế hoạch dài và giữ mục tiêu | Điểm dựa trên chất lượng plan và trạng thái hoàn thành |
| WideSearch | Search và tổng hợp nhiều nguồn | Độ đúng câu trả lời và độ đầy đủ của evidence |
| BrowseComp/BrowseComp-zh | Tìm kiếm câu trả lời khó bằng web | Answer correctness sau chuỗi browse/tool calls |
| MMLU-Pro | Kiến thức và reasoning đa lĩnh vực | Accuracy trên câu hỏi trắc nghiệm khó |
| MMLU-Redux | Bộ MMLU được rà soát lại | Tỷ lệ câu trả lời đúng |
| SuperGPQA | Kiến thức chuyên gia | Accuracy trên câu hỏi cấp chuyên gia |
| C-Eval | Kiến thức học thuật tiếng Trung | Accuracy trên nhiều môn học |
| GPQA | Scientific reasoning cấp chuyên gia | Tỷ lệ câu trả lời đúng |
| HLE | Câu hỏi cực khó liên ngành | Exact-match hoặc judged correctness |
| HLE-Verified | Phiên bản HLE được xác minh chi tiết | Đúng theo verification protocol và error taxonomy |
| LiveCodeBench v6 | Sinh code trên bài tương đối mới | Tỷ lệ code pass test tự động |
| SecCodeBench | Sinh hoặc sửa code có yêu cầu security | Pass test chức năng và/hoặc security checks |
| HMMT | Toán thi đấu nâng cao | Accuracy hoặc điểm trung bình theo bài |
| IMOAnswerBench | Toán phong cách Olympic | So sánh đáp án cuối với đáp án chuẩn |
| AIME26 | Toán thi đấu có đáp án số nguyên | Tỷ lệ bài có đáp án đúng |
| MMMU | Multimodal reasoning cấp đại học | Accuracy trên ảnh, biểu đồ và sơ đồ |
| MMMU-Pro | Multimodal reasoning khó hơn | Accuracy trên câu hỏi nhiều bước |
| MathVista | Visual mathematics | Tỷ lệ đáp án toán đúng từ hình ảnh/biểu đồ |
| RealWorldQA | Hiểu cảnh và vật thể thế giới thực | Accuracy trên visual QA |
| MMBench | Visual question answering tổng quát | Accuracy hoặc normalized accuracy |
| SimpleVQA | VQA cơ bản | Exact match hoặc semantic judged correctness |
| HallusionBench | Hallucination trong visual understanding | Tỷ lệ phân biệt đúng thông tin có/không có trong ảnh |
| OmniDocBench1.5 | Hiểu tài liệu đa dạng | Điểm tổng hợp layout, OCR và nội dung |
| OCRBench/CC-OCR | Nhận dạng chữ trong ảnh/tài liệu | Accuracy hoặc normalized OCR score |
| AI2D | Hiểu sơ đồ khoa học | Accuracy trên câu hỏi về diagram |
| VideoMME | Video understanding có/không subtitle | Accuracy trên câu hỏi video |
| VideoMMMU | Multimodal reasoning trên video | Accuracy |
| MLVU/MVBench/LVBench/MMVU | Temporal và long-video understanding | Accuracy trên các task video |
| RefCOCO | Grounding object theo referring expression | Tỷ lệ bounding box đạt IoU threshold |
| CountBench | Đếm đối tượng trong ảnh | Accuracy của số lượng dự đoán |
| ScreenSpot Pro | Grounding mục tiêu trên màn hình | Tỷ lệ click/box đúng vị trí |
| OSWorld-Verified | Thao tác máy tính GUI | Tỷ lệ task đạt trạng thái cuối qua verifier |
| AndroidWorld | Thao tác ứng dụng Android | Tỷ lệ task hoàn thành trong emulator |

## 3. Kết quả so sánh Qwen3.5 → Qwen3.6

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

## 4. Nhận xét

Qwen3.6 cải thiện mạnh nhất ở **agentic coding và tool use**. Terminal-Bench tăng 11.0 điểm, MCPMark tăng 10.0 điểm, SkillsBench tăng 24.3 điểm, còn SWE-bench Multilingual tăng 6.9 điểm. Điều này phù hợp với mục tiêu của Qwen3.6: repository reasoning, frontend workflow, terminal và multi-step agent.

Ngược lại, MMLU-Pro gần như không đổi, HLE giảm nhẹ, còn MMMU và MathVista chỉ tăng khoảng 0.2–0.3 điểm. Vì vậy Qwen3.6 nên được hiểu là một **targeted post-training update** trên nền kiến trúc Qwen3.5, không phải một bước nhảy lớn về backbone hoặc năng lực multimodal tổng quát.

Các điểm agent không chỉ đo model thuần túy. SWE-bench, Terminal-Bench và MCPMark còn phụ thuộc vào agent scaffold, tool, số lượt, context management, verifier và environment. QwenWebBench sử dụng auto-render kết hợp multimodal judge; đây là benchmark nội bộ, nên cần thận trọng khi so sánh với benchmark public.

## 5. Cảnh báo về tính so sánh

- SWE-bench dùng internal agent scaffold, bash/file-edit tools, context lớn và sampling cụ thể.
- Terminal-Bench dùng Harbor/Terminus-2, timeout 3 giờ và trung bình 5 runs.
- SkillsBench dùng 78 self-contained tasks, loại API-dependent tasks và báo cáo trung bình 5 runs.
- QwenWebBench là benchmark nội bộ, dùng auto-render và multimodal judge.
- Raw delta chỉ có ý nghĩa trong cùng benchmark; không cộng hoặc xếp hạng trực tiếp các delta khác metric.
- Chưa có independent reproduction đầy đủ cho toàn bộ bảng điểm Qwen3.6.

## 6. Nguồn và mã đánh giá

- [Qwen3.6 official repository](https://github.com/QwenLM/Qwen3.6)
- [Qwen3.6 model collection](https://huggingface.co/collections/Qwen/qwen36)
- [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen-Agent: agent framework, MCP và tool use](https://github.com/QwenLM/Qwen-Agent)
- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [LiveCodeBench](https://github.com/livecodebench/livecodebench)
- [GPQA](https://github.com/idavidrein/gpqa)
- [MMMU](https://github.com/MMMU-Benchmark/MMMU)
- [MathVista](https://github.com/lupantech/MathVista)
- [HallusionBench](https://github.com/FuxiaoLiu/LRV-Instruction)
- [HLE-Verified dataset](https://huggingface.co/datasets/skylenage/HLE-Verified)
- [Terminal-Bench](https://github.com/terminal-bench/terminal-bench)
- [SkillsBench](https://github.com/benchflow-ai/skillsbench)
- [NL2Repo](https://github.com/epoch-research/NL2Repo)

Các model card Hugging Face là nguồn để xem score cụ thể của từng checkpoint. Các repository benchmark là nguồn để xem dataset, evaluator, test harness hoặc metric; score giữa các nguồn chỉ nên so sánh khi protocol giống nhau.
