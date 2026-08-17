# Qwen3.5 — Đánh giá

> Các benchmark dưới đây là những benchmark được model card/blog của Qwen công bố hoặc nhắc tới. Khi trình bày cần ghi rõ **reported results**; không mặc nhiên coi đây là independent reproduction. Điểm số có thể thay đổi theo checkpoint, prompt, thinking mode, tool access, agent scaffold và số sample.

## Nguồn xem kết quả và mã đánh giá

Các link chính thức để kiểm tra model card, bảng điểm và mã triển khai:

- [Qwen3.5 trên Hugging Face](https://huggingface.co/Qwen/models): danh sách checkpoint và model card chính thức.
- [Qwen3.5-35B-A3B trên Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B): checkpoint MoE 35B, khoảng 3B tham số được kích hoạt mỗi token.
- [Qwen3.5-35B-A3B-FP8 trên Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8): bản FP8 của checkpoint MoE để xem config và cách triển khai.
- [Qwen3.5-9B trên Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B): model card của checkpoint dense nhỏ.
- [Qwen3.5-397B-A17B-FP8 trên Hugging Face](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8): model card có bảng Benchmark Results của checkpoint lớn.
- [Qwen3.5 trên GitHub](https://github.com/QwenLM/Qwen3.5): mã nguồn, hướng dẫn chạy và các thảo luận về evaluation.
- [Qwen-Agent trên GitHub](https://github.com/QwenLM/Qwen-Agent): framework agent, tool calling, MCP và tài liệu DeepPlanning.
- [Qwen3.5 model collection](https://huggingface.co/collections/Qwen/qwen35): collection các checkpoint Qwen3.5 trên Hugging Face.

Trong các model card Hugging Face, tìm mục **Benchmark Results** để xem điểm theo đúng checkpoint. Không nên lấy điểm từ một checkpoint rồi gán cho toàn bộ dòng Qwen3.5.

## 1. Ma trận phân nhóm năng lực

| Nhóm năng lực           | Benchmark tiêu biểu                                                   | Năng lực chính được kiểm tra                                              |
| -------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Knowledge                  | MMLU-Pro, MMLU-Redux, SuperGPQA, C-Eval                                 | Kiến thức đa lĩnh vực, kiến thức chuyên gia và kiến thức tiếng Trung |
| STEM và reasoning         | GPQA, HLE, AIME, HMMT, IMOAnswerBench                                   | Scientific reasoning, câu hỏi chuyên gia và toán thi đấu                  |
| Code generation            | LiveCodeBench                                                           | Sinh code đúng trên các bài lập trình mới                                |
| Coding agent               | SWE-bench Verified, SWE-bench Multilingual/Pro, Terminal-Bench, NL2Repo | Đọc repository, sửa code, chạy test và hoàn thành task bằng terminal     |
| General agent và tool use | TAU-bench, Tool Decathlon, MCPMark, MCP-Atlas                           | Chọn tool, điền arguments, phối hợp nhiều tool và hoàn thành workflow   |
| Search và planning        | WideSearch, DeepPlanning                                                | Tìm kiếm, tổng hợp evidence và lập kế hoạch nhiều bước                |
| Vision-language            | MMMU, MMMU-Pro, MathVista, RealWorldQA, MMBench, SimpleVQA              | Hiểu ảnh, biểu đồ, tài liệu, visual math và câu hỏi thế giới thực   |
| Multimodal hallucination   | HallusionBench                                                          | Phát hiện và hạn chế việc bịa thông tin không có trong ảnh            |

## 2. Benchmark chi tiết

| Benchmark                                                      | Đánh giá gì?                                                    | Cách tính chính                                                                      |
| -------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)            | Kiến thức và reasoning đa lĩnh vực                            | Accuracy trên câu hỏi trắc nghiệm khó thuộc nhiều môn                          |
| MMLU-Redux                                                     | Kiểm tra lại năng lực MMLU với bộ câu hỏi được rà soát | Tỷ lệ câu trả lời đúng                                                           |
| SuperGPQA                                                      | Kiến thức chuyên gia và reasoning chuyên ngành                | Accuracy trên câu hỏi trắc nghiệm cấp chuyên gia                                 |
| C-Eval                                                         | Kiến thức học thuật và chuyên môn bằng tiếng Trung         | Accuracy trên các môn học tiếng Trung                                              |
| [GPQA](https://github.com/idavidrein/gpqa)                      | Scientific reasoning cấp chuyên gia                               | Tỷ lệ trả lời đúng câu hỏi vật lý, hóa học và sinh học khó               |
| HLE                                                            | Câu hỏi cực khó liên ngành                                    | Exact-match hoặc judged correctness tùy task                                          |
| AIME                                                           | Toán thi đấu cấp cao                                            | Tỷ lệ bài có đáp án số nguyên đúng                                           |
| HMMT                                                           | Toán thi đấu nâng cao                                           | Accuracy hoặc điểm trung bình theo bài toán                                       |
| IMOAnswerBench                                                 | Giải toán theo phong cách Olympic                                | So sánh đáp án cuối với đáp án chuẩn, có thể dùng evaluator cho lời giải |
| [LiveCodeBench](https://github.com/livecodebench/livecodebench) | Sinh code trên bài tương đối mới                             | Tỷ lệ bài code pass toàn bộ test tự động                                        |
| [SWE-bench Verified](https://www.swebench.com/)                 | Sửa issue thực tế trong repository                               | Tỷ lệ patch làm toàn bộ test xác minh pass                                        |
| SWE-bench Multilingual/Pro                                     | Coding agent trên nhiều ngôn ngữ hoặc task khó hơn           | Success rate dựa trên test của repository                                            |
| Terminal-Bench                                                 | Thao tác terminal và môi trường máy tính                     | Tỷ lệ task đạt trạng thái cuối hoặc pass test của sandbox                      |
| NL2Repo                                                        | Tạo hoặc sửa repository từ mô tả tự nhiên                   | Đánh giá bằng build, test và tiêu chí repository của task                       |
| TAU-bench                                                      | Sử dụng tool theo policy trong hội thoại                        | Tỷ lệ hoàn thành task với tool và arguments đúng                                |
| Tool Decathlon                                                 | Sử dụng nhiều loại tool                                         | Đánh giá tool selection, argument correctness và task completion                    |
| MCPMark                                                        | Sử dụng MCP tools                                                 | Đánh giá chọn tool, tham số, chuỗi gọi tool và kết quả cuối                  |
| MCP-Atlas                                                      | Phối hợp nhiều MCP tool trong workflow                           | Tỷ lệ hoàn thành workflow nhiều bước                                             |
| WideSearch                                                     | Tìm kiếm và tổng hợp thông tin từ nhiều nguồn              | Answer correctness và độ đầy đủ của evidence                                    |
| DeepPlanning                                                   | Lập kế hoạch và thực hiện task nhiều bước                  | Đánh giá tính đúng của plan và success ở trạng thái cuối                    |
| [MMMU](https://github.com/MMMU-Benchmark/MMMU)                  | Multimodal reasoning ở cấp đại học                             | Accuracy trên câu hỏi ảnh, biểu đồ và sơ đồ                                  |
| [MMMU-Pro](https://github.com/tsinghua-lm/MMMU)                 | Multimodal academic reasoning khó hơn                             | Accuracy trên câu hỏi multimodal nhiều bước                                       |
| [MathVista](https://github.com/lupantech/MathVista)             | Toán dựa trên hình ảnh                                         | Tỷ lệ đáp án toán đúng sau khi đọc biểu đồ, hình học hoặc sơ đồ      |
| RealWorldQA                                                    | Hiểu hình ảnh trong thế giới thực                             | Accuracy trên câu hỏi về vật thể, không gian và tình huống                    |
| MMBench                                                        | Visual question answering tổng quát                               | Accuracy hoặc normalized accuracy trên câu hỏi trắc nghiệm ảnh                   |
| SimpleVQA                                                      | Visual question answering cơ bản                                  | Exact match hoặc judged semantic correctness                                           |
| [HallusionBench](https://github.com/FuxiaoLiu/LRV-Instruction)  | Hallucination khi hiểu hình ảnh                                  | Đo tỷ lệ phân biệt đúng thông tin có và không có trong ảnh                 |

## 3. Diễn giải các loại metric

- **Accuracy:** số câu trả lời đúng chia cho tổng số câu hỏi.
- **Exact match:** câu trả lời phải khớp đáp án chuẩn, thường dùng cho số, lựa chọn hoặc output có format cố định.
- **Pass rate / success rate:** tỷ lệ task hoàn thành hoàn toàn; thường được kiểm tra bằng unit test, verifier hoặc trạng thái cuối của environment.
- **LLM/judge-based score:** evaluator hoặc người chấm đánh giá độ đúng, chất lượng lời giải, evidence hoặc mức độ hoàn thành khi exact match không phù hợp.

Coding-agent benchmark khác code generation truyền thống:

```text
Đọc repository → tìm bug → hiểu dependency → sửa nhiều file
      → chạy test → quan sát lỗi → lặp lại → xuất patch hợp lệ
```

SWE-bench, Terminal-Bench và NL2Repo vì vậy đo khả năng hoàn thành chuỗi hành động, không chỉ đo việc sinh đúng một function như HumanEval.

Vision-language benchmark có luồng tổng quát:

```text
Image / chart / document / real scene
                ↓
          visual understanding
                ↓
      answer / math / explanation
```

MMMU và MMMU-Pro thiên về multimodal academic reasoning; MathVista tập trung vào visual math; RealWorldQA đo nhận biết thế giới thực; HallusionBench kiểm tra xu hướng bịa thông tin từ hình ảnh.

## 4. Qwen3.5-35B-A3B — kết quả được công bố

Đây là **reported results tiêu biểu của đúng checkpoint Qwen3.5-35B-A3B**, lấy từ [model card chính thức trên Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B). Phần 5 giữ toàn bộ score để tra cứu.

### Language, reasoning và agent

| Nhóm          | Benchmark          | Qwen3.5-35B-A3B |
| -------------- | ------------------ | --------------: |
| Kiến thức | MMLU-Pro |            85,3 |
| STEM/lý luận | HLE với CoT |            22.4 |
| STEM/lý luận | Kim cương GPQA |            84,2 |
| Mã hóa | Băng ghế dự bị SWE đã được xác minh |            69,2 |
| Mã hóa | Ghế Nhà Ga 2 |            40,5 |
| Mã hóa | LiveCodeBench v6 |            74,6 |
| Tổng đại lý | TAU2-Băng ghế |            81,2 |
| Tổng đại lý | Lập kế hoạch sâu |            22.8 |

### Thị giác-ngôn ngữ

| Nhóm         | Benchmark                  | Qwen3.5-35B-A3B |
| ------------- | -------------------------- | --------------: |
| STEM/câu đố | MMMU |            81,4 |
| STEM/câu đố | MathVista mini |            86,2 |
| Ảo giác | HallusionBench |            67,9 |
| Tài liệu/OCR | OCBench |            91.0 |
| Không gian | CountBench |            97,8 |
| Video | VideoMME có phụ đề |            86,6 |
| Video | VideoMME không có phụ đề |            82,5 |

Ghi chú: `TIR-Bench` báo cáo điểm khi bật/tắt Code Interpreter; `BabyVision` và `V*` cũng dùng định dạng with-CI/without-CI trong model card. `--` nghĩa là chưa có điểm hoặc không áp dụng.

## 5. Phụ lục: toàn bộ score từ model card Qwen3.5-35B-A3B

### Ngôn ngữ

| Nhóm                 | Benchmark          | Score |
| --------------------- | ------------------ | ----: |
| Kiến thức | MMLU-Pro |  85,3 |
| Kiến thức | MMLU-Redux |  93,3 |
| Kiến thức | Đánh giá C |  90,2 |
| Kiến thức | SuperGPQA |  63,4 |
| Hướng dẫn sau | IFEgiá trị |  91,9 |
| Hướng dẫn sau | IFBench |  70,2 |
| Hướng dẫn sau | Đa thử thách |  60,0 |
| Bối cảnh dài | AA-LCR |  58,5 |
| Bối cảnh dài | LongBench v2 |  59,0 |
| STEM & lý luận | HLE với CoT |  22.4 |
| STEM & lý luận | Kim cương GPQA |  84,2 |
| STEM & lý luận | HMMT 25/02 |  89,0 |
| STEM & lý luận | HMMT 25/11 |  89,2 |
| Mã hóa | Băng ghế dự bị SWE đã được xác minh |  69,2 |
| Mã hóa | Ghế Nhà Ga 2 |  40,5 |
| Mã hóa | LiveCodeBench v6 |  74,6 |
| Mã hóa | Lực Lượng Mã |  2028 |
| Mã hóa | OJBench |  36.0 |
| Mã hóa | FullStackBench vi |  58,1 |
| Mã hóa | FullStackBench zh |  55,0 |
| Tổng đại lý | BFCL-V4 |  67,3 |
| Tổng đại lý | TAU2-Băng ghế |  81,2 |
| Tổng đại lý | Ghế VITA |  31.9 |
| Tổng đại lý | Lập kế hoạch sâu |  22.8 |
| Đại lý tìm kiếm | HLE với công cụ |  47,4 |
| Đại lý tìm kiếm | DuyệtComp |  61.0 |
| Đại lý tìm kiếm | DuyệtComp-zh |  69,5 |
| Đại lý tìm kiếm | Tìm kiếm rộng rãi |  57,1 |
| Đại lý tìm kiếm | Con dấu-0 |  41,4 |
| Đa ngôn ngữ | MMMLU |  85,2 |
| Đa ngôn ngữ | MMLU-ProX |  81.0 |
| Đa ngôn ngữ | NOVA-63 |  57,1 |
| Đa ngôn ngữ | BAO GỒM |  79,7 |
| Đa ngôn ngữ | PIQA toàn cầu |  86,6 |
| Đa ngôn ngữ | PolyMATH |  64,4 |
| Đa ngôn ngữ | WMT24++ |  76,3 |
| Đa ngôn ngữ | TUYỆT VỜI |  86,6 |

### Tầm nhìn-Ngôn ngữ

| Nhóm                  | Benchmark                    |       Score |
| ---------------------- | ---------------------------- | ----------: |
| STEM và câu đố | MMMU |        81,4 |
| STEM và câu đố | MMMU-Pro |        75,1 |
| STEM và câu đố | MathVision |        83,9 |
| STEM và câu đố | MathVista mini |        86,2 |
| STEM và câu đố | DynaMath |        85,0 |
| STEM và câu đố | ZEROBench |           8 |
| STEM và câu đố | ZEROBench-sub |        34.1 |
| STEM và câu đố | VlmsAreBlind |        97,0 |
| STEM và câu đố | BabyVision | 38,4 / 29,6 |
| VQA tổng hợp | RealWorldQA |        84.1 |
| VQA tổng hợp | MMStar |        81,9 |
| VQA tổng hợp | MMBenchEN-DEV-v1.1 |        91,5 |
| VQA tổng hợp | Đơn giảnVQA |        58,3 |
| VQA tổng hợp | HallusionBench |        67,9 |
| Hiểu tài liệu | OmniDocBench1.5 |        89,3 |
| Hiểu tài liệu | CharXiv (RQ) |        77,5 |
| Hiểu tài liệu | MMLongBench-Doc |        59,5 |
| Hiểu tài liệu | CC-OCR |        80,7 |
| Hiểu tài liệu | AI2D_TEST |        92,6 |
| Hiểu tài liệu | OCBench |        91.0 |
| Trí tuệ không gian | ERQA |        64,8 |
| Trí tuệ không gian | CountBench |        97,8 |
| Trí tuệ không gian | RefCOCO (trung bình) |        89,2 |
| Trí tuệ không gian | ODInW13 |        42,6 |
| Trí tuệ không gian | EmbSpatialBench |        83,1 |
| Trí tuệ không gian | RefSpatialBench |        63,5 |
| Trí tuệ không gian | LingoQA |        79,2 |
| Trí tuệ không gian | Hypersim |        13.1 |
| Trí tuệ không gian | SUNRGBD |        33,4 |
| Trí tuệ không gian | Nuscene |        14.6 |
| Hiểu video | VideoMME (có phụ đề) |        86,6 |
| Hiểu video | VideoMME (không có phụ đề) |        82,5 |
| Hiểu video | VideoMMMU |        80,4 |
| Hiểu video | MLVU |        85,6 |
| Hiểu video | MVBench |        74,8 |
| Hiểu video | LVBench |        71,4 |
| Hiểu video | MMVU |        72,3 |
| Agent trực quan | ScreenSpot Pro |        68,6 |
| Agent trực quan | OSWorld-Đã được xác minh |        54,5 |
| Agent trực quan | Thế giới Android |        71.1 |
| Công cụ gọi | TIR-Băng ghế dự bị | 55,5 / 38,0 |
| Công cụ gọi | V* | 92,7 / 89,5 |
| VQA y tế | NGỌT |        78,7 |
| VQA y tế | PMC-VQA |        62.0 |
| VQA y tế | MedXpertQA-MM |        61,4 |

Các score dạng `x / y` là kết quả **with Code Interpreter / without Code Interpreter** theo model card. `--` trong bảng gốc nghĩa là chưa có score hoặc không áp dụng.

## 6. Đánh giá kết quả Qwen3.5-35B-A3B

> “Qwen3.5-35B-A3B cho thấy năng lực khá cân bằng giữa language, coding agent và vision-language. Ở knowledge, model đạt 85.3 trên MMLU-Pro, 93.3 trên MMLU-Redux và 90.2 trên C-Eval. Điều này cho thấy model có nền tảng kiến thức tốt, nhưng SuperGPQA 63.4 và HLE with CoT 22.4 cho thấy các câu hỏi chuyên gia và bài toán cực khó vẫn là thách thức.”

> “Về coding, model đạt 69.2 trên SWE-bench Verified, 40.5 trên Terminal Bench 2 và 74.6 trên LiveCodeBench v6. Các benchmark này cho thấy model không chỉ sinh code, mà còn có khả năng sửa repository, thao tác terminal và giải bài code mới. Tuy nhiên kết quả agent phụ thuộc vào scaffold, tool, số lượt và môi trường đánh giá.”

> “Ở agent và search, TAU2-Bench đạt 81.2, BFCL-V4 đạt 67.3, DeepPlanning đạt 22.8 và BrowseComp-zh đạt 69.5. Model có khả năng gọi tool và xử lý hội thoại theo policy khá tốt, nhưng lập kế hoạch dài và duy trì nhiều bước vẫn khó hơn các task tool calling đơn lẻ.”

> “Về vision-language, model đạt 81.4 trên MMMU, 75.1 trên MMMU-Pro và 86.2 trên MathVista mini. Các kết quả OCR và document cũng tốt, với OmniDocBench1.5 đạt 89.3, AI2D đạt 92.6 và OCRBench đạt 91.0. CountBench đạt 97.8, cho thấy khả năng đếm đối tượng trong ảnh mạnh.”

> “Với video, VideoMME đạt 86.6 khi có subtitle và 82.5 khi không có subtitle. Chênh lệch này cho thấy text phụ trợ vẫn đóng vai trò quan trọng trong video understanding. HallusionBench đạt 67.9, nghĩa là model có khả năng hạn chế hallucination nhưng vẫn còn bịa thông tin trong một số trường hợp.”

> “Tổng kết, Qwen3.5-35B-A3B là một model MoE có năng lực đa phương thức tốt với chi phí activation khoảng 3B tham số mỗi token. Điểm mạnh nằm ở instruction following, tool calling, coding, OCR, visual counting và video understanding. Điểm hạn chế là reasoning cực khó, long-context retrieval và các workflow agent nhiều bước.”

## 7. Quy tắc trình bày kết quả

- Luôn xác định model/checkpoint và phiên bản benchmark.
- Ghi rõ thinking enabled/disabled, tool access và agent scaffold.
- Phân biệt zero-shot, few-shot, self-consistency và multi-turn.
- Dùng cụm “reported by Qwen” nếu chưa có reproduction độc lập.
- Không suy ra training dataset từ việc một benchmark xuất hiện trong model card.
- Bảng điểm ở trên là của đúng Qwen3.5-35B-A3B; không trộn với Qwen3.6 hoặc checkpoint khác.

![TODO: bản đồ benchmark theo capability](Image/qwen35_evaluation_map.png)
![TODO: vòng đánh giá coding agent](Image/qwen35_coding_agent_eval.png)
