# Các metric dùng để đánh giá mô hình Qwen

> **Câu hỏi:** Mỗi điểm số trong đánh giá ngôn ngữ, thị giác-ngôn ngữ và tác tử
> của Qwen có ý nghĩa gì?
>
> **Phạm vi:** Chỉ các hàm tính điểm. Nội dung, quy mô, cách chia và giấy phép
> bộ dữ liệu nằm trong [datasets.md](datasets.md); prompt, tiền xử lý, phiên bản
> bộ đánh giá và bảng kết quả Qwen nằm trong [benchmarks.md](benchmarks.md);
> mục tiêu huấn luyện nằm trong [loss.md](loss.md). Nghiên cứu được kiểm tra
> ngày 22-07-2026.

## Ranh giới khái niệm

- **Bộ dữ liệu** gồm các mẫu, phương thức, chú thích, cách chia và giấy phép.
- **Giao thức benchmark** quy định cách chia, prompt, tiền xử lý và bộ đánh giá
  được sử dụng.
- **Metric** chuyển dự đoán thành điểm số.

Điểm số chỉ có thể so sánh khi cả ba yếu tố khớp nhau. Cùng một công thức
accuracy có thể cho kết quả khác sau khi đổi bộ phân tích câu trả lời hoặc các
frame được lấy mẫu; thay đổi đó thuộc về giao thức, không thuộc bản thân accuracy.

## Metric phân loại và lấy mẫu cốt lõi

Với tính đúng sai nhị phân của mỗi mẫu $c_i\in\{0,1\}$,

$$
\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}c_i.
$$

Cao hơn là tốt hơn. Các bộ trắc nghiệm như MMLU-Pro, GPQA, MMMU-Pro, MathVista
và Video-MME thường quy về dạng này sau bước phân tích câu trả lời. Một điểm
tổng hợp phải nêu rõ trục nhóm:

- **micro/pooled accuracy** gộp mọi mẫu nên nhóm lớn nhận trọng số cao hơn;
- **macro accuracy** lấy trung bình accuracy của các nhóm nên mỗi môn, miền hoặc
  tác vụ được đặt tên nhận trọng số bằng nhau.

Lấy mẫu lặp lại có hai cách tổng hợp khác nhau. Lấy trung bình tính đúng trên các
lần lấy mẫu ước lượng accuracy kỳ vọng của một lần lấy mẫu. Nó không phải
`pass@k`. Với $n$ mẫu cho một bài toán và $c$ mẫu đúng, bộ ước lượng của HumanEval là

$$
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
$$

xác suất ít nhất một trong $k$ lần lấy mẫu thành công. `pass@k` thường tăng theo
$k$; tính đúng trung bình của mẫu không nhận được lợi thế “chỉ cần một lần thành
công” này. [Bài báo và bộ ước lượng HumanEval][humaneval]

## Tổng hợp khả năng tuân thủ chỉ dẫn

IFEval cung cấp nhiều phép kiểm tra ràng buộc nhị phân cho mỗi prompt:

$$
\mathrm{InstructionAcc}
=\frac{\text{passed constraints}}{\text{all constraints}},
\qquad
\mathrm{PromptAcc}
=\frac{1}{N}\sum_i\mathbf{1}[\text{all constraints for prompt }i\text{ pass}].
$$

Prompt accuracy là phép tổng hợp AND nên nghiêm ngặt hơn với prompt có nhiều
ràng buộc. Kiểm tra nghiêm ngặt hay nới lỏng làm thay đổi bộ đánh giá và thuộc về
giao thức benchmark; các biến thể chính xác Qwen sử dụng được ghi trong
[benchmarks.md](benchmarks.md). [Triển khai IFEval][ifeval-code]

## Metric cho văn bản, OCR và trả lời câu hỏi tài liệu

Các metric tự do tổng quát không thể thay thế lẫn nhau:

- **Exact match (EM)** là tính bằng nhau nhị phân sau phép chuẩn hóa đã công bố.
- **Token F1** là trung bình điều hòa giữa precision và recall của token trùng khớp.
- **ANLS** lấy trung bình độ tương đồng chỉnh sửa đã chuẩn hóa tốt nhất so với
  đáp án tham chiếu; các kết quả có độ tương đồng thấp bị đặt về 0 theo ngưỡng
  của benchmark.

ANLS cho điểm một phần với khác biệt chính tả/chỉnh sửa; EM thì không. Token F1
bỏ qua thứ tự token và có thể thưởng cho phần trùng khớp không liên quan. Phép
chuẩn hóa, cách xử lý nhiều đáp án tham chiếu và ngưỡng ANLS là một phần của hợp
đồng bộ chấm điểm.
[ST-VQA ANLS][st-vqa]

**OCRBench gốc không dùng EM, token F1 hay ANLS**. Mỗi mẫu trong 1.000 mẫu nhận
điểm 0/1 từ bộ chấm chính thức dựa trên chuỗi con, sau đó cộng số mẫu đạt:

$$
\mathrm{OCRBenchRaw}=\sum_{i=1}^{1000}c_i,\qquad 0\leq score\leq1000.
$$

Điểm tối đa của năm hạng mục lần lượt là 300 cho nhận dạng văn bản, 200 cho VQA
văn bản trong cảnh, 200 cho VQA tài liệu, 200 cho trích xuất thông tin chính và
100 cho nhận dạng biểu thức viết tay. Phần lớn hạng mục khớp chuỗi con tham
chiếu không phân biệt hoa thường; nhánh biểu thức viết tay loại bỏ khoảng trắng
và giữ nguyên kiểu chữ. Giá trị 0–100 được hiển thị có thể là kết quả chuẩn hóa,
nhưng chỉ được phép chuyển đổi khi đã xác nhận cùng 1.000 mẫu và bộ chấm.
[Bộ chấm OCRBench][ocrbench-code]

## Metric grounding

Với hộp dự đoán $B_p$ và hộp tham chiếu $B_g$,

$$
\mathrm{IoU}(B_p,B_g)=\frac{|B_p\cap B_g|}{|B_p\cup B_g|}.
$$

Đánh giá hiểu biểu thức tham chiếu RefCOCO thường báo cáo

$$
\mathrm{Acc@0.5}=\frac{1}{N}\sum_i
\mathbf{1}[\mathrm{IoU}(B_{p,i},B_{g,i})>0.5].
$$

Đây là grounding accuracy theo ngưỡng, không phải IoU trung bình hay mAP phát
hiện. Nó cho biết tần suất đối tượng được nhắc tới được định vị đủ tốt, nhưng
không cho biết các hộp thành công ôm sát tới mức nào. [Bài báo RefCOCO][refcoco]

## Metric cho tác tử và sử dụng công cụ

**Tỷ lệ resolved của SWE-bench** là tỷ lệ các mẫu kho mã được thử mà harness giải
quyết hoàn toàn:

$$
\mathrm{ResolvedRate}=\frac{\text{instances marked fully resolved}}
{\text{attempted instances}}.
$$

Bộ chấm chính thức hiện tại yêu cầu mọi kiểm thử `FAIL_TO_PASS` và `PASS_TO_PASS`
đều đạt mới tính là giải quyết hoàn toàn; sửa được một phần kiểm thử không được
tính là resolved. Điểm số là kết quả hệ thống nhị phân và không thể hiện chất
lượng bản vá, chi phí hay thời gian. [Bộ chấm SWE-bench][swebench-grader]

**BFCL accuracy** là tỷ lệ đạt của bộ đánh giá có phiên bản trên các trường hợp
gọi hàm. Điều kiện đạt có thể gồm kiểm tra tên hàm, đối số, kiểu/giá trị, thực
thi, tính không liên quan và trạng thái nhiều lượt. Cách tổng hợp thay đổi giữa
các bản BFCL; do đó `BFCL v3` và `BFCL-V4` là hai hợp đồng metric khác nhau,
không phải hai phép đo trên một trục ổn định. [Đánh giá BFCL][bfcl-v1]
[BFCL V4][bfcl-v4]

**Elo** tổng hợp ưu tiên tương đối theo cặp so với một tập đối thủ cụ thể. **Điểm
LLM-judge** tổng hợp quyết định từ một bộ chấm, prompt và rubric cụ thể. Cả hai
đều không phải xác suất tuyệt đối của tính đúng sự thật.

## Hiệu quả và độ bất định

Độ trễ, số token được sinh, bộ nhớ đỉnh, thông lượng và chi phí tiền tệ là các
metric hiệu quả riêng biệt. Cần báo cáo phân bố như độ trễ trung vị và p95, cùng
tỷ lệ hết thời gian/lỗi; không lấy trung bình chúng vào task accuracy nếu không
có hàm tiện ích rõ ràng.

Với $N$ mẫu nhị phân, luôn báo cáo $N$ và tốt nhất là cả khoảng tin cậy nhị thức.
Các lần sinh lặp từ cùng một prompt và các mẫu được nhóm theo chủ đề không hoàn
toàn độc lập; kết quả theo từng seed hoặc khoảng bootstrap trên đơn vị mẫu/nhóm
phù hợp trung thực hơn việc thêm chữ số thập phân.

## Danh sách kiểm tra khi báo cáo metric

Với mỗi con số, cần ghi:

```text
tên metric và chiều tốt hơn (cao hơn/thấp hơn)
công thức hoặc phiên bản chính xác của bộ đánh giá
miền/thang hiển thị và trục tổng hợp
phiên bản bộ dữ liệu và phần chia được đánh giá
prompt, tiền xử lý, bộ phân tích câu trả lời và giải mã
số mẫu, seed, mẫu số và khoảng tin cậy
```

Ba dòng đầu định nghĩa metric; các dòng còn lại gắn metric vào một giao thức
benchmark có thể tái lập.

## Nguồn

- Chen và cộng sự. *Evaluating Large Language Models Trained on Code*.
  [Bài báo và bộ ước lượng][humaneval]
- Zhou và cộng sự. *IFEval*. [Triển khai chính thức][ifeval-code]
- Biten và cộng sự. *Scene Text Visual Question Answering*. [Bài báo][st-vqa]
- Liu và cộng sự. *OCRBench*. [Bộ chấm chính thức][ocrbench-code]
- Yu và cộng sự. *Modeling Context in Referring Expressions*. [Bài báo][refcoco]
- Nhóm SWE-bench. [Mã chấm điểm chính thức][swebench-grader]
- Berkeley Function Calling Leaderboard. [Metric V1][bfcl-v1] ·
  [Cách tổng hợp V4][bfcl-v4]

[humaneval]: https://arxiv.org/abs/2107.03374
[ifeval-code]: https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_lib.py
[st-vqa]: https://arxiv.org/abs/1907.00490
[ocrbench-code]: https://github.com/qywh2023/OCRbench/blob/main/OCRBench/example.py
[refcoco]: https://arxiv.org/abs/1608.00272
[swebench-grader]: https://github.com/SWE-bench/SWE-bench/blob/master/swebench/harness/grading.py
[bfcl-v1]: https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#evaluation-metrics
[bfcl-v4]: https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html
