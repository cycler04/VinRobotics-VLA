# Qwen3.6 so với Qwen3.5: Chỉ những khác biệt đáng kể

> **Câu hỏi nghiên cứu:** Qwen3.6 có thay đổi đáng kể kiến trúc Qwen3.5 hay không,
> và điều gì thực sự đã thay đổi trong kiến trúc, luồng dữ liệu, pretraining và
> post-training?
>
> **Ngày nghiên cứu:** 2026-07-20. Báo cáo này so sánh các checkpoint trọng số mở
> Qwen3.6-27B và Qwen3.6-35B-A3B với các phiên bản Qwen3.5 cùng kích thước.
> Các biến thể hosted Plus/Flash/Max không được đưa vào các kết luận về kiến trúc
> vì cấu hình nội bộ của chúng không được công khai.
>
> **Tài liệu nền:** Để biết toàn bộ kiến trúc và nền tảng huấn luyện không thay đổi, xem
> [Qwen3.5: Kiến trúc, luồng dữ liệu, pretraining và post-training](../Qwen3.5/qwen3.5_architecture_and_training.md).

## Câu trả lời ngắn

**Qwen3.6 không phải một kiến trúc mới về bản chất.** Đối với cả hai cặp mô hình mở
cùng kích thước, phần tổng quan mô hình chính thức, cấu hình, artifact tokenizer và cấu trúc
checkpoint đều giữ nguyên thiết kế Qwen3.5. Qwen3.6 vẫn được tải thông qua các class
`Qwen3_5ForConditionalGeneration` hoặc `Qwen3_5MoeForConditionalGeneration`.
Đường hình ảnh/video, early fusion, bộ giải mã với tỷ lệ Gated DeltaNet 3:1 so với
full attention, FFN dense hoặc sparse MoE, multimodal RoPE, MTP và ngữ cảnh gốc
262.144 token đều không thay đổi.

Thay vào đó, những khác biệt lớn là:

1. **hành vi mô hình được nhắm mục tiêu:** mạnh hơn đáng kể về coding agent, nhiệm vụ đầu cuối,
   suy luận ở cấp repository, skill và sinh giao diện người dùng;
2. **giữ lại thinking trace:** một đường prompt tùy chọn giữ lại phần suy luận ở các lượt
   assistant trước qua ranh giới lượt người dùng, và Qwen cho biết 3.6 còn được huấn luyện
   để sử dụng các trace đó;
3. **phạm vi phát hành:** chỉ có checkpoint dense 27B và MoE 35B-A3B, thay vì
   dòng Qwen3.5 từ 0,8B đến 397B.

Các nguồn công khai không tiết lộ đủ thông tin để dựng lại công thức pretraining hoặc
post-training khác biệt của Qwen3.6. Có thể khẳng định trọng số và trọng tâm năng lực đã
thay đổi. Sẽ **không** an toàn nếu khẳng định một corpus mới cụ thể, chương trình SFT,
thuật toán RL, reward hay phương pháp distillation.

## Kiến trúc: về cơ bản không thay đổi

Các bảng kiến trúc trong model card chính thức giống hệt nhau ở từng cặp cùng kích thước.
So sánh trực tiếp cấu hình cũng dẫn đến cùng kết luận.

| Thành phần | Qwen3.5-27B -> Qwen3.6-27B | Qwen3.5-35B-A3B -> Qwen3.6-35B-A3B |
| -------------------------- | ----------------------------------------------------------- | ------------------------------------------------------ |
| Class kiến trúc khi chạy | `qwen3_5`, không đổi | `qwen3_5_moe`, không đổi |
| Bố cục bộ giải mã | 64 layer; `16 x (3 GDN + 1 full attention)` | 40 layer; `10 x (3 GDN + 1 full attention)` |
| Chiều rộng hidden | 5.120 | 2.048 |
| Attention | 24 Q head/4 KV head; head dim 256 | 16 Q head/2 KV head; head dim 256 |
| Gated DeltaNet | 48 V head/16 QK head; head dim 128 | 32 V head/16 QK head; head dim 128 |
| FFN hoặc MoE | FFN dense, intermediate 17.408 | 256 routed expert; top-8 + 1 shared; expert width 512 |
| Vision encoder | Cùng encoder 27 layer, chiều rộng 1.152 và thiết lập patch/merger | Giống nhau |
| Tokenizer và vocabulary | Các artifact tokenizer giống nhau; padded vocabulary 248.320 | Giống nhau |
| Vị trí và ngữ cảnh | Cùng multimodal RoPE; ngữ cảnh gốc 262.144 | Giống nhau |
| MTP | Một layer MTP, được huấn luyện nhiều bước | Cùng cấu trúc |

Nguồn: [Model card Qwen3.5-27B][qwen35-27b],
[model card Qwen3.6-27B][qwen36-27b],
[model card Qwen3.5-35B-A3B][qwen35-35b] và
[model card Qwen3.6-35B-A3B][qwen36-35b]. Các cấu hình có thể thực thi cũng giữ
nguyên loại mô hình và bố cục layer Qwen3.5:
[cấu hình 3.5-27B][qwen35-27b-config], [cấu hình 3.6-27B][qwen36-27b-config],
[cấu hình 3.5-35B][qwen35-35b-config] và [cấu hình 3.6-35B][qwen36-35b-config].

Có những khác biệt nhỏ về artifact nhưng chúng không tạo thành một cấu trúc mô hình mới.
Qwen3.6 biểu diễn tường minh một số giá trị mặc định cấu hình từng được để ẩn. Trọng số
expert MTP của bản 35B được đóng gói trong tensor hợp nhất thay vì tuần tự hóa thành các
tensor expert riêng, trong khi kích thước MTP/MoE và tổng số tham số không thay đổi.
Đây là khác biệt về đóng gói checkpoint/runtime, không phải bằng chứng về một thuật toán
MTP mới.

Kết luận thực tế rất đơn giản: không có module neural mới hoặc đường dữ liệu neural đã sửa
đổi nào đáng để ghi thành tài liệu riêng. Chính repository Qwen mô tả 3.6 là được xây dựng
trên Qwen3.5 và hướng người đọc về công thức triển khai Qwen3.5 để phục vụ mô hình.
[Repository Qwen3.6][qwen36-repo]

## Thay đổi đáng kể 1: thinking trace có thể vượt qua ranh giới lượt người dùng

Qwen3.5 đã hỗ trợ thinking và có thể giữ phần suy luận trong vòng lặp công cụ nhiều bước
hiện tại. Chat template mặc định của nó loại phần suy luận khỏi các lượt assistant cũ khi
dựng prompt tiếp theo. Qwen3.6 bổ sung điều kiện sau:

```text
preserve_thinking == true
    -> tuần tự hóa phần suy luận lịch sử của assistant thành <think>...</think>
preserve_thinking không có hoặc false
    -> giữ hành vi kiểu Qwen3.5 trước đây
```

Đường đầu vào sau thay đổi:

```text
message đã lưu
  -> tách reasoning_content của assistant khỏi nội dung cuối
  -> chat template
       mặc định: bỏ phần suy luận của lượt cũ
       preserve_thinking: giữ block <think> của lượt cũ
  -> tokenize cuộc hội thoại dài hơn
  -> bộ giải mã đa phương thức Qwen3.5 không đổi
  -> phản hồi hoặc tool call tiếp theo
```

Thay đổi quan trọng trong template là một điều kiện OR bổ sung quanh việc giữ lại trace
lịch sử. So sánh [template Qwen3.5][qwen35-template] với
[template Qwen3.6][qwen36-template]. Qwen cũng tuyên bố rõ rằng 3.6 được
“huấn luyện thêm để giữ lại và tận dụng” thinking trace lịch sử.
[Phần preserve thinking của Qwen3.6-27B][qwen36-27b]

Khác biệt này có ý nghĩa:

- nó **không phải** bộ nhớ lặp bên ngoài cửa sổ ngữ cảnh;
- nó **không** thêm module attention hoặc memory bank;
- trace trước chiếm token ngữ cảnh khi được tuần tự hóa;
- tuy nhiên, nó có thể tránh phải tính lại cùng một phần suy luận và tạo một prefix ổn định
  có thể tái sử dụng trong KV cache, đặc biệt hữu ích trong các phiên agent dài.

Vì vậy, preserve thinking vừa là thay đổi nhỏ ở lớp serving/template, vừa là thay đổi năng
lực post-training thực sự. Bật cờ này trên Qwen3.5 sẽ đưa trace cũ vào một phân phối đầu vào
mà Qwen chưa tuyên bố là đã được huấn luyện tương đương.

## Thay đổi đáng kể 2: trọng tâm huấn luyện chuyển sang coding agent

Các nguồn chính thức nhất quán mô tả Qwen3.6 là bản cập nhật về độ ổn định và tính hữu ích
trong thực tế, tập trung vào workflow terminal, suy luận ở cấp repository và công việc agent
lặp. Bằng chứng mạnh nhất là so sánh benchmark công bố, được khớp theo cùng kích thước trong
model card 3.6.

| Benchmark | 27B: Qwen3.5 -> Qwen3.6 | Chênh lệch thô | 35B-A3B: Qwen3.5 -> Qwen3.6 | Chênh lệch thô |
| ---------------------- | ----------------------: | --------: | --------------------------: | --------: |
| SWE-bench Verified | 75,0 -> 77,2 | +2,2 | 70,0 -> 73,4 | +3,4 |
| SWE-bench Pro | 51,2 -> 53,5 | +2,3 | 44,6 -> 49,5 | +4,9 |
| SWE-bench Multilingual | 69,3 -> 71,3 | +2,0 | 60,3 -> 67,2 | +6,9 |
| Terminal-Bench 2.0 | 41,6 -> 59,3 | +17,7 | 40,5 -> 51,5 | +11,0 |
| SkillBench Avg@5 | 27,2 -> 48,2 | +21,0 | 4,4 -> 28,7 | +24,3 |
| NL2Repo | 27,3 -> 36,2 | +8,9 | 20,5 -> 29,4 | +8,9 |
| QwenWebBench | 1.068 -> 1.487 | +419 | 978 -> 1.397 | +419 |

Nguồn và ghi chú đánh giá: [bảng benchmark Qwen3.6-27B][qwen36-27b] và
[bảng benchmark Qwen3.6-35B-A3B][qwen36-35b]. Chênh lệch thô chỉ nên được diễn giải
trong từng benchmark; thang đo của chúng không hoán đổi được. QwenWebBench là benchmark
nội bộ và một số đánh giá agent dùng scaffold agent cùng thiết lập tài nguyên do Qwen nêu,
vì vậy các con số này là bằng chứng định hướng chứ không phải bằng chứng độc lập về hiệu
năng triển khai.

Mức tăng không đồng đều trên mọi năng lực. Ví dụ, Qwen3.6-27B thấp hơn một chút so với
Qwen3.5-27B trên MathVista (87,4 so với 87,8) và DynaMath (85,6 so với 87,7), trong khi
nhiều điểm thị giác khác chênh lệch chưa đến một điểm. Mô hình 35B-A3B cũng giảm nhẹ trên
Claw-Eval Pass^3 (50,0 so với 51,0). Mẫu kết quả này phù hợp với một bản cập nhật có mục
tiêu cho coding/agent hơn là một bước nhảy kiến trúc tổng quát.

## Điều gì đã thay đổi trong pretraining và post-training?

### Đã xác minh

- Các artifact phát hành là checkpoint đã qua post-training và model card gắn nhãn toàn bộ
  giai đoạn huấn luyện là “Pretraining & Post-training”.
- Qwen3.6 có trọng số mới nhưng giữ nguyên kiến trúc.
- Qwen cho biết các mô hình được huấn luyện thêm để giữ và sử dụng thinking trace lịch sử.
- Ngôn ngữ phát hành chính thức và mẫu benchmark xác định coding agent, sinh giao diện người
  dùng, suy luận repository, sử dụng công cụ và độ ổn định là các mục tiêu năng lực chính.

### Không được công khai

Tại ngày nghiên cứu, chưa có báo cáo kỹ thuật Qwen3.6 hoặc công thức huấn luyện hoàn chỉnh.
Các bài đăng phát hành, repository, model card, cấu hình và template không nêu:

- 3.6 có bắt đầu từ trọng số Qwen3.5 thông qua continued pretraining hay không, hoặc bao nhiêu
  phần pretraining đã được chạy lại;
- số token pretraining mới, nguồn corpus, tỷ lệ trộn code/đa phương thức hoặc mốc cắt dữ liệu;
- các giai đoạn SFT, thành phần dataset, số trajectory hoặc quy trình rejection sampling;
- thuật toán RL, hàm reward, judge, môi trường, curriculum hoặc compute;
- quy trình distillation, safety alignment hoặc kiểm soát contamination trong đánh giá.

Do đó, “chủ yếu là post-training” là một diễn giải hợp lý dựa trên khoảng cách phát hành ngắn,
kiến trúc không đổi, huấn luyện thinking trace được bổ sung rõ ràng và mức tăng tập trung vào
agent, nhưng vẫn chỉ là **suy luận**, không phải công thức được công bố. Các model card chính
thức tiếp tục đề cập cả pretraining và post-training, vì vậy báo cáo này không tuyên bố rằng
không có pretraining.

## Phạm vi phát hành cũng thay đổi

Tính đến ngày nghiên cứu, bộ sưu tập Qwen3.6 chính thức trên Hugging Face chỉ chứa hai cấu
trúc mô hình mở—dense 27B và MoE 35B-A3B—cùng các bản sao FP8 của chúng.
[Bộ sưu tập Qwen3.6][qwen36-collection] Qwen3.5 được phát hành với các biến thể 0,8B, 2B,
4B, 9B, 27B, 35B-A3B, 122B-A10B và 397B-A17B. Vì vậy, Qwen3.6 nên được hiểu là bản làm
mới có trọng tâm cho hai checkpoint thực dụng, thay vì sự thay thế hoàn chỉnh cho dòng
Qwen3.5.

Không nên dùng tên các biến thể hosted Qwen3.6 Plus/Flash/Max để suy ra số tham số hoặc kiến
trúc ẩn. Hành vi sản phẩm công khai của chúng có thể khác nhau, nhưng không có cấu hình cùng
kích thước nào có thể kiểm tra để hỗ trợ so sánh ở cấp kiến trúc.

## Kết luận

Đối với kiến trúc và luồng dữ liệu cấp module, tiếp tục dùng báo cáo Qwen3.5:
**không có delta Qwen3.6 đáng kể nào cần ghi thành tài liệu riêng.** Phần thay đổi có ý nghĩa
trong 3.6 là cập nhật trọng số/huấn luyện cho coding agent, cộng với cách tùy chọn để đưa phần
suy luận lịch sử đi tiếp qua prompt. Mức tăng được báo cáo lớn nhất đúng ở những nơi Qwen cho
biết họ tập trung—nhiệm vụ terminal, công việc repository, skill và sinh web/frontend—trong
khi thị giác và suy luận tổng quát chủ yếu tăng dần hoặc có kết quả trái chiều.

## Nguồn

Tất cả nguồn dưới đây là repository, artifact mô hình hoặc trang phát hành chính thức của
Qwen, được truy cập ngày 2026-07-20.

- [Repository Qwen3.6 chính thức][qwen36-repo]
- [Model card Qwen3.6-27B và phương pháp đo benchmark][qwen36-27b]
- [Model card Qwen3.6-35B-A3B và phương pháp đo benchmark][qwen36-35b]
- [Bộ sưu tập mô hình Qwen3.6 chính thức][qwen36-collection]
- [Model card Qwen3.5-27B][qwen35-27b]
- [Model card Qwen3.5-35B-A3B][qwen35-35b]
- [Chat template Qwen3.5][qwen35-template]
  / [template Qwen3.6][qwen36-template]
- [Các artifact cấu hình cùng kích thước không đổi][qwen35-27b-config]
  / [3.6-27B][qwen36-27b-config]
  / [3.5-35B-A3B][qwen35-35b-config]
  / [3.6-35B-A3B][qwen36-35b-config]

[qwen36-repo]: https://github.com/QwenLM/Qwen3.6
[qwen36-collection]: https://huggingface.co/collections/Qwen/qwen36
[qwen35-27b]: https://huggingface.co/Qwen/Qwen3.5-27B
[qwen36-27b]: https://huggingface.co/Qwen/Qwen3.6-27B
[qwen35-35b]: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
[qwen36-35b]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
[qwen35-template]: https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/chat_template.jinja
[qwen36-template]: https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/chat_template.jinja
[qwen35-27b-config]: https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json
[qwen36-27b-config]: https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json
[qwen35-35b-config]: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json
[qwen36-35b-config]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json
