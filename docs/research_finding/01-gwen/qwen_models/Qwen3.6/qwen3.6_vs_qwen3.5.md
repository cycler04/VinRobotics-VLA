# Qwen3.6 vs Qwen3.5: Chỉ những khác biệt đáng kể

> **Câu hỏi nghiên cứu:** Qwen3.6 có thay đổi đáng kể Qwen3.5 không
> kiến trúc và những gì thực sự đã thay đổi trong kiến trúc, luồng dữ liệu,
> trước huấn luyện và sau huấn luyện?
>
> **Ngày nghiên cứu:** 2026-07-20. Báo cáo này so sánh trọng lượng mở
> Điểm kiểm tra Qwen3.6-27B và Qwen3.6-35B-A3B có cùng kích thước Qwen3.5
> đối tác. Các biến thể Hosted Plus/Flash/Max bị loại trừ khỏi kiến trúc
> xác nhận quyền sở hữu vì cấu hình nội bộ của họ không được công khai.
>
> **Cơ sở:** Để biết toàn bộ kiến trúc và nền tảng huấn luyện không thay đổi, hãy xem
> [Qwen3.5: Kiến trúc, Luồng dữ liệu, Huấn luyện trước và Sau huấn luyện](../Qwen3.5/qwen3.5_architecture_and_training.md).

## Câu trả lời ngắn gọn

**Qwen3.6 không phải là kiến trúc mới về mặt vật chất.** Đối với cả hai cặp mở có cùng kích thước,
tổng quan về mô hình chính thức, cấu hình, tạo phẩm token và điểm kiểm tra
cấu trúc giữ lại thiết kế Qwen3.5. Qwen3.6 vẫn tải qua
`Qwen3_5ForConditionalGeneration` hoặc `Qwen3_5MoeForConditionalGeneration`
các lớp học. Đường dẫn hình ảnh/video, kết hợp sớm, Gated DeltaNet 3:1/chú ý đầy đủ
bộ giải mã, FFN dày đặc hoặc MoE thưa thớt, RoPE đa phương thức, MTP và 262.144 token gốc
bối cảnh vẫn không thay đổi.

Thay vào đó, sự khác biệt lớn là:

1. **hành vi của mô hình được nhắm mục tiêu:** mã hóa agent mạnh hơn nhiều, công việc đầu cuối,
   lý luận, kỹ năng và tạo giao diện người dùng ở cấp độ kho lưu trữ;
2. **duy trì tư duy:** đường dẫn nhắc nhở chọn tham gia sẽ giữ lại trợ lý trước đó
   lý luận theo lượt của người dùng và Qwen cho biết 3.6 cũng là
   được huấn luyện để sử dụng những dấu vết đó;
3. **phạm vi phát hành:** chỉ có 27B điểm kiểm tra dày đặc và 35B-A3B MoE, thay vào đó
   hơn họ 0,8B-to-397B Qwen3.5.

Các nguồn công khai không tiết lộ đủ thông tin để xây dựng lại một cách khác biệt
Công thức trước hoặc sau huấn luyện Qwen3.6. Có thể nói rằng trọng lượng
và sự nhấn mạnh vào khả năng đã thay đổi. Sẽ **không** an toàn khi yêu cầu một quyền lợi mới cụ thể
kho ngữ liệu, chương trình giảng dạy SFT, thuật toán RL, phần thưởng hoặc phương pháp chắt lọc.

## Kiến trúc: thực tế không thay đổi

Các bảng kiến trúc thẻ mô hình chính thức giống hệt nhau trong mỗi bảng có cùng kích thước
cặp. So sánh cấu hình trực tiếp cho cùng một kết luận.

| Thành phần | Qwen3.5-27B -> Qwen3.6-27B | Qwen3.5-35B-A3B -> Qwen3.6-35B-A3B |
| -------------------------- | ----------------------------------------------------------- | ------------------------------------------------------ |
| Lớp kiến ​​trúc thời gian chạy | `qwen3_5`, không thay đổi | `qwen3_5_moe`, không thay đổi |
| Bố trí bộ giải mã | 64 lớp;`16 x (3 GDN + 1 full attention)` | 40 lớp;`10 x (3 GDN + 1 full attention)` |
| Chiều rộng ẩn | 5.120 | 2.048 |
| Chú ý | 24 đầu Q/4 KV; đầu mờ 256 | 16 đầu Q/2 KV; đầu mờ 256 |
| Gated DeltaNet | đầu 48V/16 QK; đầu mờ 128 | đầu 32V/16 QK; đầu mờ 128 |
| FFN hoặc MoE | FFN dày đặc, trung cấp 17,408 | 256 chuyên gia được định tuyến; top-8 + 1 được chia sẻ; chiều rộng chuyên gia 512 |
| Bộ mã hóa tầm nhìn | cùng bộ mã hóa 27 lớp, chiều rộng 1.152 và cài đặt vá/hợp nhất | giống nhau |
| Tokenizer và từ vựng | các tạo phẩm token tương tự; từ vựng đệm 248.320 | giống nhau |
| Vị trí và bối cảnh | cùng một RoPE đa phương thức; bản xứ 262,144 | giống nhau |
| MTP | một lớp MTP, được huấn luyện với nhiều bước | cùng một cấu trúc liên kết |

Nguồn: [Thẻ mẫu Qwen3.5-27B] [qwen35-27b],
[Thẻ mẫu Qwen3.6-27B] [qwen36-27b],
[Thẻ mô hình Qwen3.5-35B-A3B] [qwen35-35b] và
[Thẻ mẫu Qwen3.6-35B-A3B] [qwen36-35b]. Các cấu hình thực thi cũng giữ nguyên
các kiểu mô hình Qwen3.5 và bố cục lớp:
[Cấu hình 3,5-27B] [qwen35-27b-config], [Cấu hình 3,6-27B] [qwen36-27b-config],
[Cấu hình 3,5-35B] [qwen35-35b-config] và [cấu hình 3,6-35B] [qwen36-35b-config].

Có những khác biệt nhỏ về tạo tác nhưng chúng không thiết lập một mô hình mới
cấu trúc liên kết. Qwen3.6 làm cho một số mặc định cấu hình tiềm ẩn trước đây trở nên rõ ràng. của nó
Trọng lượng chuyên gia 35B MTP được đóng gói trong các tensor hợp nhất thay vì được tuần tự hóa như
các tensor chuyên gia riêng lẻ, trong khi kích thước MTP/MoE và tổng tham số
số lượng không thay đổi. Đây là sự khác biệt về bao bì điểm kiểm tra/thời gian chạy, không phải
bằng chứng về thuật toán MTP mới.

Kết luận thực tế rất đơn giản: không có mô-đun mới hoặc mô-đun thần kinh được sửa đổi nào.
đường dẫn dữ liệu có giá trị ghi lại. Kho lưu trữ riêng của Qwen mô tả 3.6 là tòa nhà
trên Qwen3.5 và gửi người đọc triển khai quay lại công thức phục vụ Qwen3.5.
[Kho lưu trữ Qwen3.6] [qwen36-repo]

## Thay đổi đáng kể 1: thinking trace có thể vượt qua ranh giới lượt người dùng

Qwen3.5 đã hỗ trợ tư duy và có thể giữ lại lý luận bên trong hiện tại
vòng lặp công cụ nhiều bước. Mẫu trò chuyện mặc định của nó đã loại bỏ lý do khỏi phiên bản cũ hơn
trợ lý quay lại khi xây dựng prompt sau. Qwen3.6 thêm điều kiện này:

```text
preserve_thinking == true
    -> serialize historical assistant reasoning as <think>...</think>
preserve_thinking absent or false
    -> retain the previous Qwen3.5-style behavior
```

Đường dẫn đầu vào kết quả là:

```text
stored messages
  -> split assistant reasoning_content from final content
  -> chat template
       default: drop old-turn reasoning
       preserve_thinking: keep old-turn <think> blocks
  -> tokenize the longer conversation
  -> unchanged Qwen3.5 multimodal decoder
  -> next response or tool call
```

Việc thay đổi mẫu vật liệu để lập luận là một điều kiện OR bổ sung xung quanh
lưu giữ dấu vết lịch sử. So sánh [mẫu Qwen3.5] [mẫu qwen35] với
[Mẫu Qwen3.6] [mẫu qwen36]. Qwen cũng tuyên bố rõ ràng rằng 3.6 là
“được huấn luyện thêm để bảo tồn và tận dụng” thinking trace lịch sử.
[Phần tư duy bảo tồn Qwen3.6-27B] [qwen36-27b]

Sự khác biệt này quan trọng:

- đó **không phải** là bộ nhớ lặp lại bên ngoài cửa sổ ngữ cảnh;
- nó **không** thêm mô-đun chú ý hoặc bộ đệm;
- dấu vết trước đó sử dụng token ngữ cảnh khi được tuần tự hóa;
- tuy nhiên nó có thể tránh tính toán lại cùng một lý luận và tạo ra một kết quả ổn định
  tiền tố có thể tái sử dụng trong bộ đệm KV, điều này đặc biệt hữu ích trong agent dài
  phiên.

Do đó, việc duy trì tư duy vừa là một thay đổi nhỏ về phục vụ/mẫu vừa là một
sự thay đổi năng lực thực sự sau huấn luyện. Bật cờ trên Qwen3.5 sẽ hiển thị
dấu vết cũ về phân phối đầu vào mà Qwen chưa yêu cầu tương đương
huấn luyện.

## Thay đổi đáng kể 2: trọng tâm huấn luyện chuyển sang coding agent

Các nguồn chính thức mô tả nhất quán Qwen3.6 là một giải pháp ổn định và phù hợp với thế giới thực.
cập nhật tiện ích tập trung vào quy trình làm việc ở giao diện người dùng, lý luận ở cấp độ kho lưu trữ và
agent lặp đi lặp lại làm việc. Bằng chứng mạnh mẽ nhất là so sánh benchmark có cùng kích thước phù hợp được công bố trong thẻ mô hình 3.6.

| Benchmark | 27B: Qwen3.5 -> Qwen3.6 | Đồng bằng thô | 35B-A3B: Qwen3.5 -> Qwen3.6 | Đồng bằng thô |
| ---------------------- | ----------------------: | --------: | --------------------------: | --------: |
| Băng ghế dự bị SWE đã được xác minh |            75,0 -> 77,2 |      +2,2 |                70,0 -> 73,4 |      +3,4 |
| Ghế dự bị SWE Pro |            51,2 -> 53,5 |      +2,3 |                44,6 -> 49,5 |      +4,9 |
| SWE-băng ghế dự bị đa ngôn ngữ |            69,3 -> 71,3 |      +2.0 |                60,3 -> 67,2 |      +6,9 |
| Terminal-Băng ghế dự bị 2.0 |            41,6 -> 59,3 |     +17,7 |                40,5 -> 51,5 |     +11.0 |
| SkillBench Trung bình5 |            27,2 -> 48,2 |     +21.0 |                 4,4 -> 28,7 |     +24,3 |
| NL2Repo |            27,3 -> 36,2 |      +8,9 |                20,5 -> 29,4 |      +8,9 |
| QwenWebBench |          1.068 -> 1.487 |      +419 |                978 -> 1.397 |      +419 |

Nguồn và ghi chú đánh giá: [Bảng benchmark Qwen3.6-27B] [qwen36-27b] và
[Bảng benchmark Qwen3.6-35B-A3B] [qwen36-35b]. Đồng bằng thô nên được giải thích
chỉ trong mỗi benchmark; quy mô của họ không thể thay thế cho nhau. QwenWebBench
là một benchmark nội bộ và một số đánh giá agent sử dụng agent đã nêu của Qwen
giàn giáo và thiết lập tài nguyên, vì vậy những con số này là bằng chứng định hướng, không phải
bằng chứng độc lập về hiệu suất triển khai.

Mức tăng không đồng đều trên tất cả các khả năng. Ví dụ: Qwen3.6-27B là
thấp hơn một chút so với Qwen3.5-27B trên MathVista (87,4 so với 87,8) và DynaMath (85,6 so với
87,7), trong khi nhiều điểm thị lực khác chênh lệch ít hơn một điểm. các
Mô hình 35B-A3B cũng thụt lùi một chút trên Claw-Eval Pass^3 (50,0 so với 51,0).
Mẫu này phù hợp với cập nhật mã hóa/agent được nhắm mục tiêu hơn là
bước nhảy vọt về kiến trúc nói chung.

## Điều gì đã thay đổi trước và sau huấn luyện?

### Đã xác minh

- Các hiện vật được phát hành là các điểm kiểm tra sau huấn luyện và nhãn thẻ của chúng
  giai đoạn huấn luyện tổng thể là “Trước huấn luyện và sau huấn luyện”.
- Qwen3.6 có trọng lượng mới trong khi vẫn giữ nguyên kiến trúc.
- Qwen cho biết các người mẫu đã được huấn luyện bổ sung về cách bảo quản và sử dụng
  thinking trace lịch sử.
- Ngôn ngữ phát hành chính thức và mẫu chuẩn xác định mã hóa agent, tạo giao diện người dùng, lý luận kho lưu trữ, sử dụng công cụ và độ ổn định là mục tiêu năng lực chính.

### Không được tiết lộ công khai

Không có báo cáo kỹ thuật Qwen3.6 hoặc công thức huấn luyện hoàn chỉnh nào có sẵn tại
ngày nghiên cứu. Các bài đăng phát hành, kho lưu trữ, thẻ mô hình, cấu hình và mẫu
không nêu:

- liệu 3.6 có bắt đầu từ tạ Qwen3.5 thông qua quá trình luyện tập trước liên tục hay không, hay bằng cách nào
  nhiều khóa huấn luyện trước đã được lặp lại;
- số lượng token trước khi huấn luyện mới, nguồn kho ngữ liệu, hỗn hợp mã/đa phương thức hoặc dữ liệu
  cắt đứt;
- Các giai đoạn SFT, thành phần tập dữ liệu, số lượng quỹ đạo hoặc lấy mẫu từ chối
  thủ tục;
- Thuật toán RL, chức năng khen thưởng, giám khảo, môi trường, chương trình giảng dạy hoặc tính toán;
- quá trình chưng cất, căn chỉnh an toàn hoặc kiểm soát đánh giá ô nhiễm.

Do đó, “chủ yếu là sau huấn luyện” là cách giải thích hợp lý cho ngắn hạn
khoảng thời gian phát hành, kiến trúc không thay đổi, dấu vết suy nghĩ được thêm vào một cách rõ ràng
huấn luyện và lợi ích của đại lý tập trung, nhưng nó vẫn là **suy luận**, không phải là
công thức được công bố. Các thẻ chính thức tiếp tục nói cả huấn luyện trước và
sau huấn luyện, vì vậy báo cáo này không cho rằng không có huấn luyện trước.

## Phạm vi phát hành cũng thay đổi

Tính đến ngày nghiên cứu, bộ sưu tập Hugging Face Qwen3.6 chính thức chứa
chỉ có hai cấu trúc liên kết mô hình mở—dày đặc 27B và 35B-A3B MoE—cộng với các bản sao FP8 của chúng.
[Bộ sưu tập Qwen3.6] [bộ sưu tập qwen36] Qwen3.5 được phát hành trên 0.8B, 2B, 4B,
Các biến thể 9B, 27B, 35B-A3B, 122B-A10B và 397B-A17B. Như vậy Qwen3.6 tốt hơn
được hiểu là sự làm mới có trọng tâm của hai điểm kiểm tra thực tế hơn là một sự hoàn chỉnh
thay thế cho dòng Qwen3.5.

Không được sử dụng tên Qwen3.6 Plus/Flash/Max được lưu trữ để suy ra tham số ẩn
số lượng hoặc kiến trúc. Hành vi sản phẩm công của họ có thể khác nhau, nhưng có
không có cấu hình cùng kích thước nào có thể kiểm tra được hỗ trợ so sánh cấp độ kiến trúc.

## Kết luận

Đối với luồng dữ liệu cấp mô-đun và kiến trúc, hãy tiếp tục sử dụng báo cáo Qwen3.5:
**không có tài liệu delta Qwen3.6 đáng kể nào.** Công việc 3.6 đầy ý nghĩa
là một bản cập nhật về trọng lượng/huấn luyện dành cho các coding agent cộng với một cách chọn tham gia để thực hiện
suy luận lịch sử thông qua gợi ý. Mức tăng được báo cáo chính xác là lớn nhất
trong đó Qwen nói rằng nó tập trung—nhiệm vụ đầu cuối, công việc kho lưu trữ, kỹ năng và web/front-
thế hệ cuối—trong khi tầm nhìn và lý luận chung chủ yếu là tăng dần hoặc
hỗn hợp.

## Nguồn

Tất cả các nguồn bên dưới là kho lưu trữ chính thức của Qwen, tạo phẩm mô hình hoặc bản phát hành
trang, truy cập 2026-07-20.

- [Kho lưu trữ chính thức của Qwen3.6] [qwen36-repo]
- [Thẻ mô hình Qwen3.6-27B và phương pháp đo benchmark] [qwen36-27b]
- [Thẻ mô hình Qwen3.6-35B-A3B và phương pháp đo benchmark] [qwen36-35b]
- [Bộ sưu tập mô hình chính thức của Qwen3.6] [qwen36-collection]
- [Thẻ mẫu Qwen3.5-27B] [qwen35-27b]
- [Thẻ mẫu Qwen3.5-35B-A3B] [qwen35-35b]
- [Mẫu trò chuyện Qwen3.5 và Qwen3.6] [qwen35-template]
  / [Mẫu Qwen3.6][qwen36-template]
- [Các tạo phẩm cấu hình có cùng kích thước không thể thay đổi] [qwen35-27b-config]
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
