# Đánh giá mô hình Qwen

> **Câu hỏi nghiên cứu:** Điểm chuẩn nào hữu ích cho việc đánh giá Qwen
> gia đình như một ngôn ngữ, ngôn ngữ tầm nhìn và xương sống tác nhân, đặc biệt là trước
> sử dụng nó bên trong hệ thống VLA?
>
> **Phạm vi:** Qwen3, Qwen3.5, Qwen3.6 và Qwen2.5-VL. Vòng kín của Qwen-VLA
> đánh giá chính sách được ghi lại riêng biệt trong
> [VLA/benchmarks.md](../VLA/benchmarks.md). Nghiên cứu được kiểm tra vào ngày 22-07-2026.

## Câu trả lời ngắn

Không có một "điểm Qwen" nào có ý nghĩa duy nhất. Một đánh giá hữu ích có ba
lớp:

1. **ngôn ngữ và lý luận** để hiểu hướng dẫn, lập kế hoạch và công cụ
   sử dụng;
2. **ngôn ngữ tầm nhìn** dành cho nhận thức, OCR, nền tảng, lý luận không gian và
   sự hiểu biết tạm thời;
3. **thực thi nhiệm vụ** dành cho tổng đài viên hoặc robot, trong đó không có điểm chuẩn cho câu trả lời
   thay thế cho sự thành công trong một môi trường.

Đối với xương sống robot, chỉ riêng MMMU là không đủ. Bảng hữu ích tối thiểu là
`MMMU-Pro + MathVista + OCRBench + RefCOCO/RefSpatialBench + VideoMME`, đã theo dõi
bởi bộ VLA vòng kín. Thẻ mẫu Qwen3.5 báo cáo tất cả các khả năng này
gia đình, trong khi bài báo Qwen-VLA cho thấy họ vẫn không dự đoán được robot
thành công của chính mình. [Thẻ mẫu Qwen3.5] [qwen35-card]
[Báo cáo Qwen-VLA, Phần 5 và 7][qwen-vla]

## Mỗi nhóm điểm chuẩn đo lường điều gì

| Năng lực | Điểm chuẩn đại diện | Điểm số hỗ trợ điều gì | Những gì nó không thiết lập |
|---|---|---|---|
| Kiến thức | MMLU-Pro, MMLU-Redux, GPQA-Diamond | Kiến thức học thuật rộng và khó QA | Nhận thức có căn cứ hoặc độ tin cậy của hành động |
| Hướng dẫn sau | IFEval, IFBench, Đa thử thách | Ràng buộc và tuân thủ định dạng | Tính khả thi về mặt vật lý hoặc khả năng phục hồi sau thất bại |
| Lý luận | MATH-500, AIME, HLE, MathVista | Lý luận bằng văn bản hoặc hình ảnh theo một giao thức trả lời xác định | Kiểm soát đường chân trời dài ổn định |
| Mã hóa và đại lý | LiveCodeBench, SWE-bench, BFCL, Băng ghế đầu cuối | Tạo mã, sửa chữa kho lưu trữ, gọi hàm, thực thi công cụ | Nền tảng trực quan chung hoặc robot |
| Tổng hợp VQA | MMMU/MMMU-Pro, MMStar, RealWorldQA | Hiểu hình ảnh đa miền | Định vị đối tượng chính xác và kiểm soát thời gian |
| OCR và tài liệu | OCRBench, OmniDocBench, TextVQA | Nhận dạng văn bản và suy luận tài liệu | Hình học hoặc thao tác 3D |
| Nối đất không gian | RefCOCO, RefSpatialBench, EmbSpatialBench, CountBench | Độ phân giải tham chiếu, vị trí, mối quan hệ và cách đếm | Chất lượng quỹ đạo có thể thực hiện được |
| Video | VideoMME, VideoMMMU, MLVU, MVBench | Hiểu biết về thời gian và video dài | Tương tác vòng kín với trạng thái thay đổi |

Báo cáo kỹ thuật Qwen3 phân biệt rõ ràng kiến ​​thức chung, căn chỉnh,
toán/lý luận, mã hóa/tác nhân và đánh giá đa ngôn ngữ. Qwen2.5-VL và
Qwen3.5 thêm các bảng riêng biệt để hiểu chung về VQA, OCR/tài liệu,
nối đất và video. Những nhóm này có nhiều thông tin hơn so với việc lấy trung bình tất cả
điểm thành một số. [Báo cáo Qwen3, Phần 4.6] [qwen3]
[Báo cáo Qwen2.5-VL, Phần 4] [qwen25-vl]

## Ranh giới giao thức điểm chuẩn

Các công thức và cách diễn giải số liệu được tách biệt trong [metrics.md](metrics.md).
Các lựa chọn giao thức bên dưới xác định dự đoán nào đạt đến từng số liệu và
do đó là một phần của báo cáo điểm chuẩn chứ không phải định nghĩa số liệu.

| Điểm chuẩn | Giao thức phải được ghim | Bằng chứng Qwen hiện tại |
|---|---|---|
| GPQA-Diamond | Nhắc nhở, trả lời trình phân tích cú pháp, giải mã và mẫu cho mỗi câu hỏi | Qwen3 lấy 10 mẫu cho mỗi câu hỏi và tính trung bình độ chính xác của chúng; đây không phải là `pass@10` |
| IFEval | Công cụ đánh giá chặt chẽ/lỏng lẻo và tổng hợp nhắc nhở/hướng dẫn | Qwen3 báo cáo độ chính xác nhanh chóng. Nhãn `IFEval` trần của Qwen3.5 để lại biến thể chính xác chưa được giải quyết |
| OCRBench | Bản phát hành gốc/v2, cầu thủ ghi bàn chính thức và thang điểm hiển thị | OCRBench gốc có tổng cộng 1.000 điểm mục nhị phân. Qwen2.5-VL báo cáo tổng số thô như 885; Qwen3.5 báo cáo 89.4 mà không ghi lại liệu đây có chính xác là chuẩn hóa `/10` hay không |
| Tham chiếuCOCO | Tập hợp phân tách, trình phân tích cú pháp hộp và tổng hợp trên val/testA/testB | Qwen3.5 báo cáo `RefCOCO(avg)` nhưng không xác định được các phần tách thành phần |
| Video-MME | Phiên bản, điều kiện phụ đề, bộ lấy mẫu khung/FPS và trình phân tích cú pháp chỉ trả lời | Qwen3.5 xác định các hàng có/không có phụ đề nhưng không công bố chính sách khung hoàn chỉnh |
| SWE-bench | Sửa đổi tập dữ liệu, khung tác nhân, công cụ, bối cảnh, thời gian chờ và số lần thử | Giá trị được báo cáo là kết quả của hệ thống; trọng lượng mô hình một mình không tái tạo nó |
| BFCL | Phiên bản, bộ danh mục, trình xử lý/đánh giá và tổng hợp danh mục | Qwen3 sử dụng BFCL v3 và Qwen3.5 sử dụng BFCL-V4; những con số không thể so sánh trực tiếp |

Công cụ đánh giá lỏng lẻo chính thức của IFEval không phải là kết hợp mờ tùy ý. Nó kiểm tra một
tập hợp các biến thể đầu ra cố định, chẳng hạn như xóa dòng đầu tiên/cuối cùng hoặc dấu hoa thị;
kiểm tra nghiêm ngặt phản hồi không thay đổi. [Triển khai IFEval][ifeval-code]
[Cài đặt đánh giá Qwen3] [qwen3]

## Ảnh chụp nhanh kết quả đã được xác minh

Các bảng bên dưới là **do nhà xuất bản báo cáo**, không được sao chép cục bộ. Họ là
được đưa vào để cho thấy những kết luận nào mà các nghị định thư chính thức ủng hộ, chứ không phải để tạo ra một kết luận
bảng xếp hạng phổ quát.

### Sự phát triển ngôn ngữ và tác nhân

| Mô hình và chế độ | MMLU-Pro | GPQA-Diamond | Bằng chứng mã hóa/tác nhân | Phiên dịch được hỗ trợ |
|---|---:|---:|---|---|
| Qwen3-235B-A22B, đang suy nghĩ | không được báo cáo trong Bảng 11 của tờ báo | 71.1 | BFCL v3 70.8; LiveCodeBench v5 70.7 | Mô hình lý luận/tác nhân mạnh mẽ với ngân sách lấy mẫu dài |
| Qwen3.5-27B | 86,1 | 85,5 | SWE-bench Đã xác minh 72.4; BFCL-V4 68.5 | Mô hình đa phương thức bản địa nhỏ hơn nhiều với điểm số tác nhân và ngôn ngữ mạnh mẽ |
| Qwen3.6-27B | 86,2 | 87,8 | SWE-bench Đã xác minh 77.2; Thiết Bị Đầu Cuối-Băng Ghế 2.0 59.3 | Đạt được lợi ích rõ ràng hơn trong việc thực thi tác nhân mã hóa so với kiến ​​thức chung |

Nguồn: [Báo cáo Qwen3, Bảng 11-12] [qwen3],
[Thẻ mẫu Qwen3.5-27B] [qwen35-card] và
[Thẻ mẫu Qwen3.6-27B] [qwen36-card].

**Đã xác minh:** Kết quả suy nghĩ và không suy nghĩ của Qwen3 sử dụng cách lấy mẫu khác nhau
cài đặt. Việc đánh giá tác nhân mã hóa của Qwen3.6 cũng phụ thuộc vào một giàn giáo cụ thể,
cửa sổ ngữ cảnh, bộ công cụ, thời gian chờ và các lần chạy lặp lại. Vì vậy, các giá trị nên
chỉ được so sánh khi các cài đặt đó khớp. [Báo cáo Qwen3, Phần 4.6] [qwen3]
[Ghi chú đánh giá Qwen3.6] [qwen36-card]

### Ảnh chụp nhanh ngôn ngữ tầm nhìn

Thẻ Qwen3.5-27B chính thức báo cáo số điểm được chọn sau:

| Năng lực | Điểm chuẩn | Qwen3.5-27B |
|---|---|---:|
| Lý luận đa phương thức chuyên gia | MMMU-Pro | 75,0 |
| Toán học trực quan | MathVista-mini | 87,8 |
| Tài liệu/OCR | OCRBench | 89,4 |
| Giới thiệu/không gian | RefCOCO trung bình | 90,9 |
| Lý luận không gian thể hiện | EmbSpatialBench | 84,5 |
| Video dài, phụ đề bị tắt | VideoMME | 82,8 |

Những con số này hỗ trợ năng lực thị giác rộng rãi nhưng chúng không phải là thước đo hành động.
VLM có thể xác định một đối tượng hoặc trả lời câu hỏi về không gian trong khi vẫn tạo ra
hành động không an toàn, bị trì hoãn hoặc không nhất quán. [Thẻ mẫu Qwen3.5,
Bảng ngôn ngữ thị giác[qwen35-card]

## Đánh giá đề xuất cho không gian làm việc này

### Cổng xương sống

Sử dụng cùng một điểm kiểm tra, bộ xử lý, chính sách lấy mẫu hình ảnh/video, mẫu lời nhắc,
chế độ tạo và độ dài đầu ra tối đa cho mỗi lần chạy. Ghi:

- sửa đổi mô hình chính xác và dtype/lượng tử hóa;
- độ phân giải hình ảnh gốc và khung video/chính sách FPS;
- chế độ suy nghĩ hoặc không suy nghĩ và giải mã các thông số;
- phiên bản chuẩn, phân chia, số liệu, người đánh giá và số lượng mẫu;
- độ trễ, bộ nhớ tối đa và lỗi cũng như điểm nhiệm vụ.

Để lựa chọn sớm, hãy chạy một bảng nhỏ gọn:

| Ưu tiên | Loại điểm chuẩn | Lý do cho dự án VLA |
|---|---|---|
| P0 | RefCOCO hoặc RefSpatialBench | Kiểm tra xem ngôn ngữ có đề cập đến đúng đối tượng hoặc khu vực hay không |
| P0 | VideoMME không có phụ đề | Kiểm tra nhận thức về thời gian mà không để lộ câu trả lời qua văn bản |
| P0 | MathVista hoặc EmbSpatialBench | Kiểm tra lý luận không gian và sơ đồ |
| P1 | OCRBench | Hữu ích khi cảnh robot có nhãn, màn hình hoặc biển báo |
| P1 | IFEval | Kiểm tra việc tuân thủ hướng dẫn kiểm soát |
| P2 | MMLU-Pro/GPQA | Kiểm tra độ tỉnh táo trên diện rộng nhưng kết hợp yếu với điều khiển robot |

### Cổng chính sách

Chỉ quảng bá đường trục sau khi đánh giá vòng kín báo cáo tỷ lệ thành công,
phân chia tổng quát, tần số điều khiển, chân trời hành động, độ trễ đồng hồ treo tường và
các hạng mục thất bại. Xem [VLA/benchmarks.md](../VLA/benchmarks.md).

## Giới hạn và những điều chưa biết

- **Đã xác minh:** Bảng điểm chính thức kết hợp các điểm chuẩn công khai và nội bộ, và
  một số kết quả tác nhân dựa vào khung nội bộ hoặc tập hợp con nhiệm vụ đã sửa đổi.
- **Đã xác minh:** Các bảng Qwen3 suy nghĩ/không suy nghĩ, Qwen3.5 và Qwen3.6 không
  tất cả đều có chung giao thức giải mã hoặc phiên bản chuẩn.
- **Không xác định:** Mức độ chồng chéo dữ liệu đào tạo cho mọi điểm chuẩn công khai là
  không được tiết lộ. Điểm cao có thể kết hợp khái quát hóa với ghi nhớ.
- **Suy ra:** Đối với lựa chọn đường trục VLA, điểm nối đất và video cao hơn
  liên quan đến nhiệm vụ hơn một sự khác biệt nhỏ trong MMLU-Pro, nhưng mối quan hệ cuối cùng
  phải được đo lường thông qua đánh giá chính sách cấp dưới.

## Nguồn

- Đội Qwen. *Báo cáo kỹ thuật Qwen3*, arXiv:2505.09388, 2025.
  [Giấy][qwen3] · [PDF cục bộ][qwen3-local]
- Đội Qwen. *Thẻ mẫu Qwen3.5-27B*, truy cập 22-07-2026.
  [Thẻ mẫu] [qwen35-card]
- Đội Qwen. *Thẻ mẫu Qwen3.6-27B*, truy cập 22-07-2026.
  [Thẻ mẫu] [qwen36-card]
- Bài và cộng sự. *Báo cáo kỹ thuật Qwen2.5-VL*, arXiv:2502.13923, 2025.
  [Giấy][qwen25-vl] · [PDF cục bộ][qwen25-vl-local]
- Vương và cộng sự. *Qwen-VLA*, arXiv:2605.30280v2, 2026.
  [Giấy][qwen-vla]
- Chu và cộng sự. *IFEval*. [Giấy][ifeval] · [Triển khai chính thức][ifeval-code]

[qwen3]: https://arxiv.org/abs/2505.09388
[qwen3-local]: ../../../papers/05-gwen/gwen-overview/qwen3_technical_report_2505.09388.pdf
[qwen35-card]: https://huggingface.co/Qwen/Qwen3.5-27B
[qwen36-card]: https://huggingface.co/Qwen/Qwen3.6-27B
[qwen25-vl]: https://arxiv.org/abs/2502.13923
[qwen25-vl-local]: ../../../papers/05-gwen/gwen-overview/qwen2.5_vl_2502.13923.pdf
[qwen-vla]: https://arxiv.org/abs/2605.30280
[nếu có]: https://arxiv.org/abs/2311.07911
[mã ifeval]: https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_lib.py
