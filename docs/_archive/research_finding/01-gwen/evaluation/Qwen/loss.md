# Các hàm loss dùng trong mô hình Qwen

> **Câu hỏi:** Những mục tiêu huấn luyện nào đã được công bố cho backbone ngôn ngữ
> và ngôn ngữ-thị giác Qwen, và nên diễn giải giá trị loss của chúng thế nào?
>
> **Phạm vi:** Qwen3 và Qwen2.5-VL, với hành động Qwen-VLA và mục tiêu RL được giữ nguyên
> trong [VLA/loss.md](../VLA/loss.md). Nghiên cứu được kiểm tra vào ngày 21-07-2026.

## Câu trả lời ngắn

Các báo cáo công khai của Qwen mô tả trình tự các giai đoạn đào tạo một cách đầy đủ hơn
hơn là họ tiết lộ chính xác tổn thất sản xuất. Các kết luận an toàn là:

- Các mô hình Qwen là trình tạo tự hồi quy, nhưng báo cáo Qwen3 không in
  mục tiêu đầy đủ của mã thông báo tiếp theo, mặt nạ nhãn, chuẩn hóa hoặc hệ số;
- Quá trình đào tạo trước Qwen3 MoE bổ sung thêm tổn thất cân bằng tải hàng loạt toàn cầu mà không có
  công thức hoặc trọng lượng đã công bố;
- Qwen3 sau đào tạo sử dụng khởi động nguội có giám sát, GRPO lý luận RL, RL chung,
  và cả phản ứng và chưng cất logit;
- Qwen2.5-VL sử dụng đào tạo trước đa phương thức, SFT và DPO, nhưng báo cáo của nó thì không
  công bố các phương trình hoặc hệ số DPO tương ứng;
- giá trị tổn thất thô không thể so sánh được giữa các họ mô hình, giai đoạn hoặc bộ dữ liệu
  trừ khi mã thông báo mục tiêu và mức giảm giống hệt nhau.

## Mất mô hình ngôn ngữ có nghĩa là gì

Đối với mô hình tự hồi quy, mục tiêu khái niệm thông thường là log âm
khả năng vượt qua mã thông báo mục tiêu:

$$
\mathcal{L}_{\text{NLL}}
= -\sum_{i \in \mathcal{T}}
\log p_\theta(w_i \mid w_{<i}, x),
$$

trong đó $x$ là bối cảnh hiển thị và $\mathcal{T}$ là tập hợp các mã thông báo
nhận nhãn. Tổn hao thấp hơn có nghĩa là mô hình gán nhiều xác suất hơn cho
mã thông báo tham chiếu theo mặt nạ và tiền xử lý chính xác đó.

Phương trình này là một định nghĩa giải thích, **không phải là khẳng định rằng báo cáo Qwen3
công bố mức giảm chính xác này**. Cho dù mã thông báo hình ảnh, mã thông báo nhắc nhở, phần đệm,
dấu vết công cụ hoặc chỉ bao gồm mã thông báo trợ lý có thể thay đổi đáng kể
con số. Sự bối rối, khi thích hợp, là $\exp(\text{mean token NLL})$; nó là
chỉ có thể so sánh được theo cùng một chính sách mã thông báo, kho dữ liệu và mặt nạ.

## Qwen3

### Luyện tập cơ bản

Báo cáo kỹ thuật Qwen3 mô tả ba giai đoạn đào tạo trước: chung
đào tạo trước hơn 30 nghìn tỷ token, đào tạo theo định hướng lý luận về
khoảng 5 nghìn tỷ token và đào tạo theo ngữ cảnh dài hạn trên hàng trăm tỷ
mã thông báo. Đối với các mô hình hỗn hợp các chuyên gia, nó đặt tên rõ ràng là **global-batch
mất cân bằng tải**, nhằm mục đích ngăn chặn việc định tuyến mã thông báo bị sập vào một
số lượng nhỏ các chuyên gia và để khuyến khích chuyên môn hóa.

Báo cáo **không** công bố:

- phương trình mất mát mô hình ngôn ngữ nhân quả chính xác hoặc mặt nạ mã thông báo;
- công thức và hệ số cân bằng tải;
- mất mát z, thuật ngữ entropy của bộ định tuyến hoặc bất kỳ bộ điều chỉnh bộ định tuyến nào khác;
- trọng lượng giảm sản xuất theo từng giai đoạn cụ thể.

Những chi tiết đó sẽ vẫn là `Unknown`; nhập chúng từ MoE khác
việc triển khai sẽ biến một khả năng chung thành một khả năng cụ thể của Qwen sai
sự thật. [Báo cáo Qwen3, Phần 2–3] [qwen3]

### Khởi động nguội và kết hợp chế độ tư duy

Khởi đầu nguội theo chuỗi suy nghĩ dài tập về các phản ứng lập luận có chọn lọc. Sau đó,
Chế độ Tư duy Fusion thực hiện việc tinh chỉnh liên tục có giám sát trên cả hai phương diện tư duy
và những ví dụ không suy nghĩ. Những giai đoạn này cho chúng ta biết mục tiêu nào đã được học, nhưng
báo cáo không nêu rõ phương trình khả năng được giám sát, mã thông báo nào được
bị che hoặc cách hai chế độ được tính trọng số. [Báo cáo Qwen3, Phần 4.1 và
4.3][qwen3]

### Lý luận RL

Qwen3 áp dụng GRPO cho 3.995 cặp trình xác minh truy vấn. Người xác minh cung cấp phần thưởng
cho các giải pháp được tạo và chính sách được tối ưu hóa từ nhiều lần triển khai.
Báo cáo nêu rõ rằng việc đào tạo sử dụng các đợt lớn và lấy mẫu ngoài chính sách, nhưng
không công bố phương trình GRPO sản xuất, hệ số KL/entropy hoặc chính xác
tổng hợp khen thưởng.

Cải tiến được báo cáo trên AIME là kết quả đánh giá, không phải là giá trị tổn thất. Nó
cho thấy giai đoạn tối ưu hóa đã thay đổi hành vi; nó không tiết lộ
quy mô hoặc độ ổn định của vật kính RL bên trong. [Báo cáo Qwen3, Phần
4.2][qwen3]

### Phần thưởng chung RL

General RL bao gồm hơn 20 loại nhiệm vụ và sử dụng ba nhóm phần thưởng:

1. phần thưởng dựa trên quy tắc cho các kết quả có thể được kiểm tra một cách xác định;
2. tính điểm dựa trên mô hình với câu trả lời tham khảo, sử dụng Qwen2.5-72B-Instruct;
3. một mô hình phần thưởng vô hướng đã học khi không có câu trả lời tham khảo.

Phần thưởng là tín hiệu đào tạo, không phải là thước đo chuẩn và không nhất thiết là
số lượng tương đương với tổn thất hợp đồng. Báo cáo không tiết lộ trình tối ưu hóa
trọng số khách quan hoặc tương đối của các nhóm phần thưởng này. [Báo cáo Qwen3,
Mục 4.4][qwen3]

### Chưng cất mạnh đến yếu

Qwen3 sử dụng hai hình thức chưng cất khác nhau cho các mẫu nhỏ hơn:

- **chưng cất ngoài chính sách:** học sinh học hỏi từ các phản hồi do một
  giáo viên lớn hơn trong các phương thức tư duy và không suy nghĩ; mất phản ứng chính xác là
  không được in;
- **chưng cất theo chính sách:** học sinh tạo ra trình tự riêng của mình, sau đó
  nhật ký mã thông báo được căn chỉnh với Qwen3-32B hoặc Qwen3-235B-A22B bằng cách giảm thiểu KL
  sự khác biệt.

Báo cáo cho biết cần giảm thiểu KL giữa nhật ký của giáo viên và học sinh, nhưng không
xác định hướng, nhiệt độ, giảm mã thông báo của $D_{\mathrm{KL}}(p\|q)$ hoặc
hệ số. Do đó, sẽ an toàn hơn nếu không xây dựng lại phương trình. Giấy
báo cáo rằng việc chưng cất theo chính sách đã mang lại kết quả mô hình nhẹ mạnh hơn
với thời gian GPU ít hơn nhiều so với việc lặp lại toàn bộ quy trình RL. [Báo cáo Qwen3,
Mục 4.5 và Bảng 21][qwen3]

## Qwen2.5-VL

### Đào tạo trước đa phương thức

Qwen2.5-VL có ba giai đoạn được ghi lại:

| Giai đoạn | Các mô-đun có thể đào tạo | Quy mô đào tạo | Độ dài chuỗi |
|---|---|---:|---:|
| Đào tạo trước về thị giác | Chỉ ViT | Mã thông báo 1,5T | 8.192 |
| Đào tạo trước đa phương thức | ViT và LLM | Mã thông báo 2,0T | 8.192 |
| Đào tạo trước ngữ cảnh dài | ViT và LLM | Mã thông báo 0,6T | 32.768 |

Báo cáo mô tả dữ liệu và lịch trình nhưng không in riêng biệt
các phương trình tương phản, căn chỉnh, mã thông báo tiếp theo hoặc mục tiêu phụ trợ và chúng
trọng lượng. Do đó, sẽ không an toàn khi khẳng định rằng tổn thất CLIP-style hoặc một tổn thất cụ thể
thuật ngữ căn chỉnh ngôn ngữ tầm nhìn được sử dụng chỉ vì những mất mát đó là phổ biến
ở nơi khác. [Báo cáo Qwen2.5-VL, Phần 2.2 và Bảng 2] [qwen25-vl]

### SFT và DPO

Sau đào tạo đầu tiên sử dụng khoảng hai triệu ví dụ được giám sát, khoảng
nửa văn bản và nửa đa phương thức, được đăng nhiều kỳ trong ChatML. Sau đó nó áp dụng trực tiếp
Tối ưu hóa tùy chọn cho các cặp ưu tiên hình ảnh-văn bản và văn bản thuần túy. ViT
bị đóng băng trong cả hai giai đoạn.

Báo cáo không công bố phương trình SFT, phương trình DPO, DPO $\beta$,
chi tiết chính sách tham chiếu hoặc trọng số thành phần. Các mô hình khen thưởng được đề cập ở
điểm phần dữ liệu và lọc dữ liệu QA ứng viên trước SFT; họ không phải
bằng chứng về một giai đoạn RLHF riêng biệt. [Báo cáo Qwen2.5-VL, Phần 2.3] [qwen25-vl]

## Cách ghi và so sánh tổn thất

Lưu trữ tổn thất thành phần riêng biệt thay vì chỉ một đại lượng vô hướng kết hợp:

| Lĩnh vực | Tại sao nó quan trọng |
|---|---|
| mục tiêu và phương trình trên giấy/tham khảo | Ngăn chặn hai số lượng không giống nhau chia sẻ một tên |
| tập dữ liệu và hỗn hợp mẫu | Đổi token dễ và khó thay đổi thang đo NLL |
| tokenizer và sửa đổi từ vựng | Thay đổi số lượng mã thông báo và không gian xác suất |
| mặt nạ và mẫu số có nhãn mã thông báo | Phân biệt tất cả mã thông báo với tính trung bình chỉ phản hồi |
| đóng gói và cắt bớt trình tự | Thay đổi mục tiêu nào nhập mức trung bình |
| hệ số tổn thất và tỷ lệ lấy mẫu | Cả hai đều ảnh hưởng đến sự đóng góp của độ dốc |
| chế độ huấn luyện/xác nhận và bước kiểm tra | Làm cho các đường cong có thể so sánh được theo thời gian |
| điểm chuẩn hạ lưu | Kiểm tra xem tổn thất thấp hơn có cải thiện hành vi hữu ích hay không |

Không so sánh mã thông báo Qwen3 NLL với mục tiêu DPO, đại diện GRPO, phần thưởng,
hoặc hành động Qwen-VLA MSE. Ngay cả hai lần mất mã thông báo cũng có thể khác nhau do mã thông báo,
mặt nạ nhãn và thành phần dữ liệu. Đường cong tổn thất chẩn đoán việc tối ưu hóa theo một
hợp đồng cố định; [metrics.md](metrics.md) xác định các hàm điểm được báo cáo,
trong khi [benchmarks.md](benchmarks.md) ghi lại các kết quả và giao thức đánh giá.

## Nguồn

- Đội Qwen. *Báo cáo kỹ thuật Qwen3*. [Giấy][qwen3] · [PDF cục bộ][qwen3-local]
- Bài và cộng sự. *Báo cáo kỹ thuật Qwen2.5-VL*. [Giấy][qwen25-vl] ·
  [PDF cục bộ][qwen25-vl-local]

[qwen3]: https://arxiv.org/abs/2505.09388
[qwen3-local]: ../../../papers/05-gwen/gwen-overview/qwen3_technical_report_2505.09388.pdf
[qwen25-vl]: https://arxiv.org/abs/2502.13923
[qwen25-vl-local]: ../../../papers/05-gwen/gwen-overview/qwen2.5_vl_2502.13923.pdf
