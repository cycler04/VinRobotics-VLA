# Diffusion hoặc flow với bộ giải mã hành động gọn

> **Phạm vi.** Kiến trúc trong đó đường trục đa phương thức tính toán bối cảnh một lần
> và một mạng có điều kiện tương đối nhỏ thực hiện lặp liên tục
> thế hệ hành động. Chính sách phổ biến cung cấp cơ chế cơ bản; Octo là
> ví dụ chính sách robot tổng quát rõ ràng nhất về đầu khuếch tán nhỏ gọn. Nguồn
> đã kiểm tra 2026-07-21.

## Ý tưởng cốt lõi

Thay vì dự đoán một ước tính hành động, bộ giải mã mô hình hóa một điều kiện
phân phối trên toàn bộ đoạn hành động có giá trị thực. Suy luận bắt đầu bằng một
tensor ngẫu nhiên và tinh chỉnh nó qua một số bước:

```text
quan sát/nhiệm vụ -> xương sống -> bối cảnh nhỏ gọn e (được tính một lần)
                                      |
Tiếng ồn hành động Gaussian x_K ------------+
        -> bộ khử nhiễu nhỏ (x_K, e, K)
        -> bộ khử nhiễu nhỏ (x_K-1, e, K-1)
        -> ...
        -> đoạn hành động liên tục x_0
```

“Nhỏ gọn” mô tả nơi năng lực của mô hình tồn tại. Ngôn ngữ trực quan hoặc chính sách
xương sống có thể là đáng kể, trong khi tính toán khử nhiễu lặp đi lặp lại là một
mạng MLP, U-Net nhỏ hoặc mạng chuỗi thời gian khiêm tốn. Không có tham số tiêu chuẩn
ngưỡng chính thức phân tách đầu nhỏ gọn khỏi DiT lớn.

## Chính sách phổ biến: cơ chế phổ biến hành động cơ bản

Chính sách khuếch tán thể hiện chính sách thị giác như một phương pháp khử nhiễu có điều kiện
quá trình khuếch tán qua các chuỗi hành động. Thiết kế chính của nó kết hợp:

- điều hòa thị giác;
- Máy biến áp khuếch tán chuỗi thời gian hoặc U-Net có điều kiện;
- dự đoán về một chân trời hành động gồm nhiều bước;
- điều khiển rút lui, trong đó chỉ một phần của đoạn dự đoán được thực thi
  trước khi quan sát tiếp theo gây ra việc lập kế hoạch lại.

Việc huấn luyện làm xáo trộn các hành động trình diễn với nhiễu Gaussian và học các
trường khử nhiễu/điểm có điều kiện. Suy luận sử dụng nhiều cập nhật khử nhiễu,
vì vậy đoạn được lấy mẫu có tính mạch lạc toàn cục chứ không phải là một tập hợp độc lập
dấu thời gian hồi quy. Các tác giả thúc đẩy điều này cho các cuộc biểu tình đa phương thức,
chuỗi hành động chiều cao và đào tạo ổn định.
[Chính sách phổ biến, §§3-4](https://arxiv.org/abs/2303.04137)

**Ranh giới.** Chính sách phổ biến ban đầu là một phương pháp học tập bắt chước vận động thị giác
chính sách, không nhất thiết phải là VLA được xây dựng trên mô hình ngôn ngữ được đào tạo trước trên Internet.
Nó thuộc về nơi này vì các đầu hành động VLA sau này sẽ sử dụng lại điều kiện của nó
mô hình khuếch tán hành động.

Từ "nhỏ gọn" phải được kiểm tra dựa trên cấu hình thực tế. các
Bộ khử nhiễu biến áp theo chuỗi thời gian của bài báo là 9M tham số cho hầu hết các tác vụ và
80M cho Nhà bếp/Push-T thực, nhưng một số bộ khử nhiễu tạm thời-CNN đã được công bố
lớn hơn nhiều. Do đó, sẽ không chính xác nếu gọi mọi Chính sách phổ biến là
biến thể nhỏ gọn chỉ từ tên phương thức. [Chính sách phổ biến, Phụ lục A.4](https://arxiv.org/abs/2303.04137)

## Octo: xương sống một lần, đầu khuếch tán nhỏ liên tục

Octo làm cho việc phân tách đầu nén trở nên rõ ràng. Một tác vụ xử lý Transformer
và mã thông báo quan sát và tạo ra các phần nhúng đọc được. Hành động nhẹ nhàng
head sau đó tạo ra một đoạn hành động liên tục với chức năng khử nhiễu DDPM-style. Chỉ một
cần có một đường truyền trục máy biến áp cho mỗi dự đoán hành động; tất cả
các bước lặp đi lặp lại chạy bên trong phần đầu nhỏ. [Tháng 10, §III-A và §III-C](https://arxiv.org/abs/2405.12213)

Cấu hình được công bố sử dụng MLP ba lớp với kích thước ẩn 256,
kết nối dư, chuẩn hóa lớp, lịch trình nhiễu cosin và 20
các bước khuếch tán Đây là một ví dụ sắc nét hơn về “bộ giải mã nhỏ gọn” hơn là chỉ đơn thuần
gọi mọi chính sách phổ biến là nhỏ. [Tháng 10, Phụ lục D](https://arxiv.org/abs/2405.12213)

```text
lịch sử tác vụ/hình ảnh -> Octo Transformer -> nhúng phần đọc ra
                                             |
                            đoạn ồn ào -> Bộ khử nhiễu MLP 3 lớp x 20
                                             |
                                             v
                                  đoạn hành động liên tục
```

Bởi vì giao diện đọc là mô-đun nên không gian hành động mới có thể nhận được một giao diện mới
đầu trong khi hầu hết các trọng lượng xương sống được luyện trước vẫn còn nguyên.

## Ví dụ về luồng nhỏ gọn: SmolVLA

SmolVLA làm cho phiên bản dòng chảy của mẫu này trở nên cụ thể. Một SmolVLM-2 đông lạnh
xương sống tạo ra các tính năng ngữ cảnh và tham số khoảng 100M
Chuyên gia về máy biến áp phù hợp với dòng chảy có điều kiện dự đoán các khối 50 hành động bằng cách sử dụng
mười bước chảy. Chính sách hoàn chỉnh có khoảng 450M thông số. Sự chú ý chéo
nhập các tính năng VLM trong khi tính năng tự chú ý nhân quả xử lý các mã thông báo hành động, do đó
mạng hành động vẫn là một mô-đun có thể huấn luyện riêng biệt thay vì được chia sẻ
Chuyên gia kiểu π0 bên trong một Transformer. [SmolVLA, §3 và §4.3](https://arxiv.org/abs/2506.01844)

Đây là một ví dụ về tỷ lệ hữu ích, không phải là bằng chứng cho thấy 100M là mức cắt chuẩn. các
bài báo cũng giới hạn bằng chứng của nó ở những nhiệm vụ tương đối đơn giản, có thời gian ngắn và
xác định hành vi dài hạn là công việc trong tương lai.

## Kết hợp khuếch tán và dòng chảy

Các cơ chế này có liên quan nhưng không nên được sử dụng làm từ đồng nghĩa chính xác.

| Bất động sản | Đầu khuếch tán DDPM-style | Đầu khớp dòng chảy |
| --- | --- | --- |
| Đối tượng đã học | Tiếng ồn, điểm số hoặc hướng khử nhiễu theo lịch trình tiếng ồn | Trường vận tốc dọc theo đường xác suất đã chọn |
| Thế hệ | Khuếch tán ngược/cập nhật DDIM-style | Tích hợp số của ODE, thường là các bước Euler |
| Đầu ra | Mẫu hoặc đoạn hành động liên tục | Mẫu hoặc đoạn hành động liên tục |
| Chi phí chia sẻ | Một số cuộc gọi đến mạng hành động | Một số cuộc gọi đến mạng hành động |

Đầu dòng nhỏ gọn phù hợp với họ này khi mạng vận tốc nhỏ
đọc có điều kiện trên các tính năng xương sống. π0 được ghi lại riêng biệt vì
mạng lưới hành động của nó là một chuyên gia Máy biến áp có tham số 300M xen kẽ với
VLM, không chỉ đơn thuần là một kết quả đọc nông. Nhìn thấy
[Chuyên gia về máy biến áp phù hợp với dòng chảy](04_flow_matching_transformer_expert.md).

## Tại sao nên sử dụng đầu phát điện nhỏ gọn?

- Đầu ra liên tục tránh lượng tử hóa trên mỗi thùng.
- Việc tạo khối chung có thể thể hiện cấu trúc thời gian tương quan.
- Việc lấy mẫu có thể thể hiện một số hành vi hợp lệ thay vì thu gọn chúng
  thành ước lượng một điểm.
- Bối cảnh xương sống có thể được lưu trữ trong khi phần đầu nhỏ thực hiện lặp lại
  các bước.
- Phần đầu có thể được thay thế bằng một phương án mới mà không nhất thiết phải xây dựng lại
  toàn bộ đường trục đa phương thức.

## Giới hạn và chế độ thất bại

- Lấy mẫu lặp lại tăng thêm độ trễ so với hồi quy một lần.
- Quá nhiều bước lấy mẫu sẽ làm giảm tỷ lệ lập kế hoạch lại có thể đạt được; quá ít có thể
  làm tổn thương chất lượng mẫu.
- Một vectơ ngữ cảnh nhỏ gọn có thể trở thành nút cổ chai thông tin. Dita là
  được đề xuất cụ thể xung quanh giả thuyết rằng một cái đầu nhỏ xíu phụ thuộc vào
  các phần nhúng được hợp nhất sớm là không đủ cho phương án chéo không đồng nhất
  dữ liệu. [Dita, §§1 và 3](https://arxiv.org/abs/2503.19757)
- Các lựa chọn về chân trời hành động và chân trời thực thi đánh đổi tính nhất quán theo thời gian đối với
  khả năng đáp ứng.
- Năng lực mô hình hóa đa phương thức không đảm bảo an toàn hoặc có giá trị về mặt vật lý
  lệnh; các ràng buộc không chuẩn hóa và kiểm soát vẫn ở bên ngoài.

## Những gì được xác minh so với suy luận?

**Đã xác minh:** Chính sách phổ biến và Octo khử nhiễu lặp đi lặp lại hành động liên tục
khối; Octo cô lập vòng lặp này trong đầu hành động MLP ba lớp sau một
đường xương sống.

**Loại kỹ thuật được suy luận:** “bộ giải mã dòng hoặc khuếch tán nhỏ gọn” là một
nhãn phân loại hữu ích, nhưng không có bài báo nào xác định ranh giới kích thước phổ quát.
Do đó, việc so sánh bộ giải mã nhỏ gọn và lớn sẽ báo cáo tham số thực tế
đếm và điều hòa cấu trúc liên kết thay vì chỉ dựa vào nhãn.

## Nguồn

- Chí và cộng sự. *Chính sách phổ biến: Học chính sách thị giác thông qua hành động
  Khuếch tán*, arXiv:2303.04137v5, 2024.
  [Giấy](https://arxiv.org/abs/2303.04137) ·
  [Dự án chính thức](https://diffusion-policy.cs.columbia.edu/)
- Nhóm người mẫu Octo và cộng sự. *Tháng 10: Chính sách về robot tổng quát nguồn mở*,
  arXiv:2405.12213, 2024. [Giấy](https://arxiv.org/abs/2405.12213) ·
  [Dự án chính thức](https://octo-models.github.io/)
- Shukor và cộng sự. *SmolVLA: Mô hình Hành động-Ngôn ngữ-Tầm nhìn dành cho Giá cả phải chăng và
  Robot hiệu quả*, arXiv:2506.01844, 2025.
  [Giấy](https://arxiv.org/abs/2506.01844) ·
  [Bản phát hành chính thức](https://huggingface.co/blog/smolvla)
- Hou và cộng sự. *Dita: Máy biến áp khuếch tán tỷ lệ dành cho nhà tổng hợp
  Chính sách Tầm nhìn-Ngôn ngữ-Hành động*, arXiv:2503.19757v2, 2025.
  [Giấy](https://arxiv.org/abs/2503.19757)
