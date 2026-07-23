# Hồi quy liên tục và dự đoán hành động song song trực tiếp

> **Phạm vi.** Dự đoán một lần các hành động cấp thấp liên tục hoặc đoạn hành động.
> “Kiểu RT-1” được xem xét rõ vì RT-1 gốc thường bị xếp nhầm vào họ này.
> Nguồn được kiểm tra ngày 2026-07-21.

## Ý tưởng cốt lõi

Bộ giải mã hồi quy liên tục ánh xạ trực tiếp bối cảnh đa phương thức tới giá trị thực
hành động:

```text
bối cảnh h -> đầu hành động song song -> A_hat in R^(H x D)
```

Ở đây, `H` là phạm vi dự đoán và `D` là chiều hành động. Một điểm chung
mục tiêu đào tạo là hồi quy L1:

```text
L = trung bình(|A_hat - A|)
```

Không giống như các mô hình mã thông báo tự hồi quy, chính sách này không tuần tự hóa mọi hành động
thứ nguyên thành các mã thông báo từ vựng. Không giống như khuếch tán hay dòng chảy, suy luận không
bắt đầu từ tiếng ồn và liên tục tinh chỉnh đoạn. Tất cả các giá trị `H x D` có thể
được tạo ra trong một lần chuyển tiếp.

## Hiệu chỉnh phân loại: RT-1 không phải là hồi quy liên tục

**Đã xác minh.** RT-1 dự đoán trực tiếp lệnh robot cho điều khiển hiện tại
bước, nhưng phân phối đầu ra của nó là phân loại. Bài báo rời rạc hóa từng
kích thước hành động của cánh tay và cơ sở thành 256 thùng và đoàn tàu thống nhất với các phân loại
entropy chéo. Bộ biến áp chỉ có bộ giải mã của nó tạo ra các đầu ra hành động với một
Chính sách tham số 35M chạy ở tần số 3 Hz. [RT-1, §3.3 và Hình 3](https://arxiv.org/abs/2212.06817)

Do đó, mô tả chính xác là:

```text
RT-1 = dự đoán phân loại trực tiếp, song song, theo chiều
     != hồi quy liên tục
     != tự hồi quy kiểu ngôn ngữ trên một chuỗi hành động
```

RT-1 vẫn là tổ tiên cấu trúc hữu ích của các đầu hồi quy trực tiếp: cả hai
tránh một bộ lấy mẫu sinh học lặp đi lặp lại và ngay lập tức tạo ra một đầu ra điều khiển.
Sự khác biệt là sự phân phối đầu ra và tổn thất.

RT-1 cũng đánh giá công thức liên tục như một phương pháp cắt bỏ: một công thức đa biến
đầu ra hành động bình thường được đào tạo với MSE. Nó hoạt động kém hơn đáng kể so với
đầu phân loại 256-bin được chọn trong nhiệm vụ nhìn thấy, nhiệm vụ không nhìn thấy của tờ giấy đó,
và đánh giá độ bền. Đây là bằng chứng cho việc thiết lập RT-1, không phải chung chung
bằng chứng cho thấy hồi quy kém hơn; kết quả OpenVLA-OFT sau này cho thấy rằng
lựa chọn xương sống, chunking, song song, dữ liệu và L1-versus-MSE
thay đổi kết quả. [RT-1, Phụ lục D.4 và Bảng 13](https://arxiv.org/abs/2212.06817)

## Đại diện VLA hiện đại: OpenVLA-OFT

OpenVLA-OFT cung cấp một phiên bản đã được xác minh rõ ràng của dòng này. Nó thích nghi với
Đường trục OpenVLA sử dụng bốn lựa chọn được liên kết:

1. thay thế giải mã mã thông báo hành động tuần tự bằng giải mã song song;
2. dự đoán một đoạn hành động gồm nhiều bước;
3. sử dụng các giá trị hành động liên tục thay vì các ký hiệu 256-bin;
4. tối ưu hóa mục tiêu hồi quy L1.

Công thức OFT của bài báo báo cáo các đoạn 8 bước trong LIBERO và các đoạn 25 bước cho
cài đặt ALOHA thực. Đoạn hành động được phát ra trong một đánh giá mô hình,
tách tần số suy luận thần kinh khỏi việc thực thi lệnh của robot
Tính thường xuyên. [OpenVLA-OFT, §I và §V-E](https://arxiv.org/abs/2502.19645)

```text
mã thông báo hình ảnh/ngôn ngữ + hình ảnh cổ tay/cảm nhận quyền sở hữu tùy chọn
                         |
                         v
                  Đường trục OpenVLA
                         |
                  khe hành động hai chiều
                         |
                         v
                Đầu hành động liên tục MLP
                         |
                         v
                  Đoạn hành động bước H
```

Đây là “trực tiếp” ở cấp độ người tạo hành động; hệ thống vẫn không chuẩn hóa
dự đoán và gửi nó qua bộ điều khiển dành riêng cho robot sau đó.

## Tại sao sử dụng nó?

**Các lợi ích đã được xác minh trong cài đặt được đánh giá.** OpenVLA-OFT nhận thấy rằng
công thức kết hợp song song/phân đoạn/liên tục/L1 tăng khả năng tạo hành động
thông lượng tăng gấp 26 lần so với OpenVLA cơ bản khi thiết lập LIBERO. Bởi vì không có
vòng lặp tự hồi quy hoặc vòng lặp khử nhiễu, độ trễ trực tiếp tăng ít hơn
với số lượng giá trị hành động đầu ra. [Dự án và bài viết OpenVLA-OFT](https://openvla-oft.github.io/)

Ưu điểm kỹ thuật là:

- suy luận một lần và mục tiêu được giám sát đơn giản;
- kết quả đầu ra có giá trị thực chính xác mà không có lỗi lượng tử hóa bin;
- các khe hành động song song giúp việc phân chia khối trở nên đơn giản;
- không có lịch trình tiếng ồn khuếch tán, bộ lấy mẫu hoặc siêu tham số bước tích hợp.

## Nó từ bỏ cái gì?

Hồi quy L1 hoặc MSE tạo ra ước tính điểm. Nếu các cuộc biểu tình có
một số tương lai hợp lệ không tương thích cho cùng một quan sát, một cách đơn giản
bộ hồi quy có thể chọn một chế độ hoặc dự đoán một sự thỏa hiệp thay vì một cách rõ ràng
mô hình phân phối có điều kiện đầy đủ. Cụ thể các tác giả OpenVLA-OFT
lưu ý rằng L1 có thể học chế độ trung bình và không cho rằng hồi quy là
nói chung là tốt hơn so với khuếch tán. [Thảo luận OpenVLA-OFT về L1 so với khuếch tán](https://openvla-oft.github.io/)

Các giới hạn khác là:

- một đoạn dài có thể trở nên cũ kỹ sau khi cảnh thay đổi;
- việc thực thi vòng lặp mở quá nhiều đoạn sẽ làm giảm phản hồi;
- Chỉ riêng lựa chọn L1/MSE không xác định ngữ nghĩa hành động hoặc chuẩn hóa;
- lợi ích từ OpenVLA-OFT kết hợp một số thay đổi, do đó chúng không cô lập
  lợi ích của việc hồi quy liên tục.

## Khi nhãn phù hợp

Chỉ gọi bộ giải mã **hồi quy liên tục** khi tất cả những điều sau đây được đáp ứng
ĐÚNG VẬY:

- mục tiêu đã học là một tensor hành động có giá trị thực;
- sự mất mát so sánh trực tiếp các giá trị được dự đoán và chứng minh, chẳng hạn như L1 hoặc
  MSE;
- quá trình triển khai sử dụng một dự đoán chuyển tiếp nguồn cấp dữ liệu thay vì từng mã thông báo
  lấy mẫu hoặc khử nhiễu/tích hợp dòng chảy lặp đi lặp lại.

Chính sách xuất ra chỉ mục bin sau đó được chuyển đổi thành float vẫn là chính sách
chính sách phân loại. Mô hình khuếch tán/dòng chảy kết thúc ở một tensor liên tục là
vẫn là một chính sách sinh sản lặp đi lặp lại, không phải là hồi quy trực tiếp.

## Nguồn

- Brohan và cộng sự. *RT-1: Máy biến áp robot để điều khiển trong thế giới thực ở quy mô*,
  §3.3 và Phụ lục D.4, arXiv:2212.06817v2.
  [Giấy](https://arxiv.org/abs/2212.06817)
- Kim, Finn và Liang. *Tinh chỉnh các mô hình hành động-ngôn ngữ-thị giác: Tối ưu hóa
  Tốc độ và Thành công*, arXiv:2502.19645, 2025.
  [Giấy](https://arxiv.org/abs/2502.19645) ·
  [Dự án chính thức](https://openvla-oft.github.io/) ·
  [Mã chính thức](https://github.com/moojink/openvla-oft)
