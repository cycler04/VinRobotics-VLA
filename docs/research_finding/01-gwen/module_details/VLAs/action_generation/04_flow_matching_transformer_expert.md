# Action expert Transformer dùng flow matching

> **Phạm vi.** VLM đã huấn luyện trước cung cấp ngữ cảnh ngữ nghĩa, còn một bộ
> trọng số Transformer riêng chuyên sinh hành động liên tục bằng flow matching.
> Các mô hình đại diện là π0 và π0,5. Nguồn được kiểm tra ngày 2026-07-21.

## Ý tưởng cốt lõi

VLM và trình tạo hành động chia sẻ bối cảnh chú ý nhưng thực hiện các công việc khác nhau:

```text
hình ảnh + hướng dẫn -> trọng lượng VLM được huấn luyện trước ------+
trạng thái robot -----------------------------------------+--> bối cảnh token được chia sẻ
đoạn hành động nhiễu -> trọng số của action expert ----------------+
                                                       |
                                              trường vận tốc v_theta
                                                       |
                                            tích hợp ODE lặp đi lặp lại
                                                       |
                                                       v
                                              đoạn hành động liên tục
```

Chuyên gia không chỉ là một phép chiếu tuyến tính cuối. Đây là một bộ tham số
Transformer riêng cho các vị trí hành động, tương tự chuyên gia theo modality trong
kiến trúc mixture-of-experts. Token hành động attention hai chiều trong
đoạn và tiền tố đa phương thức.

Chính xác hơn, π0 là một phép tính Transformer với hai bộ định tuyến token
trọng số: token hình ảnh/ngôn ngữ sử dụng chuyên gia do PaliGemma khởi tạo, trong khi trạng thái
và token hành động ồn ào sử dụng chuyên gia hành động nhỏ hơn. Họ trao đổi
thông tin thông qua sự tự chú ý được chia sẻ. Do đó, “chuyên gia riêng biệt” không
có nghĩa là bộ mã hóa-giải mã tách rời chỉ được kết nối bởi một vectơ ngữ cảnh.

## Kiến trúc π0

π0 bắt đầu từ PaliGemma, VLM được huấn luyện trước tham số 3B và thêm khoảng 300M
các tham số chuyên gia hành động được khởi tạo ngẫu nhiên. Mô hình hoàn chỉnh có khoảng
Thông số 3.3B. Hình ảnh, ngôn ngữ và trạng thái proprioception hình thành nên
quan sát điều hòa; mục tiêu là chân trời của các hành động trong tương lai của `H=50`.
[π0, §IV](https://arxiv.org/abs/2410.24164)

Đối với đoạn `A` đã được chứng minh và nhiễu Gaussian `epsilon`, π0 lấy mẫu một luồng
thời gian `tau` và tạo điểm trung gian:

```text
A_tau = tau * A + (1 - tau) * epsilon
vận tốc mục tiêu = A - epsilon
```

Chuyên gia hành động tìm hiểu trường vận tốc có điều kiện từ đoạn nhiễu
về phía khối dữ liệu. Khi suy luận, nó bắt đầu ở nhiễu Gaussian và tích phân
từ `tau=0` đến `tau=1`. Việc triển khai được báo cáo sử dụng mười chuyển tiếp Euler
các bước (`delta=0.1`) và lưu trữ các khóa/giá trị chú ý tiền tố để mỗi bước
chỉ tính toán lại hậu tố hành động. [π0, §IV](https://arxiv.org/abs/2410.24164)

Điều này khác với đầu DDPM: mục tiêu đã học là tốc độ dòng chảy dọc theo một
đường dẫn xác suất rõ ràng và việc triển khai tích hợp ODE thay vì
theo chuỗi Markov có nhiễu ngược.

## Tại sao lại là chuyên gia thay vì token hành động?

Thiết kế π0 duy trì đường dẫn ngôn ngữ và nhận thức đã được huấn luyện trước của VLM trong khi
đưa ra các giá trị robot liên tục theo tính toán của riêng chúng. Bài viết sử dụng điều này để
dự đoán các khối tần số cao cho các tác vụ được đánh giá ở tần số lên tới 50 Hz. liên tục
việc tạo chung tránh tạo ra hàng trăm token từ vựng tương quan cho
một giây chuyển động khéo léo. [giấy π0](https://arxiv.org/abs/2410.24164)

Việc phân chia cũng cho phép phân bổ không đối xứng:

- VLM lớn hơn nhập kiến ​​thức ngữ nghĩa ở quy mô Internet;
- chuyên gia nhỏ hơn chuyên về cảm giác bản thể, hành động ồn ào và vận động
  độ chính xác;
- sự chú ý kết nối cả hai mà không buộc các giá trị động cơ thông qua một văn bản
  từ vựng.

## π0,5: đào tạo trước rời rạc, triển khai luồng

π0,5 là sự lai ghép có chủ ý. Giai đoạn đầu tiên rộng lớn của nó đại diện cho hành động của robot
với token FAST và đào tạo chúng với dự đoán token tiếp theo cùng với web,
các nhiệm vụ nền tảng và ngữ nghĩa cấp cao. Trong quá trình đào tạo sau, nó bổ sung thêm
Chuyên gia hành động kiểu π0 và mất lưu lượng cho các hành động cấp thấp liên tục.
[π0,5, §IV và Hình 3](https://arxiv.org/abs/2504.16054)

Khi suy luận, cùng một mô hình thực hiện hai thao tác giải mã khác nhau:

```text
nhiệm vụ tổng thể + quan sát
  -> giải mã văn bản tự hồi quy
  -> nhiệm vụ phụ cấp cao, ví dụ: "nhấc đĩa lên"
  -> chuyên gia về luồng điều kiện về nhiệm vụ con đó
  -> mười bước tích hợp dòng chảy
  -> đoạn hành động cấp thấp liên tục
```

Văn bản cấp cao được tạo ra ít thường xuyên hơn; chuyên gia hành động cung cấp
các khối điều khiển nhanh. do đó π0,5 không thể được phân loại chỉ từ
biểu diễn tiền huấn luyện: nó thuộc họ FAST/tự hồi quy trong quá trình
một phần của chương trình đào tạo và nhóm chuyên gia về dòng chảy trong quá trình triển khai ở cấp độ thấp.

## Mối quan hệ với Qwen-VLA

Không nên coi Qwen-VLA như một chuyên gia về Transformer kiểu π0 khác. Trong π0,
Mã thông báo VLM và token robot chọn các bộ trọng lượng khác nhau trong một mã chung
Tính toán Transformer. Qwen-VLA lần đầu tiên tính toán các trạng thái, dự án ẩn của VLM
chúng thành một DiT 16 khối riêng biệt, nối chúng với các token hành động ồn ào,
và chạy tính năng tự chú ý chung **bên trong bộ giải mã xuôi dòng**.

Bài báo Qwen-VLA đôi khi gọi mô-đun này là “chuyên gia hành động”, nhưng nó
ranh giới kiến ​​trúc là bộ giải mã hành động DiT chứ không phải định tuyến token của π0
trọng lượng chuyên gia. Do đó nó được ghi lại dưới
[Transformer khuếch tán lớn](05_large_diffusion_transformer.md), không phải là
đại diện của gia đình này. [Qwen-VLA, §§2.2-2.5](https://arxiv.org/abs/2605.30280)

## Điểm mạnh

- các khối liên tục, mạch lạc mà không cần lượng tử hóa token hành động;
- một phân phối có điều kiện tổng quát có thể biểu diễn nhiều giá trị hợp lý
  quỹ đạo;
- công suất dành riêng cho động cơ không yêu cầu thay thế đường dẫn VLM đã được huấn luyện trước;
- các tính năng tiền tố được lưu trong bộ nhớ đệm giúp giảm công việc lặp lại trong quá trình tích hợp;
- chuyên gia có thể sử dụng sự chú ý hành động hai chiều trong khi con đường ngôn ngữ
  vẫn có tính chất tự hồi quy.

## Chi phí và những câu hỏi chưa được giải quyết

- mười đánh giá của chuyên gia chậm hơn so với một biến hồi quy một lần ở mức bằng nhau cho mỗi lần vượt qua
  trị giá;
- số bước lưu lượng và bộ giải số ảnh hưởng đến độ trễ và độ chính xác;
- phân phối biểu cảm hơn chỉ hữu ích nếu các cuộc biểu tình thực sự
  chứa các chế độ có ý nghĩa thay vì nhiễu chú thích;
- Kết quả của π0 kết hợp kiến ​​trúc, dữ liệu đa hiện thân và dữ liệu trước/sau
  công thức đào tạo, để họ không cô lập các chuyên gia thiết kế;
- Tương tự như vậy, lợi ích của π0,5 không thể chỉ quy cho việc kết hợp luồng vì
  Việc đào tạo trước FAST và giám sát nhiệm vụ phụ cấp cao cũng thay đổi hệ thống.

**Thông báo trước về trạng thái triển khai.** Bài viết π0,5 mô tả hiện tượng tự hồi quy
tạo nhiệm vụ phụ cấp cao theo sau là chuyên gia về luồng. `openpi` công cộng
README, tại phiên bản được kiểm tra vào ngày 21-07-2026, cho biết hỗ trợ π0,5 bị giới hạn ở
đầu khớp dòng chảy; đường dẫn lấy mẫu tiêu chuẩn của nó sử dụng lời nhắc được cung cấp và
trực tiếp chạy vòng lặp dòng chảy. Thời gian chạy đã phát hành không nên được mô tả là
tái tạo đầy đủ giai đoạn giải mã văn bản theo cấp bậc của tờ giấy mà không cần
đường dẫn điểm kiểm tra/mã cụ thể chứng minh điều đó.
[Kho lưu trữ openpi chính thức](https://github.com/Physical-Intelligence/openpi)

## Nguồn

- Đen và cộng sự. *π0: Mô hình luồng hành động-ngôn ngữ-thị giác cho Robot thông thường
  Kiểm soát*, §IV, arXiv:2410.24164v4, 2026.
  [Giấy](https://arxiv.org/abs/2410.24164)
- Trí tuệ thể chất và cộng sự. *π0,5: Mô hình Hành động-Ngôn ngữ-Tầm nhìn với
  Tổng quát hóa thế giới mở*, §IV, arXiv:2504.16054, 2025.
  [Giấy](https://arxiv.org/abs/2504.16054)
- Pertsch và cộng sự. *FAST: Mã thông báo hành động hiệu quả cho hành động-ngôn ngữ-chân trời
  Các mẫu*, arXiv:2501.09747. [Giấy](https://arxiv.org/abs/2501.09747)
- Vương và cộng sự. *Qwen-VLA: Thống nhất Mô hình Hành động-Ngôn ngữ-Tầm nhìn giữa các Nhiệm vụ,
  Môi trường và Phương án Robot*, arXiv:2605.30280v2.
  [Giấy](https://arxiv.org/abs/2605.30280)
