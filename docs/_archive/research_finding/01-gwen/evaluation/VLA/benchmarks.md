# Bằng chứng đánh giá và điểm chuẩn Qwen-VLA

> **Câu hỏi:** Các đánh giá Qwen-VLA được công bố thực sự thiết lập điều gì,
> và những gì vẫn chưa được xác minh?
>
> **Phạm vi:** Qwen-VLA arXiv:2605.30280v2 và kho lưu trữ chính thức của nó, đã được kiểm tra trên
> 22-07-2026. Kiến trúc và đào tạo được đề cập trong
> [qwen_vla_details.md](../../qwen_models/Qwen-VLA/qwen_vla_details.md).

## Câu trả lời ngắn

Qwen-VLA được đánh giá như một checkpoint tổng quát trên thao tác, điều hướng,
chuyển giao trong thế giới thực và các thiết lập ngoài phân phối. Bằng chứng công bố
mạnh nhất là độ bao phủ rộng của một chính sách, không phải bằng chứng rằng mọi điểm
số đều có thể so sánh trực tiếp với mọi baseline chuyên biệt.

Bài báo đưa tin:

- thành công mô phỏng trong phân phối cao trên nền tảng một cánh tay và hai cánh tay;
- truyền ALOHA trong thế giới thực với lợi ích lớn từ việc khởi tạo được huấn luyện trước;
- điều hướng môi trường liên tục mang tính cạnh tranh;
- chuyển giao zero-shot sang các tác vụ thao tác tĩnh và động không nhìn thấy được;
- sự cắt bỏ kết nối đồng đào tạo VL, SFT và RL để đo lường sự thành công của chính sách.

Tất cả các giá trị bên dưới đều **do tác giả báo cáo**. Không có checkpoint hoặc đánh giá Qwen-VLA
mã đã có sẵn trong kho chính thức vào thời điểm được kiểm tra, vì vậy kết quả
chưa được sao chép trong không gian làm việc này. [Kho lưu trữ chính thức] [qwen-vla-repo]

**Lưu ý:** OOD là viết tắt của **out-of-distribution (ngoài phân phối)**, tức thử
mô hình trên dữ liệu khác đáng kể so với phân phối huấn luyện. Đánh giá OOD đo khả
năng khái quát hóa ngoài trải nghiệm huấn luyện.

## Ranh giới giao thức điểm chuẩn

Công thức, phương hướng và giải thích được tách biệt trong
[metrics.md](metrics.md). Điểm chuẩn VLA còn khắc phục môi trường,
nhiệm vụ IDs, phân chia, robot, camera, bộ điều khiển, quy tắc đặt lại/hết thời gian, thành công
vị ngữ, hạt giống và số lượng triển khai.

| Suite | Ranh giới giao thức cần thiết để diễn giải kết quả |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| LIBERO / Đơn giản hơn / RoboCasa / RoboTwin | Các biến thể nhiệm vụ chính xác, sửa đổi trình mô phỏng, phân phối trạng thái ban đầu, mẫu số triển khai và vị từ thành công |
| ALOHA | Thiết lập robot thực, camera/hiệu chuẩn, chính sách can thiệp/đặt lại người vận hành, số lần thử và cấu trúc OOD-axis |
| R2R / RxR VLN-CE | Bản phát hành `Val-Unseen`, tập hợp con ngôn ngữ, toàn cảnh cảm biến, chính sách điểm tham chiếu, khoảng cách trắc địa và ngưỡng thành công |
| DOMINO | `DOMINO@alpha`, cấp độ nhiệm vụ, phương án, điều kiện sạch/ngẫu nhiên, đầu vào khung hình hiện tại/lịch sử và trạng thái không bắn/tinh chỉnh |
| EVT-Bench | Nhiệm vụ một mục tiêu, cài đặt một chế độ xem và vị từ theo dõi chính xác theo từng bước |
| NAVSIM | v1 PDMS thay vì v2 EPDMS,`navtest`, đầu vào cảm biến/lịch sử và sửa đổi trình mô phỏng giả |
| EBench | Sửa đổi, lựa chọn nhóm nhiệm vụ, phân chia phiên bản, phiếu đánh giá tín chỉ một phần cho mỗi nhiệm vụ, hạt giống và tổng hợp nhiệm vụ chéo |
| Bộ RobotWorld | Thiết lập thế hệ, tập hợp con được đánh giá, mô hình/lời nhắc đánh giá và tổng hợp thành phần |

Kết quả NAVSIM của Qwen-RobotNav cung cấp quỹ đạo chân thực từ ba
các khung hình trước đó làm lịch sử. Bảng của nó gọi các biện pháp này là “vòng kín”, nhưng
NAVSIM v1 PDMS sử dụng mô phỏng giả không phản ứng trong bốn giây, vì vậy nó không được
được đọc dưới dạng xác suất thành công của vòng kín phản ứng. [Số liệu NAVSIM][navsim]
[Qwen-RobotNav, Phần 5][robotnav]

Qwen-RobotManip đặt tên cho Bảng EBench, PnP đơn giản và Long Horizon là “tách rời”,
trong khi các tài liệu EBench hiện tại phân biệt các họ nhiệm vụ với các phân tách dữ liệu như
như Xác thực-Đào tạo, Xác thực-Chưa nhìn thấy và Kiểm tra. Ghim bản sửa đổi EBench là
do đó là một phần của việc tái sản xuất, không phải là nhãn hiệu mỹ phẩm. [EBench][băng ghế dài]

## Kết quả mô phỏng chính

Bài viết sử dụng độ dài đoạn hành động `H = 16` và báo cáo tỷ lệ thành công nhiệm vụ trung bình
theo giao thức StarVLA. Qwen-VLA được đào tạo chung theo các phương án;
baseline chuyên môn được tinh chỉnh riêng cho từng điểm chuẩn.

| Người mẫu |         LIBERO |   RoboCasa-GR1 | Đơn giản hơn-WidowX |  RoboTwin Dễ dàng |  RoboTwin Khó |
| --------------------------------------- | -------------: | -------------: | -------------: | -------------: | -------------: |
| Qwen-VLA-Base |           90,8 |           40,4 |           64,3 |           64,3 |           66,4 |
| Qwen-VLA-Instruct | **97,9** | **56,7** | **73,7** | **86.1** | **87,2** |
| Giá trị chuyên môn tốt nhất được liệt kê trong Bảng 4 |           98,6 |           58,3 |           64,6 |           86,0 |           85,0 |

**Đã được xác minh từ bài báo:** Qwen-VLA-Instruct có khả năng cạnh tranh bằng hoặc cao hơn
liệt kê các chuyên gia trên hầu hết các cột trong khi sử dụng một chính sách chung.

**Giới hạn so sánh:** các hàng chuyên gia và tổng quát được đào tạo khác nhau
chế độ. Bảng này thể hiện một kết quả tổng quát mạnh mẽ theo quan điểm của tác giả.
giao thức; nó không tách biệt dữ liệu, điện toán, kiến ​​trúc hoặc ngân sách điều chỉnh như nhau.
[Qwen-VLA, Mục 5.1.1 và Bảng 4][qwen-vla]

## Chuyển ALOHA trong thế giới thực

Cả hai biến thể Qwen-VLA ALOHA đều sử dụng cùng một kiến ​​trúc. Một chuyến tàu từ đầu;
các tinh chỉnh khác từ Qwen-VLA-Base.

| Cài đặt | Từ đầu | Từ Qwen-VLA-Base | Sự khác biệt |
| ---------------------------------------------- | -----------: | -----------------: | ---------: |
| Sáu loại nhiệm vụ trong miền, thành công trung bình |         48,5 |     **83,6** |   +35,1 trang |
| Năm hạng mục OOD, thành công trung bình |         36,2 |     **76,9** |   +40,7 trang |

Các danh mục **OOD** là màu sắc, phiên bản đối tượng, vị trí, nền và
chỉ dẫn. Đây là bằng chứng rõ ràng nhất trong báo cáo rằng việc đào tạo trước trên diện rộng,
không chỉ kiến ​​trúc, cải thiện khả năng truyền tải vì hai biến thể có chung
kiến trúc giống nhau. Nó vẫn là một nền tảng robot và một bộ phòng thí nghiệm hữu hạn
điều kiện. [Qwen-VLA, Mục 5.1.2 và Bảng 5-6][qwen-vla]

## Điều hướng

Qwen-VLA được đánh giá trên phần tách `Val-Unseen` của R2R và RxR trong VLN-CE bằng cách sử dụng
một hành động điểm tham chiếu cửa sổ trượt.

| Người mẫu |         R2R OS |         R2R SR |        R2R SPL |         RxR SR |        RxR SPL |       RxR nDTW |
| ------------------ | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: |
| Qwen-VLA-Base |           61,7 |           53,8 |           49,4 |           55,1 |           45,8 |           56,2 |
| Qwen-VLA-Instruct | **69,0** | **57,5** |           51,2 | **59,6** | **47,8** |           57,1 |
| Đường cơ sở StreamVLN |           64,2 |           56,9 | **51,9** |           52,9 |           46.0 | **61,9** |

Qwen-VLA-Instruct dẫn đầu các baseline mở được liệt kê về tỷ lệ thành công, nhưng không phải mọi
thước đo chất lượng đường đi. Vấn đề này: SR cao hơn với nDTW thấp hơn có nghĩa là đích đến
thành công và độ chính xác của quỹ đạo nên vẫn là những kết luận riêng biệt.
[Qwen-VLA, Mục 5.1.3 và Bảng 7][qwen-vla]

## Thao tác OOD tĩnh và động

| Điểm chuẩn | Phân biệt đào tạo/đánh giá |     Qwen-VLA-Base |           Qwen-VLA-Instruct | So sánh mạnh mẽ trên giấy |
| -------------- | -------------------------------------------------------------- | ----------------: | --------------------------: | ----------------------------- |
| SimplerEnv-OOD | Tinh chỉnh chọn và đặt cầu; kiểm tra sáu loại nhiệm vụ chưa nhìn thấy |           25.3 SR |           **32.0 SR** | pi0.5: 12.6 SR |
| DOMINO | Zero-shot trên tất cả 35 dãy động; không tinh chỉnh động | 21.1 SR / 37.4 MS | **26,6 SR / 39,5 MS** | LingBot-VA: 24.1 SR / 36.1 MS |

SimplerEnv-OOD thăm dò các hướng dẫn không gian chưa được nhìn thấy, nguyên thủy và đối tượng màu
ràng buộc. DOMINO thăm dò thao tác đối tượng chuyển động và thực thi liên tục
chất lượng. Đây là nhiều thông tin hơn cho sự mạnh mẽ hơn so với phân phối trong
trung bình, mặc dù thành công tuyệt đối vẫn ở mức thấp trên cả hai bộ.
[Qwen-VLA, Phần 5.1.4-5.1.5 và Bảng 8-9][qwen-vla]

## Sự cắt bỏ hỗ trợ những gì

### Giai đoạn sau đào tạo

Việc triển khai RL chỉ được thu thập trong SimplerEnv với phần thưởng thành công nhị phân.

| Sân khấu |        Đơn giản hơn |       RoboCasa |           RoboTwin E/H |         LIBERO |    Đơn giản hơn OOD |          DOMINO SR/MS |
| ----- | -------------: | -------------: | ---------------------: | -------------: | -------------: | --------------------: |
| CPT |           64,3 |           40,4 |            64,3 / 66,4 |           90,8 |           25.3 |           21.1 / 37.4 |
| + SFT |           70,8 |           56,0 |            86,3 / 87,1 |           97,8 |           31.6 |           25,7 / 39,1 |
| + RL | **73,7** | **56,7** | 86,1 / **87,2** | **97,9** | **32.0** | **26,6 / 39,5** |

**Đã xác minh:** SFT cung cấp phần lớn mức tăng. RL thêm +2,9 trang khi triển khai
môi trường và những thay đổi nhỏ ở nơi khác, bao gồm -0,2 pp trên RoboTwin-Easy.
Điều này hỗ trợ việc chuyển giao khiêm tốn mà không bị lãng quên rộng rãi; nó không
ủng hộ tuyên bố rằng RL cải thiện đồng đều mọi tác vụ. [Qwen-VLA, Mục 5.2.3][qwen-vla]

### Đồng đào tạo và trạng thái VL

- Việc trộn dữ liệu VL với dữ liệu hành động sẽ cải thiện RoboCasa-GR1 thêm 4,9 pp và RoboTwin
  2,0 x 4,6 pp trong quá trình cắt bỏ được báo cáo, trong khi LIBERO và Simpler tương tự nhau.
- Việc thêm trạng thái chung sẽ thay đổi thành công của RoboTwin tối đa là +0,7 pp trên Dễ và
  +1,3 pp trên Hard, do đó mô hình mặc định bỏ qua trạng thái cảm nhận bản thể.

Đây là những phát hiện cụ thể về điểm chuẩn, không phải là tuyên bố chung về tất cả VLAs.
[Qwen-VLA, Phần 5.2.2 và 5.2.4][qwen-vla]

## Các mô hình robot Qwen liên quan

Qwen-VLA là mô hình thao tác/điều hướng hợp nhất. Ba báo cáo sau đó của Qwen
chuyên môn hóa hoặc thay đổi vấn đề, vì vậy điểm số của họ sẽ không được gộp vào
Bàn Qwen-VLA.

| Người mẫu | Đầu ra/vấn đề | Bằng chứng được công bố mạnh mẽ | Bằng chứng phản biện quan trọng |
| --------------- | --------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen-RobotManip | Chính sách thao túng liên tục | LIBERO-Plus 89.0; RoboTwin-Clean2Rand Cứng 62,6; EBench 45.6 SR | Một số bộ OOD mới do tác giả đề xuất; sinh sản bên ngoài không được báo cáo |
| Qwen-RobotNav | Chính sách điều hướng | Toàn cảnh RxR Val-Unseen 76,5 SR/65,7 SPL; VLNVerse hạt mịn 63,75 SR/57,93 SPL | Tỷ lệ theo dõi EVT là cao nhất, nhưng thành công 77,4/78,6 kém các giá trị chuyên môn 86,0-86,9; kết quả lái xe phụ thuộc rất nhiều vào lịch sử trước đó |
| Qwen-RobotWorld | Mô hình thế giới video tương lai có điều kiện về ngôn ngữ | Tổng số nguồn mở tốt nhất trong các bảng DreamGen và WorldModelBench được báo cáo | Nó không xuất ra các hành động thực thi của robot; một số số liệu QA/IF sử dụng Qwen2.5-VL làm đánh giá, tạo ra sự thiên vị tiềm năng cho người đánh giá-gia đình |

Phát hiện hữu ích nhất của RobotManip là điểm số trong quá trình phân phối đã được
gần bão hòa, trong khi OOD kiểm tra các biến thể xước và biến thể được huấn luyện trước riêng biệt. Vì
ví dụ: mô hình cào của nó đạt điểm 78,3 trên LIBERO-Plus so với 89,0 được huấn luyện trước và
22,6 so với 62,6 trên RoboTwin-Clean2Rand Hard. Điều này hỗ trợ điều trị được kiểm soát
sự thay đổi phân phối dưới dạng đánh giá mô hình nền tảng chính chứ không phải là phụ lục.
[Qwen-RobotManip, Bảng 3-5][robotmanip]

Qwen-RobotNav cung cấp kết quả phủ định hữu ích: theo dõi tỷ lệ và thành công của nhiệm vụ
có thể di chuyển theo các hướng khác nhau. Mô hình 4B của nó báo cáo tỷ lệ theo dõi 90,0 nhưng
77,4 thành công, thấp hơn thành công 86,9 của ABot-N0. Điểm tiêu đề điều hướng duy nhất
sẽ che giấu hành vi này. [Qwen-RobotNav, Bảng 6][robotnav]

Qwen-RobotWorld thuộc nhóm nghiên cứu đánh giá vì các mô hình thế giới có thể trở thành
trình mô phỏng, nhà phê bình chính sách hoặc trình tạo dữ liệu tổng hợp. Các bảng hiện tại của nó làm
**không** thiết lập rằng chính sách đào tạo về video được tạo sẽ cải thiện khả năng thực thi
điều khiển. [Qwen-RobotWorld, Phần 5][thế giới robot]

## Khoảng cách đánh giá

- **Đã xác minh:** Hầu hết các nhiệm vụ định lượng vẫn có thời hạn ngắn và dựa trên điểm chuẩn;
  bài báo gọi việc triển khai trong thời gian dài và khắc phục lỗi là các vấn đề mở.
- **Đã xác minh:** Bằng chứng OOD trong thế giới thực là từ ALOHA và một tập hợp hình ảnh được giới hạn
  và ca dạy; nó không phải là sự sao chép giữa các phòng thí nghiệm.
- **Đã xác minh:** Kho lưu trữ chính thức hiện đang trình bày báo cáo và kết quả
  nhưng không có mô hình có thể tải xuống, triển khai suy luận hoặc khai thác đánh giá.
- **Không xác định:** khoảng tin cậy, phương sai từ lần chạy này đến lần chạy khác và số lượng chính xác
  triển khai đánh giá không được báo cáo nhất quán bên cạnh mỗi tổng hợp.
- **Không xác định:** độ trễ kiểm soát từ đầu đến cuối, thời hạn bị bỏ lỡ, mức sử dụng bộ nhớ và
  thông lượng phụ thuộc vào phần cứng.
- **Không xác định:** mức độ nghiêm trọng của va chạm, tình huống suýt va chạm không an toàn, tỷ lệ can thiệp và
  phục hồi sau sự cố một phần.
- **Suy luận:** Quyết định triển khai sẽ ảnh hưởng đến thành công, độ trễ và
  thất bại về mặt an toàn nặng nề hơn so với sự khác biệt nhỏ về điểm số trong phân phối.

## Giao thức tái tạo tối thiểu

Đối với mỗi checkpoint và nhiệm vụ, ghi:

1. sửa đổi môi trường và mã, nhiệm vụ IDs, hạt giống và số lần triển khai;
2. phương án robot, máy ảnh, tiền xử lý hình ảnh, tần số điều khiển và hành động
   chân trời chunk;
3. ngữ nghĩa hành động, đơn vị, khung tọa độ, chuẩn hóa và bộ kẹp
   quy ước;
4. định nghĩa vị từ thành công, thời gian chờ, can thiệp và va chạm;
5. thành công trung bình với khoảng tin cậy cộng với kết quả trên mỗi nhiệm vụ;
6. phân phối độ trễ suy luận, trễ thời hạn, bộ nhớ cao nhất và phần cứng;
7. phân loại lỗi: nhận thức, nền tảng, lập kế hoạch, kiểm soát, phục hồi hoặc
   lỗi môi trường.

Nếu không có các trường này, tổng hợp được sao chép sẽ khó so sánh với
giấy tờ và không đủ để đưa ra quyết định triển khai robot.

## Nguồn

- Vương và cộng sự. *Qwen-VLA: Thống nhất Mô hình Hành động-Ngôn ngữ-Tầm nhìn giữa các Nhiệm vụ,
  Môi trường và các phương án Robot*. arXiv:2605.30280v2, 2026.
  [Giấy][qwen-vla] · [PDF cục bộ][qwen-vla-local]
- Đội Qwen. *Kho lưu trữ chính thức của Qwen-VLA*. Truy cập 2026-07-22.
  [Kho lưu trữ] [qwen-vla-repo]
- Đội Qwen. *Qwen-RobotManip*. arXiv:2606.17846, 2026.
  [Giấy][robotmanip] · [PDF cục bộ][robotmanip-local]
- Đội Qwen. *Qwen-RobotNav*. arXiv:2606.18112, 2026.
  [Giấy][robotnav] · [PDF cục bộ][robotnav-local]
- Đội Qwen. *Qwen-Thế giới robot*. arXiv:2606.17030, 2026.
  [Giấy][thế giới robot] · [PDF cục bộ][robotworld-local]
- Fang và cộng sự. *DOMINO*. [Giấy][domino]
- Đội NAVSIM. [Định nghĩa số liệu chính thức][navsim]
- Thực tập sinh Robotics. [Tài liệu EBench][ebench]

[qwen-vla]: https://arxiv.org/abs/2605.30280v2
[qwen-vla-local]: ../../../papers/05-gwen/vla-spec/qwen_vla_2605.30280.pdf
[qwen-vla-repo]: https://github.com/QwenLM/Qwen-VLA
[điều khiển robot]: https://arxiv.org/abs/2606.17846
[robotmanip-local]: ../../../papers/05-gwen/vla-spec/qwen_robotmanip_2606.17846.pdf
[robotnav]: https://arxiv.org/abs/2606.18112
[robotnav-local]: ../../../papers/05-gwen/vla-spec/qwen_robotnav_2606.18112.pdf
[thế giới robot]: https://arxiv.org/abs/2606.17030
[robotworld-local]: ../../../papers/05-gwen/vla-spec/qwen_robotworld_2606.17030.pdf
[domino]: https://arxiv.org/abs/2603.15620
[điều hướng]: https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md
[bàn ghế]: https://internrobotics.github.io/EBench-doc/
