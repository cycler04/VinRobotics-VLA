# Bộ dữ liệu và môi trường được sử dụng để đánh giá các mô hình Qwen VLA

> **Câu hỏi:** Những tập dữ liệu đánh giá, loại tác vụ, modality, hiện thân, quy mô
> và ranh giới phân chia nào đứng sau các kết quả robotics đã công bố của Qwen?
>
> **Phạm vi:** Qwen-VLA cộng với các bộ được chọn trong Qwen-RobotManip, Qwen-RobotNav và
> Qwen-RobotWorld. Nghiên cứu được kiểm tra vào ngày 22-07-2026.

## Tập dữ liệu, môi trường và triển khai là khác nhau

Đánh giá robot có thể kết hợp ba loại artifact:

1. một **tập dữ liệu** về các minh họa, hướng dẫn hoặc quỹ đạo;
2. một **môi trường** tạo ra các quan sát và áp dụng các hành động;
3. **giao thức triển khai** chỉ định nhiệm vụ, hạt giống, đặt lại và chấm dứt.

LIBERO và RoboTwin cung cấp cả quỹ đạo đào tạo và nhiệm vụ mô phỏng;
SimplerEnv chủ yếu là môi trường đánh giá thực tế trên sim; ALOHA là một
hệ thống thực tế trong nhà. Việc gọi tất cả chúng là “bộ dữ liệu” sẽ che giấu những khác biệt này.
Định nghĩa điểm số được lưu giữ trong [metrics.md](metrics.md); các giao thức triển khai và
Giá trị kết quả Qwen nằm trong [benchmarks.md](benchmarks.md); mục tiêu đào tạo
nằm trong [loss.md](loss.md).

## Bản đồ bộ đánh giá

| Suite | Loại và phương án | Thang đo được xuất bản | Quan sát → đầu ra | Phân vùng/cấu hình đã phát hành |
|---|---|---:|---|---|
| LIBERO | Các tác vụ và bản trình diễn trên bàn một tay mô phỏng | Kho ngữ liệu gốc: 130 nhiệm vụ | RGB/trạng thái + ngôn ngữ → hành động thao tác liên tục | Bộ không gian, đối tượng, mục tiêu và LIBERO-100/90/10 |
| Đơn giản hơn-WidowX | Môi trường WidowX thực trên sim | Public SimplerEnv: 4 nhóm nhiệm vụ WidowX | RGB + ngôn ngữ → Hành động Descartes/xoay/kẹp 7D delta ở tần số 5 Hz | Nhóm nhiệm vụ WidowX và các biến thể trực quan |
| RoboCasa-GR1 | Nhà bếp hình người hai tay GR1 mô phỏng | 24 nhiệm vụ nguyên tử được mô tả bởi Qwen-VLA | tầm nhìn/trạng thái nhà bếp + ngôn ngữ → hành động hai tay | Cấu hình GR1 bị giới hạn; khác biệt với RoboCasa365 |
| RoboTwin 2.0 | Mô phỏng cánh tay kép SAPIEN | 50 nhiệm vụ, 5 phương án, >100.000 quỹ đạo được giải phóng | multiview RGB/trạng thái + ngôn ngữ → hành động hai tay | Gói nhiệm vụ/hiện thân; điều kiện sạch và ngẫu nhiên |
| ALOHA trong nhà | Robot hai tay thực sự | 6 danh mục nhiệm vụ trong miền + 5 trục OOD; số lượng cuộc biểu tình chưa biết | 3 camera RGB + ngôn ngữ → thao tác hai tay | Không có gói công khai có thể tái sử dụng hoặc bảng kê khai phân chia |
| R2R trong VLN-CE | Điều hướng trong nhà liên tục ở Matterport3D | 4.475 con đường độc đáo; v1.3 có 10.819 tập tàu và 1.839 tập hướng dẫn chưa được xem | RGB/độ sâu + hướng dẫn bằng tiếng Anh → điểm tham chiếu/điều hướng liên tục | train, val-seen, val-unseen và kiểm tra |
| RxR trong VLN-CE | Điều hướng trong nhà liên tục đa ngôn ngữ | 126.069 hướng dẫn/cặp đường dẫn | tầm nhìn toàn cảnh/liên tục + hướng dẫn EN/HI/TE → đường dẫn điều hướng | tàu, val-seen, val-unseen, tiêu chuẩn kiểm tra và thử thách thử nghiệm |
| SimplerEnv-OOD | Bộ sưu tập tác vụ WidowX OOD tĩnh | 6 nhiệm vụ được giao trong 3 cảnh | RGB + new instruction relation/primitive → 7D action | Các mối quan hệ bị giữ lại, các ràng buộc nguyên thủy và đối tượng màu |
| DOMINO | Bộ dữ liệu thao tác động và trình mô phỏng | 35 nhiệm vụ, 5 phương án, khoảng 117.000 quỹ đạo chuyên gia | multiview được đồng bộ hóa RGB/quyền sở hữu + ngôn ngữ → hành động liên tục | cài đặt sạch và ngẫu nhiên theo tên miền |

"Tỷ lệ đã xuất bản" mô tả nguồn công khai hoặc mô tả tập dữ liệu, không phải
mẫu số triển khai đánh giá. Số lượng thử nghiệm và hạt giống thuộc về điểm chuẩn
giao thức.

## Thao tác mô phỏng

### LIBERO

LIBERO là chuẩn mực học tập suốt đời với các phần trình diễn được điều khiển từ xa bởi con người.
Bộ sưu tập ban đầu chứa 130 nhiệm vụ:

- `LIBERO-Spatial`: 10 nhiệm vụ nhấn mạnh vào mối quan hệ không gian;
- `LIBERO-Object`: 10 nhiệm vụ nhận dạng đối tượng khác nhau;
- `LIBERO-Goal`: 10 nhiệm vụ thay đổi mục tiêu được yêu cầu;
- `LIBERO-100`: 100 nhiệm vụ, cũng được tổ chức dưới dạng đào tạo trước LIBERO-90 và
  Nhiệm vụ xuôi dòng LIBERO-10.

Các quan sát có thể bao gồm không gian làm việc RGB, RGB ở cổ tay và trạng thái robot/môi trường;
ngôn ngữ xác định nhiệm vụ. Hành động bản địa là liên tục và chính thức
các ví dụ sử dụng bảy giá trị, nhưng người chạy phải đọc bộ điều khiển môi trường
thay vì suy ra ngữ nghĩa vật lý từ chiều dài vectơ.

Mã là MIT và bộ dữ liệu được phát hành là CC BY 4.0. [Kho lưu trữ LIBERO] [libero]
[Giấy LIBERO][giấy libero]

### SimplerEnv và Simpler-WidowX

SimplerEnv đánh giá các chính sách robot thực trong mô phỏng bằng cách kết hợp trực quan và
tổng hợp biến thể. Nó hỗ trợ các phiên bản Google Robot và WidowX/Bridge;
môi trường công cộng hiện tại liệt kê sáu nhóm nhiệm vụ của Google và bốn nhóm WidowX
gia đình nhiệm vụ. Dự án đã xác nhận mối tương quan giữa sim và thực với khoảng
1.500 tập đánh giá thực tế trên mỗi miền robot, nhưng con số đó **không** là
Mẫu số đánh giá Qwen-VLA.

Đối với WidowX, chính sách sẽ nhận hình ảnh và hướng dẫn ngôn ngữ rồi phát ra
Hoạt động delta bảy chiều: dịch chuyển, xoay góc trục và bộ kẹp.
Môi trường công cộng chạy điều khiển WidowX ở tần số 5 Hz. [Dự án SimplerEnv][đơn giản hơn]
[Kho lưu trữ] [repo đơn giản hơn]

### RoboCasa-GR1 so với RoboCasa365

Qwen-VLA đánh giá 24 nhiệm vụ nhà bếp nguyên tử bằng máy hình người hai tay GR1
cấu hình. Đây là cấu hình giới hạn của hệ sinh thái RoboCasa, không phải
điểm chuẩn RoboCasa365 mới hơn.

RoboCasa365 hiện tại có 365 nhiệm vụ—65 nhiệm vụ nguyên tử và 300 nhiệm vụ tổng hợp—trên hơn
2.500 cảnh nhà bếp và 3.200 đồ vật. Kho dữ liệu của nó báo cáo hơn 600 giờ
các cuộc biểu tình của con người và hơn 1.600 giờ dữ liệu robot tổng hợp. các
phần đào tạo trước bao gồm 300 nhiệm vụ với 100 màn trình diễn của con người cho mỗi nhiệm vụ
(30.000 bản trình diễn), với các cảnh/nhiệm vụ mục tiêu riêng biệt để đánh giá.

Do đó, số Qwen-VLA 24 ​​tác vụ và kết quả RoboCasa365 mô tả
các nhóm nhiệm vụ khác nhau và phải tách biệt. Mã RoboCasa là MIT;
nội dung/bộ dữ liệu được phát hành là CC BY 4.0. [Dự án RoboCasa][robocasa]
[Giấy Robocasa][giấy robocasa]

### RobotTwin 2.0

RoboTwin 2.0 là nền tảng cánh tay kép SAPIEN-based với 50 tác vụ và 5 tác vụ được hỗ trợ
hiện thân của robot. Bản phát hành công khai của nó chứa hơn 100.000 quỹ đạo,
được lưu trữ trên mỗi tập trong HDF5 cùng với các quan sát và hành động. Đánh giá tách biệt một
điều kiện sạch/dễ dàng hơn từ điều kiện ngẫu nhiên/khó hơn.

Bố cục máy ảnh, bậc tự do và kích thước hành động khác nhau tùy theo phương án, vì vậy bộ chuyển đổi phải
giữ lại siêu dữ liệu phương án và ngữ nghĩa của bộ điều khiển. Mã kho lưu trữ là MIT;
nguồn được tư vấn không thiết lập một giấy phép cho mọi nội dung được tải xuống
quỹ đạo và tài sản. [Nhiệm vụ RoboTwin][nhiệm vụ robotwin]
[Hướng dẫn sưu tập][dữ liệu robotwin]

## ALOHA trong thế giới thực

Nền tảng Qwen-VLA ALOHA có hai cánh tay 6-DoF với các kẹp hàm song song và
ba camera RGB: hai chế độ xem cổ tay và một chế độ xem góc nhìn thứ nhất. Sáu trong miền của nó
hạng mục nhiệm vụ là:

1. chọn và đặt;
2. dọn bàn;
3. xếp bát;
4. gắp bát/đặt đồ vật;
5. gấp khăn;
6. thao tác hạt mịn.

Đánh giá OOD thay đổi màu sắc, đối tượng, vị trí, nền hoặc
ánh sáng và từ ngữ hướng dẫn. Đây là những trục được điều khiển, không phải là một trục riêng biệt
tập dữ liệu có mục đích chung.

Bài viết và kho lưu trữ chính thức không xuất bản gói ALOHA có thể tái sử dụng hoặc
số lần trình diễn, thử nghiệm mỗi nhiệm vụ, tần suất kiểm soát, quy ước hành động,
tách bảng kê khai hoặc giấy phép dữ liệu. Tỷ lệ phần trăm trong bảng điểm chuẩn không
tìm lại các mẫu số còn thiếu. [Qwen-VLA, Mục 5.1.2][qwen-vla]

## Bộ dữ liệu điều hướng

### Matterport3D, R2R và VLN-CE

R2R được tích hợp trong 90 cảnh trong nhà Matterport3D. Tập dữ liệu dựa trên biểu đồ gốc
chứa 7.189 đường dẫn và 21.567 hướng dẫn tiếng Anh, thông thường có ba
hướng dẫn trên mỗi đường dẫn. VLN-CE chuyển đổi các tuyến đường này thành Môi trường sống liên tục
tập phim; Bao bì v1.3 của nó giữ lại 4.475 quỹ đạo R2R độc đáo và tiết lộ:

| Chia | Các tập hướng dẫn | Cảnh |
|---|---:|---:|
| xe lửa | 10,819 | 61 |
| val-nhìn | 778 | 53 |
| val-không nhìn thấy | 1.839 | 11 |
| kiểm tra | 3,408 | 18 |

Số lượng đường dẫn đồ thị, số lượng quỹ đạo liên tục duy nhất và tập lệnh
số đếm là ba mẫu số khác nhau. Trong `val-unseen`, các cảnh được diễn ra
từ đào tạo. RGB/quan sát độ sâu và hướng dẫn
được ánh xạ tới các hành động chuyển động hoặc điểm tham chiếu liên tục. Yêu cầu lưới Matterport3D
quyền truy cập và điều khoản riêng biệt ngay cả khi chú thích tập và mã được công khai.
[Giấy R2R][r2r] [Dữ liệu VLN-CE][vln-ce-data]

### Phòng-ngang-phòng

RxR chứa 126.069 cặp hướng dẫn/đường dẫn trên khoảng 16.500 Matterport3D
quỹ đạo. Hướng dẫn được thu thập bằng tiếng Anh, tiếng Hindi và tiếng Telugu, với
dấu vết đặt ra của người hướng dẫn/người theo dõi dày đặc và thời gian thực hiện từng tư thế. Sự phân chia của nó bao gồm
đào tạo, val-seen, val-unseen, tiêu chuẩn kiểm tra và thử thách thử nghiệm; cảnh chưa nhìn thấy/thử nghiệm
tách rời khỏi các cảnh tập luyện.

Chú thích RxR là CC BY; Nội dung Matterport3D có
điều khoản riêng. [Giấy RxR] [rxr] [Kho lưu trữ chính thức] [rxr-repo]

## OOD và thao tác động

### SimplerEnv-OOD

SimplerEnv-OOD là một bộ sưu tập nhiệm vụ do tác giả tạo trên WidowX với sáu nhiệm vụ được tổ chức
nhiệm vụ trên ba cảnh trên bàn: MoveAway,
MoveRight, PlaceNear, PlaceRight, PutFront và StackYellow. Các trục bị giữ lại
bao gồm các quan hệ không gian, hành động nguyên thủy và các ràng buộc đối tượng màu sắc.

Tên nhiệm vụ và ranh giới đã được ghi lại, nhưng đã phát hành tập IDs và một
gói dữ liệu độc lập thì không.
[Qwen-VLA, Mục 5.1.4][qwen-vla]

### DOMINO

DOMINO nhắm mục tiêu thao tác động thay vì tĩnh. Giấy hiện tại
mô tả 35 nhiệm vụ, năm phương án robot và khoảng 117.000 quỹ đạo chuyên gia,
với các cài đặt rõ ràng và ngẫu nhiên theo tên miền. Một quỹ đạo chứa đồng bộ
đầu/cổ tay RGB, các vị trí khớp cảm nhận bản thân và các tư thế tác động cuối cùng, cùng với
hoạt động liên tục của robot. Nhiệm vụ liên quan đến việc đánh chặn, theo dõi và tính thời gian
tương tác với các vật thể chuyển động.

Mã công khai/điểm nhập dữ liệu tồn tại nhưng giấy phép toàn bộ tập dữ liệu thì không
được xác nhận từ các nguồn sơ cấp được tư vấn. [Giấy DOMINO] [domino]
[Kho lưu trữ] [domino-repo]

## Bộ robot Qwen chuyên dụng

Báo cáo sau này của Qwen bổ sung thêm nhiều dãy phòng. Sau đây có quy mô đủ rõ ràng
trong các báo cáo chính sẽ hữu ích như dữ kiện dữ liệu:

| Suite | Loại dữ liệu/đầu ra | Công bố quy mô và chi tiết |
|---|---|---|
| EBench | Các tập thao tác trên thiết bị di động của Isaac Sim → điều khiển bằng hai tay/di động | 26 loại nhiệm vụ và 794 trường hợp đánh giá; mặt bàn khéo léo, nhóm gắp/đặt di động và các nhóm có tầm nhìn dài |
| RoboTwin-IF | thao tác mô phỏng theo hướng dẫn | 5 bộ nhiệm vụ với các mẫu hướng dẫn được cung cấp sẵn; tổng số tập không được báo cáo |
| RoboCasa365 | thao tác nhà bếp mô phỏng rộng rãi | 365 nhiệm vụ, >2.500 cảnh, 3.200 đối tượng; khác biệt với RoboCasa-GR1 |
| EWMBench | thế hệ video tương lai có điều kiện hành động | 21 mẫu trong 7 nhiệm vụ; cố ý nhỏ |
| WorldModelBench | hướng dẫn/chất lượng/đánh giá vật lý bằng video được tạo ra | 350 trường hợp; đầu ra là video chứ không phải hành động của robot |

LIBERO-Plus, RoboTwin-Clean2Rand và RoboTwin-XE xác định nhiễu loạn hữu ích hoặc
trục phương án, nhưng nguồn Qwen được tư vấn không đưa ra tổng số rõ ràng
của các đợt đánh giá. Họ không nên được chỉ định một thang đo được phát minh.
[Qwen-RobotManip] [robotmanip] [Qwen-RobotWorld] [thế giới robot]

## Hợp đồng nhập chuẩn

Các tensor hành động khác nhau không thể được kết hợp một cách an toàn chỉ từ hình dạng. Một địa phương
bộ chuyển đổi nên giữ lại:

```text
episode_id, task_id, scene_id, tách, hạt giống
văn bản và ngôn ngữ hướng dẫn
tên máy ảnh, hình dạng/dtype RGB, dấu thời gian, hiệu chuẩn
trường trạng thái, đơn vị, thứ tự và dấu thời gian
bộ điều khiển hành động, khung, đơn vị, quy ước xoay và kẹp
tần số điều khiển, đường chân trời chunk, lý do đầu cuối/hết thời gian
vị ngữ thành công, sự kiện can thiệp và va chạm
sửa đổi nguồn, giấy phép/điều khoản và thống kê chuẩn hóa
```

## Tính khả dụng và chi tiết chưa được giải quyết

- Các điểm vào công khai tồn tại cho LIBERO, SimplerEnv, RoboCasa, RoboTwin, VLN-CE,
  R2R/RxR và DOMINO, mặc dù nội dung mô phỏng có thể có các điều khoản riêng biệt.
- Dữ liệu ALOHA của Qwen-VLA không được xuất bản dưới dạng gói điểm chuẩn có thể sử dụng lại.
- Nhiệm vụ công khai chính xác IDs hoặc các bảng kê khai có thể sử dụng lại không có sẵn cho một số
  Các cấu hình do Qwen tạo, đặc biệt là ALOHA và SimplerEnv-OOD.
- Sự chồng chéo ở cấp độ mẫu giữa hỗn hợp tiền huấn luyện lớn của Qwen và công khai
  bộ đánh giá không thể được loại trừ khỏi bài báo.
- Không có bộ phần mềm nào được tải xuống hoặc đưa vào không gian làm việc này cho báo cáo này.

## Nguồn

- Vương và cộng sự. *Qwen-VLA*. [Giấy][qwen-vla] · [PDF cục bộ][qwen-vla-local]
- Lưu và cộng sự. *LIBERO*. [Giấy][libero-giấy] · [Kho lưu trữ][libero]
- Li và cộng sự. *SimpleEnv*. [Giấy][giấy đơn giản hơn] · [Dự án][đơn giản hơn]
- Nasiriany và cộng sự. *RoboCasa*. [Giấy][robocasa-giấy] · [Dự án][robocasa]
- Nhóm RoboTwin. [Tài liệu về tác vụ][robotwin-task] · [Hướng dẫn sưu tập][robotwin-data]
- Anderson và cộng sự. *R2R*. [Giấy][r2r]
- Krantz và cộng sự. *VLN-CE*. [Giấy][vln-ce] · [Trang dữ liệu][vln-ce-data]
- Ku và cộng sự. *RxR*. [Giấy][rxr] · [Kho lưu trữ][rxr-repo]
- Fang và cộng sự. *DOMINO*. [Giấy][domino] · [Kho lưu trữ][domino-repo]
- Đội Qwen. *Qwen-RobotManip*. [Giấy][người máy]
- Đội Qwen. *Qwen-Thế giới robot*. [Giấy][thế giới robot]

[qwen-vla]: https://arxiv.org/abs/2605.30280v2
[qwen-vla-local]: ../../../papers/05-gwen/vla-spec/qwen_vla_2605.30280.pdf
[libero]: https://github.com/Lifelong-Robot-Learning/LIBERO
[giấy libero]: https://arxiv.org/abs/2306.03310
[đơn giản hơn]: https://simpler-env.github.io/
[giấy đơn giản hơn]: https://arxiv.org/abs/2405.05941
[repo đơn giản hơn]: https://github.com/simpler-env/SimplerEnv
[robocasa]: https://robocasa.ai/
[giấy robot]: https://arxiv.org/abs/2406.02523
[nhiệm vụ robotwin]: https://robotwin-platform.github.io/doc/tasks/
[dữ liệu robotwin]: https://robotwin-platform.github.io/doc/usage/collect-data.html
[r2r]: https://arxiv.org/abs/1711.07280
[vln-ce]: https://arxiv.org/abs/2004.02857
[vln-ce-data]: https://jacobkrantz.github.io/vlnce/data
[rxr]: https://arxiv.org/abs/2010.07954
[rxr-repo]: https://github.com/google-research-datasets/RxR
[domino]: https://arxiv.org/abs/2603.15620
[domino-repo]: https://github.com/H-EmbodVis/DOMINO
[điều khiển robot]: https://arxiv.org/abs/2606.17846
[thế giới robot]: https://arxiv.org/abs/2606.17030
