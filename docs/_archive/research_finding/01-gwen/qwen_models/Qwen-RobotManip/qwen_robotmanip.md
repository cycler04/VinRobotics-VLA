# Qwen-RobotManip: Tóm tắt bài thuyết trình

> **Nguồn đầy đủ sự thật:** [Báo cáo đầy đủ Qwen-RobotManip](qwen_robotmanip_details.md). Phiên bản ngắn này
> cố tình bỏ qua các chi tiết triển khai, tính toán dữ liệu đầy đủ, các phương trình và cảnh báo benchmark. Bài viết chính: [Qwen-RobotManip v2](https://arxiv.org/abs/2606.17846v2).

## Thông điệp chính

> **Qwen-RobotManip căn chỉnh các biểu diễn, chuyển động và hành vi không đồng nhất của robot trước khi mở rộng quy mô
> huấn luyện thao tác.**

- Hỗ trợ các hiện thân một tay máy, hai tay máy, gripper, bàn tay khéo léo, robot di động và robot hình người.
- Sử dụng Qwen3.5-4B để suy luận đa phương thức.
- Sử dụng DiT phù hợp với luồng để tạo ra các khối hành động liên tục.
- Biểu diễn trạng thái và hành động trong một mẫu 80-D dùng chung có mặt nạ.
- Sử dụng delta của end-effector trong hệ tọa độ camera để cải thiện khả năng chuyển giao giữa các robot.

## Kiến trúc

![Cấu trúc Qwen-RobotManip](Image/architecture_overview.png)

| Thành phần | Bài thuyết trình mang đi |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| Đầu vào | RGB đa góc nhìn, prompt hiện thân có cấu trúc, proprioception hiện tại, hình học camera và lịch sử tùy chọn |
| Xương sống VLM | Qwen3.5-4B cùng xử lý tầm nhìn, ngôn ngữ và bối cảnh lịch sử |
| Action expert | 10 khối DiT, rộng 768, 12 đầu |
| Chú ý chéo | Thay thế điều hòa hình ảnh và ngôn ngữ trên các khối DiT |
| Đầu ra | Đoạn hành động liên tục trong không gian 80-D chuẩn |
| Suy luận | Bốn bước tích hợp Euler trong thiết lập mặc định được báo cáo |

### Ví dụ thực tế: Đầu vào cho hành động của robot

Sau đây là **bản trình bày được tái tạo**, không phải lược đồ API được phát hành. Nó cho thấy làm thế nào một
quyết định hai cánh tay có thể được tập hợp:

```yaml
images:
  - {camera: left_external, rgb: <image>, calibrated: true}
  - {camera: front_external, rgb: <image>, calibrated: true}
  - {camera: right_wrist, rgb: <image>, calibrated: true}

prompt: |
  embodiment: robot_aloha
  instruction: Lấy đồ chơi khỏi bàn và đặt nó lên tấm thảm.
  speed: 1000
  fps: 30
  camera view direction: arm side

state:
  canonical_80d: [q_left_1, ..., gripper_left, q_right_1, ..., gripper_right, 0, ...]
  active_mask:   [1, ..., 1, 1, ..., 1, 0, ...]

reference_camera:
  left_arm: front_external
  right_arm: right_wrist
history: <hình ảnh, trạng thái và đoạn hành động đã thực thi trước đó>
```

```mermaid
flowchart LR
    I[Ba chế độ xem RGB] --> V[Token trực quan Qwen3.5]
    P[Hướng dẫn và prompt thực hiện] --> L[Token ngôn ngữ Qwen3.5]
    H[Lịch sử trực quan tùy chọn] --> V
    S[Lịch sử hành động và trạng thái 80-D được che giấu] --> SM[MLP trạng thái và lịch sử]
    N[Đoạn hành động 80-D ồn ào] --> D[DiT phù hợp với dòng chảy 10 khối]
    C[Hình dạng camera và thời gian dòng chảy] --> D
    V --> D
    L --> D
    SM --> D
    D --> E[Bốn bước khử nhiễu Euler]
    E --> A[Đoạn hành động 80-D được che giấu]
    A --> X[Giải mã các khe hoạt động và thực thi]
```

Bên trong DiT, các khối liên tiếp luân phiên cross-attention đến đặc trưng hình ảnh và ngôn ngữ. Đầu ra
là **một đoạn hành động trong tương lai**, không phải lệnh của robot bằng ngôn ngữ tự nhiên. Một giải mã minh họa đơn giản hóa là:

| Bước đầu ra | Các trường chuẩn đang hoạt động | Giải thích vật lý |
| ----------- | ----------------------------------------------------- | ----------------------- |
| (t+1) | EEF trái `(+0.02, 0.00, -0.01)`; gripper `open` | Tiếp cận đồ chơi |
| (t+2) | EEF trái `(0.00, 0.00, -0.02)`; gripper `close` | Hạ xuống và gắp |
| (t+3) | EEF trái `(+0.04, +0.03, +0.05)`; gripper `closed` | Di chuyển về phía tấm thảm |

Các giá trị ở trên chỉ minh họa vùng delta của khung máy ảnh. Bài báo không công bố tensor của mẫu này,
độ dài đoạn, tên trường API hoặc lệnh điều khiển.

## Tại sao Biểu diễn 80-D lại quan trọng

```text
Tay máy trái:  29 chiều
Tay máy phải:  29 chiều
Dự phòng:      22 chiều
Tổng:          80 chiều
```

Mỗi khối cánh tay 29-D chứa:

| Trường | Số chiều |
| ------------------ | ---------: |
| Vị trí khớp |          7 |
| Trạng thái end-effector |          9 |
| Kẹp |          1 |
| Bàn tay khéo léo |         12 |

Điểm mấu chốt: mặt nạ ngăn ngừa các khớp bị thiếu hoặc cánh tay không được sử dụng trở thành giám sát giả có giá trị bằng 0.

## Ba lớp căn chỉnh

```mermaid
flowchart LR
    D[Dữ liệu robot không đồng nhất] --> R[Căn chỉnh biểu diễn]
    R --> M[Căn chỉnh chuyển động]
    M --> B[Căn chỉnh hành vi]
    B --> P[Chính sách thao túng chia sẻ]
```

| Căn chỉnh | Thay đổi gì | Tại sao nó giúp ích |
| -------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Biểu diễn | Ánh xạ mọi robot vào cùng một mẫu 80-D | Các khớp tương đương và end-effector chiếm các vị trí nhất quán |
| Chuyển động | Thể hiện vùng delta của bộ end-effector trong khung máy ảnh đã chọn | Chuyển động giống nhau về mặt trực quan trở nên giống nhau về mặt số lượng |
| Hành vi | Điều kiện về ID robot, FPS, tốc độ tập, hướng camera và lịch sử gần đây | Chính sách có thể thích ứng với động học và phong cách thực thi mà không cần cập nhật trọng lượng |

## Prompt hiện thân mẫu

```text
embodiment: robot_aloha
instruction: Lấy đồ chơi khỏi bàn và đặt nó lên tấm thảm.
speed: 1000
fps: 30
camera view direction: arm side
```

Thận trọng khi trình bày: `speed` có nghĩa là độ dài tập tính theo thùng 500 bước, không phải mét/giây vật lý.

## Dữ liệu huấn luyện

![Khối huấn luyện Qwen-RobotManip](Image/training_data_corpus.png)

| Nhóm dữ liệu |     Quy mô báo cáo | Giải thích chính |
| --------------------------- | -----------------: | ------------------------------------------------------------------------ |
| Trình diễn robot trực tiếp |           11.420 giờ | Chín nguồn dữ liệu robot mở trên nhiều hiện thân |
| Video góc nhìn thứ nhất của con người |            1.933 giờ | Các tập con EgoDex, VITRA và EgoVerse đã lọc |
| Tổng hợp từ người sang robot |           24.808 giờ | Bắt nguồn từ các video giống nhau của con người trên 15 hình thái robot |
| Tổng số thao tác |           38.161 giờ | Được tác giả làm tròn lên khoảng 38.100 h |
| VL đồng huấn luyện | Khoảng 28 triệu ví dụ | Duy trì nhận thức, ngôn ngữ, lý luận không gian và khả năng ECoT |

Giờ tổng hợp có nguồn gốc từ thang đo chứ không phải trải nghiệm độc lập của con người.

## Luồng huấn luyện

```mermaid
flowchart LR
    R[Trình diễn robot] --> C[Sắp xếp và căn chỉnh]
    H[Video con người] --> S[Tổng hợp từ người sang robot]
    S --> C
    C --> VLA[Lô VLA]
    VL[VL và dữ liệu lý luận] --> VLB[lô VL]
    VLA --> T[Huấn luyện trước luồng kép 9:1]
    VLB --> T
    T --> SFT[SFT tổng hợp tên miền]
```

| Loại batch | Mục tiêu | Tỷ lệ báo cáo |
| ---------- | ------------------------------------- | -------------: |
| VLA | Flow matching có mặt nạ trên các đoạn hành động |      Khoảng 90% |
| VL | Dự đoán token tiếp theo tự hồi quy |      Khoảng 10% |

Bài viết báo cáo tỷ lệ nhưng không phải là thứ tự lô lặp lại cố định hoặc chính sách lấy mẫu hoàn chỉnh theo từng nguồn.

## Điểm nổi bật của đánh giá

| Đánh giá |                  Kết quả báo cáo chính | Nó thể hiện điều gì |
| ------------------------ | ------------------------------------: | ------------------------------------------------------------ |
| LIBERO |                 99,1 SR; bối cảnh 99,2 | Hiệu suất phân phối mạnh mẽ nhưng gần bão hòa |
| LIBERO-Plus |                    89,0; bối cảnh 91,4 | Mạnh mẽ đối với bảy trục nhiễu loạn OOD |
| RoboTwin-Clean2Rand Hard |                    62,6; bối cảnh 69,4 | Bối cảnh giúp theo chuyển cảnh kết hợp |
| RoboCasa365 |                         tổng 35,9 SR | Thao tác với đường chân trời dài và bố cục vẫn còn khó khăn |
| RoboTwin-IF |                       SR trung bình 72,2 | Làm theo mẫu ngôn ngữ có cấu trúc |
| RoboTwin-XE |                       SR trung bình 23,9 | Chuyển giao zero-shot giữa các hiện thân vẫn còn nhiều thách thức |
| Bảng RoboChallenge30 v1 | 45% nhiệm vụ thành công; 59,83 điểm quá trình | Đánh giá robot thực đa nền tảng |

## Phạm vi hành động

![Phân loại hành động nguyên tử Qwen-RobotManip](Image/atomic_action_taxonomy.png)

## Trang trình bày cuối cùng: Năm điểm

1. Cải tiến chính là **căn chỉnh nhiều hiện thân**, không chỉ là tập dữ liệu lớn hơn.
2. Mẫu và mặt nạ 80-D giúp các hình thái robot khác nhau được huấn luyện cùng nhau.
3. Delta trong hệ tọa độ camera căn chỉnh tọa độ hành động với quan sát trực quan.
4. Lịch sử đóng vai trò như một mô tả ngầm về hành vi và động học của robot.
5. Kết quả OOD và chuyển giao giữa robot đầy hứa hẹn, nhưng artifact tổng hợp, độ trễ và điều khiển phản ứng vẫn còn hạn chế.
