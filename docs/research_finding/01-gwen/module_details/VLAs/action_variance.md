# Các biểu diễn không gian hành động hiện đại trong mô hình Vision-Language-Action (VLA)

## Tổng quan

**Không gian hành động** xác định **chính sách dự đoán điều gì** làm đầu ra của một mô hình Vision-Language-Action (VLA).

Điều quan trọng là phải phân biệt **loại tác vụ** với **không gian hành động**.

```text
Chỉ dẫn
        ↓
Tác vụ
        ↓
Không gian hành động
        ↓
Bộ điều khiển robot
        ↓
Chuyển động của robot
```

Ví dụ:

```text
Chỉ dẫn:
"Nhấc chiếc cốc màu đỏ lên."

Tác vụ:
Thao tác

Không gian hành động:
[
 Δx,
 Δy,
 Δz,
 Δroll,
 Δpitch,
 Δyaw,
 gripper
]
```

Các mô hình VLA khác nhau có thể giải quyết cùng một tác vụ trong khi sử dụng những không gian hành động hoàn toàn khác nhau.

---

# Phân loại các không gian hành động hiện đại

```text
Không gian hành động
│
├── Không gian khớp
│     ├── Vị trí khớp
│     ├── Vận tốc khớp
│     └── Mô-men xoắn khớp
│
├── Không gian Descartes
│     ├── Tư thế của đầu công tác
│     ├── Gia số của đầu công tác
│     ├── Tư thế hai tay
│     └── Bàn tay khéo léo
│
├── Không gian di động
│     ├── Vận tốc đế
│     ├── Điểm tham chiếu điều hướng
│     └── Điều khiển lái
│
├── Không gian quỹ đạo
│     ├── Đoạn hành động
│     ├── Quỹ đạo con người
│     └── Đường đi tương lai
│
└── Không gian tiềm ẩn
      └── Biểu diễn nhúng chuyển động được học
```

---

# 1. Không gian vị trí khớp

Chính sách dự đoán trực tiếp các vị trí khớp mong muốn.

```text
[
 joint1,
 joint2,
 joint3,
 ...
 jointN
]
```

Ví dụ

```text
[
0.31,
-1.08,
0.72,
...
]
```

Bộ điều khiển chỉ cần di chuyển từng động cơ đến góc mong muốn.

### Ưu điểm

- Đơn giản
- Chính xác
- Điều khiển phần cứng trực tiếp

### Nhược điểm

- Đặc thù theo robot
- Khó chuyển giao giữa các hình thái robot

Các robot tiêu biểu

- Franka
- UR5
- xArm
- Nhiều bộ dữ liệu điều khiển từ xa

---

# 2. Không gian vận tốc khớp

Thay vì vị trí, mô hình dự đoán vận tốc khớp.

```text
[
 joint1_velocity,
 joint2_velocity,
 ...
]
```

Ví dụ

```text
[
0.3,
-0.2,
0.4
]
```

Bộ điều khiển tích phân vận tốc để thu được các vị trí khớp trong tương lai.

### Ưu điểm

- Điều khiển mượt
- Điều khiển servo liên tục

### Nhược điểm

- Sai lệch tích phân
- Yêu cầu trạng thái hiện tại của robot

---

# 3. Không gian mô-men xoắn khớp

Biểu diễn hành động ở mức thấp nhất.

```text
[
 torque1,
 torque2,
 ...
]
```

Ví dụ

```text
[
2.1,
-0.8,
4.3
]
```

Điều khiển trực tiếp mô-men xoắn của động cơ.

### Ưu điểm

- Quyền điều khiển tối đa
- Phù hợp với các hành vi động

### Nhược điểm

- Phụ thuộc rất nhiều vào robot
- Khó huấn luyện
- Nhạy với độ trễ

Chủ yếu được sử dụng trong

- Học tăng cường
- Chuyển động của robot hình người

---

# 4. Không gian tư thế của đầu công tác

Dự đoán tư thế của bàn tay robot thay vì các khớp.

```text
[
 x,
 y,
 z,
 roll,
 pitch,
 yaw,
 gripper
]
```

Đôi khi hướng được biểu diễn bằng quaternion

```text
[
 x,
 y,
 z,
 qx,
 qy,
 qz,
 qw,
 gripper
]
```

Sau đó robot giải bài toán động học nghịch.

### Ưu điểm

- Có khả năng chuyển giao cao hơn
- Dễ áp dụng giữa các cánh tay robot hơn

### Nhược điểm

- Yêu cầu bộ giải IK

---

# 5. Không gian gia số của đầu công tác

Đây là biểu diễn hành động phổ biến nhất trong các VLA hiện đại.

Thay vì vị trí tuyệt đối:

```text
[
 Δx,
 Δy,
 Δz,
 Δroll,
 Δpitch,
 Δyaw,
 gripper
]
```

Ví dụ

```text
[
+2 cm,
-1 cm,
0,
0,
0,
5°,
close
]
```

Mỗi hành động là một hiệu chỉnh nhỏ so với tư thế hiện tại.

### Ưu điểm

- Ổn định
- Có tính phản ứng
- Dễ học
- Khả năng chuyển giao tốt hơn

Được sử dụng bởi

- π0
- π0.5
- OpenVLA
- RT-2 (được rời rạc hóa)
- DiffusionVLA

---

# 6. Không gian hành động hai tay

Dành cho robot có hai cánh tay.

```text
[
 tư thế cánh tay trái,

 tư thế cánh tay phải,

 bộ kẹp trái,

 bộ kẹp phải
]
```

Ví dụ

```text
[
EE trái,

EE phải,

Mở,

Đóng
]
```

Phổ biến với

- ALOHA
- Mobile ALOHA
- Robot hình người hai tay

---

# 7. Không gian bàn tay khéo léo

Thay vì một giá trị bộ kẹp, từng ngón tay đều được điều khiển.

```text
[
 cánh tay,

 các khớp ngón cái,

 các khớp ngón trỏ,

 các khớp ngón giữa,

 các khớp ngón áp út,

 các khớp ngón út
]
```

Có thể gồm

- 20
- 30
- 40+
  chiều

Các robot điển hình

- Shadow Hand
- Inspire Hand
- Allegro Hand

---

# 8. Không gian điểm tham chiếu điều hướng

Phổ biến với robot di động.

```text
[
 Δx,
 Δy,
 Δheading
]
```

Ví dụ

```text
[
1.2 m,
0.4 m,
20°
]
```

Biểu diễn điểm tham chiếu tiếp theo thay vì các lệnh điều khiển bánh xe.

### Ưu điểm

- Cấp cao
- Không phụ thuộc vào robot

Được sử dụng bởi

- Robot di động
- Điều hướng trong nhà
- Điều hướng Qwen-VLA

---

# 9. Không gian vận tốc đế

Điều khiển di động ở cấp thấp hơn.

```text
[
 vận tốc tuyến tính,

 vận tốc góc
]
```

Thông thường

```text
[
v,
ω
]
```

Robot chuyển đổi các giá trị này thành tốc độ bánh xe.

Các robot điển hình

- Truyền động vi sai
- Ngăn xếp điều hướng ROS

---

# 10. Không gian điều khiển lái xe

Xe tự hành.

```text
[
 đánh lái,

 ga,

 phanh
]
```

hoặc

```text
[
 đánh lái,

 gia tốc
]
```

Ví dụ

- Lái xe đầu-cuối
- Ô tô tự hành

---

# 11. Không gian quỹ đạo con người

Thay vì các lệnh robot, dự đoán chuyển động tương lai của con người.

```text
[
 tư thế cổ tay,

 tư thế bàn tay,

 tư thế cơ thể
]
```

Thường được biểu diễn dưới dạng

- phép tịnh tiến cổ tay
- phép xoay cổ tay
- cấu hình khớp bàn tay
- các khớp cơ thể

Ứng dụng

- Học bắt chước
- Dự đoán chuyển động con người
- EgoVLA
- Qwen-VLA

---

# 12. Không gian đoạn hành động

Thay vì dự đoán một hành động,

hãy dự đoán đồng thời nhiều hành động tương lai.

Thay vì

```text
Action
```

dự đoán

```text
[
Action_t,

Action_t+1,

Action_t+2,

...

Action_t+15
]
```

Về mặt toán học

$$
A\in\mathbb{R}^{H\times D}
$$

trong đó

- H = chân trời dự đoán
- D = chiều hành động

Ví dụ

```text
[
 [Δx Δy Δz g],
 [Δx Δy Δz g],
 [Δx Δy Δz g],
 ...
]
```

Ưu điểm

- Tần suất suy luận thấp hơn
- Chuyển động mượt hơn
- Tính nhất quán theo thời gian tốt hơn

Đây hiện là định dạng đầu ra chủ đạo cho các VLA dựa trên diffusion và flow.

---

# 13. Không gian hành động ẩn

Thay vì dự đoán các lệnh vật lý,

chính sách dự đoán một vector nhúng đã học được.

```text
[
 z1,
 z2,
 ...
 z128
]
```

Một bộ điều khiển riêng biệt giải mã

```text
Biểu diễn ẩn

↓

Hành động robot
```

Ưu điểm

- Nhỏ gọn
- Không phụ thuộc vào robot
- Điều khiển phân cấp

Nhược điểm

- Khó diễn giải
- Cần bộ giải mã

---

# So sánh

| Không gian hành động       | Chiều điển hình | Bộ điều khiển cần thiết             | Chuyển giao giữa các robot          |
| -------------------------- | --------------: | ----------------------------------- | ----------------------------------- |
| Vị trí khớp                |            6–40 | Không                               | Kém                                 |
| Vận tốc khớp               |            6–40 | Nhỏ                                 | Kém                                 |
| Mô-men xoắn khớp           |            6–40 | Tối thiểu                           | Rất kém                             |
| Tư thế đầu công tác        |             7–8 | IK                                  | Tốt                                 |
| Gia số đầu công tác        |               7 | IK                                  | Xuất sắc                             |
| Hai tay                    |           14–20 | IK                                  | Tốt                                 |
| Bàn tay khéo léo           |           20–50 | IK                                  | Trung bình                          |
| Điểm tham chiếu điều hướng |               3 | Bộ điều khiển điều hướng            | Xuất sắc                             |
| Vận tốc đế                 |             2–3 | Bộ điều khiển di động               | Trung bình                          |
| Điều khiển lái xe          |             2–3 | Bộ điều khiển phương tiện           | Trung bình                          |
| Quỹ đạo con người          |           20–50 | Ánh xạ lại                          | N/A                                 |
| Đoạn hành động             |             H×D | Giống không gian hành động nền tảng | Giống không gian hành động nền tảng |
| Hành động ẩn               |          32–512 | Chính sách giải mã                  | Xuất sắc                             |

---

# Các VLA phổ biến sử dụng không gian hành động nào?

| Mô hình      | Không gian hành động chính                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| RT-2         | Gia số đầu công tác được rời rạc hóa                                                                    |
| OpenVLA      | Gia số đầu công tác được rời rạc hóa                                                                    |
| π0          | Đoạn hành động gia số đầu công tác liên tục                                                              |
| π0.5        | Đoạn hành động gia số đầu công tác liên tục                                                              |
| DiffusionVLA | Đoạn hành động liên tục                                                                                  |
| DexVLA       | Đoạn hành động cánh tay + bàn tay khéo léo                                                               |
| Qwen-VLA     | Không gian đa hành động hợp nhất (khớp, đầu công tác, điều hướng, quỹ đạo con người)                    |

Ghi chú về Liên tục và Rời rạc hóa:

| Khía cạnh        | Hành động liên tục                                  | Hành động được rời rạc hóa                                             |
| ---------------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| Đầu ra           | Số thực. Giá trị trực tiếp của điều khiển hành động. | Token/lớp rời rạc, mỗi token là một hành động được định nghĩa.          |
| .Ví dụ           | `[0.021, -0.14, 0.003, 0.8]`                       | `[523, 112, 901, 45]`                                                  |
| Dự đoán          | Hồi quy                                              | Phân loại / dự đoán token tiếp theo                                    |
| Hàm mất mát      | MSE, L1, Flow Matching                              | Cross-Entropy                                                          |
| Mô hình điển hình | π0, Diffusion Policy, ACT, OpenVLA                | RT-2, Qwen-VLA, các biến thể RoboFlamingo                              |
| Độ chính xác     | Rất cao                                              | Bị giới hạn bởi lượng tử hóa                                            |
| Phù hợp với LLM  | Ít tự nhiên hơn                                      | Rất tự nhiên                                                           |

---

# Xu hướng

## VLA thời kỳ đầu (2022–2024)

Chủ yếu là

- Vị trí khớp
- Gia số đầu công tác
- Token hành động

---

## VLA nền tảng hiện đại (2025–2026)

Đang chuyển dịch theo hướng

- Các đoạn hành động liên tục
- Dự đoán nhiều bước
- Diffusion / Flow Matching
- Không gian hành động đa hiện thân

---

## Hướng phát triển tương lai

Nghiên cứu đang ngày càng khám phá

- Các biểu diễn hành động ẩn
- Không gian hành động phổ quát
- Điều khiển toàn thân cho robot hình người
- Dự đoán hợp nhất quỹ đạo robot + con người
- Các hành động có thể chuyển giao giữa những hiện thân

Xu hướng đang rời xa các lệnh khớp dành riêng cho từng robot để hướng tới các biểu diễn hành động liên tục, có cấp độ cao hơn và có thể chuyển giao, đồng thời vẫn duy trì khả năng thực thi trên nhiều hiện thân robot khác nhau.
