
# Diffusion Transformer (DiT)

## DiT là gì?

**Diffusion Transformer (DiT)** là **backbone dựa trên Transformer** cho mô hình sinh.
Nó thay thế U-Net truyền thống trong các mô hình khuếch tán.

Khác với **flow matching**, **DDPM** hay **rectified flow**, DiT **không phải mục
tiêu huấn luyện**. Nó là mạng nơ-ron học phép ánh xạ:

$$
f_\theta(x_t, t, c)
$$

Ở đâu:

- \(x_t\): latent nhiễu
- \(t\): timestep
- \(c\): thông tin điều hòa (văn bản, hình ảnh, trạng thái robot, v.v.)

Mô hình dự đoán mục tiêu được xác định bởi mục tiêu đào tạo.

---

## Mối quan hệ với khớp luồng

DiT và Flow Match giải quyết các vấn đề khác nhau.

| Thành phần | Vai trò |
| ----------------------- | ----------------------------------------------------------- |
| **DiT** | Kiến trúc mạng nơ-ron (Transformer Backbone) |
| **Kết hợp luồng** | Mục tiêu đào tạo dạy cho mạng những gì cần dự đoán |

Kiến trúc DiT tương tự cũng có thể được đào tạo bằng cách sử dụng:

- DDPM (dự đoán tiếng ồn)
- velocity prediction
- score matching
- Flow matching
- Mô hình nhất quán

Chỉ có mục tiêu dự đoán thay đổi.

---

## Kiến trúc tổng hợp

```text
Tiềm ẩn ồn ào + timestep + điều kiện
                │
                ▼
        Transformer khuếch tán
                │
                ▼
      Mục tiêu dự đoán (tiếng ồn, dòng chảy,
      tốc độ, hành động, v.v.)
```

---

## Ứng dụng phổ biến

### 1. Tạo hình ảnh

Việc sử dụng phổ biến nhất.

```text
Chữ
  │
Bộ mã hóa văn bản
  │
Tiềm ẩn + Tiếng ồn
  │
 DiT
  │
Hình ảnh
```

Ví dụ:

- FLUX
- Khuếch tán ổn định 3
- Hình ảnh Qwen

---

### 2. Tạo video

Thay vì các bản vá hình ảnh, mô hình xử lý **các bản vá không gian-thời gian** (thời gian × chiều cao × chiều rộng).

Ví dụ:

- Sora
- Thể loại phim

---

### 3. Tạo âm thanh

DiT có thể tạo hoặc tăng cường âm thanh từ các biểu diễn tiềm ẩn.

Các ứng dụng bao gồm:

- Tổng hợp giọng nói
- Thế hệ âm nhạc
- Chỉnh sửa âm thanh

---

### 4. Thế hệ 3D

Đầu vào có thể là:

- Đám mây điểm
- Voxel
- Biểu diễn 3D tiềm ẩn

Các ứng dụng bao gồm:

- Tạo đối tượng 3D
- Thế hệ CAD
- Kết xuất thần kinh

---

### 5. Người máy

Các mô hình nền tảng robot hiện đại sử dụng DiT để dự đoán các hành động liên tục của robot.

```text
Hình ảnh + Trạng thái Robot + Hướng dẫn
                │
                ▼
               DiT
                │
                ▼
      Quỹ đạo hành động tương lai
```

Đầu ra có thể bao gồm:

- Tư thế tác động cuối
- Lệnh chung
- Hành động điều hướng
- Quỹ đạo thao tác

#### DiT trong Qwen-VLA

Qwen-VLA đặt **bộ giải mã hành động DiT 16 khối riêng biệt** sau nó
Xương sống ngôn ngữ tầm nhìn Qwen3.5-4B. VLM trước tiên tạo ra các trạng thái ẩn từ
hình ảnh, hướng dẫn và lời nhắc thực hiện. Sau khi chiếu tới DiT
chiều rộng, các mã thông báo ngữ cảnh đó được nối với các mã thông báo hành động ồn ào dự kiến
và được xử lý bằng cách sử dụng khả năng tự chú ý của khớp, điều hòa AdaLN theo bước thời gian và
RoPE nhiều phần.

```text
Qwen3.5 VLM trạng thái ẩn + đoạn hành động ồn ào + timestep dòng chảy
                              │
                              ▼
                   Bộ giải mã DiT 16 khối
                              │
                              ▼
                    trường vận tốc hành động
                              │
                    cập nhật Euler lặp đi lặp lại
                              ▼
                  đoạn hành động liên tục
```

Một lần forward qua DiT dự đoán **tốc độ dòng**, chứ không phải lệnh cuối cùng. Bắt đầu từ
Nhiễu tác động Gaussian, một số bước Euler liên tục gọi giá trị xấp xỉ
DiT tham số 1,15B cho đến khi nó tạo ra hành động hoặc quỹ đạo `H × K` cuối cùng
tensor. Do đó, “DiT” xác định kiến ​​trúc bộ giải mã, trong khi luồng điều kiện
kết hợp xác định quá trình học tập và tạo ra nó. [Qwen-VLA, §§2.2–2.5](https://arxiv.org/abs/2605.30280)
Xem [báo cáo chi tiết về bộ giải mã hành động](action_generation/large_diffusion_transformer.md#qwen-vla-the-dit-action-decoding)
cho mặt nạ tensor, phân tích tham số và giao diện đa phương án.

---

### 6. Tạo chuyển động

DiT có thể dự đoán chuyển động của con người hoặc robot trong tương lai.

Ví dụ:

- Dự đoán tư thế con người
- Lập kế hoạch quỹ đạo robot
- Thế hệ hoạt hình

---

### 7. Mô hình khoa học

Các nhà nghiên cứu cũng áp dụng DiT vào dữ liệu khoa học liên tục như:

- Thế hệ phân tử
- Cấu trúc protein
- Thiết kế vật liệu
- Mô phỏng vật lý

---

## Tại sao DiT lại phổ biến

So với U-Nets, Transformers cung cấp:

| Tính năng | U-Net | DiT |
| ----------------------------- | -------- | --------- |
| Sự chú ý toàn cầu | Hạn chế | ✓ |
| Nhân rộng mô hình quy mô lớn | Trung bình | Xuất sắc |
| Điều hòa đa phương thức | Trung bình | Xuất sắc |
| Hỗ trợ mã thông báo có độ dài thay đổi | Hạn chế | ✓ |

Những ưu điểm này đã khiến DiT trở thành xương sống được ưa chuộng cho nhiều mô hình nền tảng thế hệ hiện đại.

---

## Bài học chính

- **DiT là kiến ​​trúc Transformer, không phải thuật toán huấn luyện.**
- Nó có thể được ghép nối với **Flow matching, DDPM, score matching, velocity prediction** và các mục tiêu khác.
- Kiến trúc tương tự được sử dụng trên **hình ảnh, video, âm thanh, 3D, robot, chuyển động và thế hệ khoa học**.
- Tính linh hoạt và khả năng mở rộng của nó đã khiến nó trở thành xương sống thống trị cho nhiều mẫu AI thế hệ gần đây.
