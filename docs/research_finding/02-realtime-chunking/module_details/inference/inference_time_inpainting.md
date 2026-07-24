# RTC inference như một bài toán inpainting trên action chunk

## Ý tưởng chính

RTC tại inference có thể được hiểu như **image inpainting**, nhưng thay vì chỉnh sửa pixel,
nó chỉnh sửa một chuỗi action theo thời gian.

Trong image inpainting:

- một phần ảnh cũ được giữ lại;
- phần bị che được sinh lại;
- vùng biên thường cần chuyển tiếp mượt để ảnh mới không bị lệch khỏi phần cũ.

Trong RTC:

- chunk trước đóng vai trò dữ liệu cũ;
- các action đã cam kết phải được giữ;
- phần overlap được dùng làm guidance;
- các action xa hơn được sinh tự do từ observation mới.

Có thể xem đây là **temporal inpainting**:

```text
Image inpainting:
known pixels | soft boundary | missing pixels

RTC:
hard prefix  | soft overlap  | free postfix
```

Ví dụ với chunk cũ:

```text
C1 = [a0 a1 a2 a3 a4 a5 a6 a7]
```

Sau khi robot đã thực thi `a0, a1`, chunk mới cần sinh có dạng:

```text
C2 = [b0 b1 b2 b3 b4 b5 b6 b7]
      └──── overlap với a2...a7 ────┘
```

RTC không đơn giản copy toàn bộ `a2...a7`. Nó giữ chắc phần gần hiện tại, sau đó giảm dần
ảnh hưởng của kế hoạch cũ để policy có thể phản ứng với observation mới.

![1784866247856](image/inference_time_inpainting/1784866247856.png)

---

## Flow sampler tiêu chuẩn

Flow policy bắt đầu từ Gaussian noise:

$$
x_0 \sim \mathcal N(0,I).
$$

Với `N` flow steps, sampler dùng Euler update:

$$
x_{k+1} = x_k + \frac{1}{N} v_\theta(o, x_k, \tau_k).
$$

Đây là sampler trong
[`FlowPolicy.action`, `model.py` dòng 160–171](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L160).

Trong Kinetix, evaluation mặc định dùng `N = 5`. Không có Heun correction, adaptive ODE
solver hay stochastic update bổ sung sau khi khởi tạo noise.

---

## RTC thêm inpainting guidance như thế nào?

Tại mỗi flow timestep, policy trước hết dự đoán velocity:

$$
v_\theta(o, x_\tau, \tau).
$$

Trong đó:

- $v_\theta$: velocity field do Flow Policy dự đoán.
- $o$: observation hiện tại (camera image, robot state, ...).
- $x_\tau$: noisy action chunk tại flow time $\tau$.
- $\tau \in [0,1]$: flow time, biểu diễn mức tiến triển của quá trình denoising (`0`: hoàn toàn nhiễu, `1`: action chunk sạch).

Velocity này chỉ hướng mà action chunk hiện tại nên di chuyển trong không gian action để dần trở thành action chunk hoàn chỉnh.

Từ velocity này, code ước lượng action chunk sạch:

$$
\hat A_1 = x_\tau + (1-\tau) v_\theta.
$$

Phép ước lượng này nằm trong hàm `denoiser` bên trong
[`FlowPolicy.realtime_action`, `model.py` dòng 235–246](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L235).

Sau đó nó so sánh chunk dự đoán với chunk cũ đã căn chỉnh:

$$
Y - \hat A_1.
$$

Sai số được nhân với soft mask `W`:

$$
e = W \odot (Y - \hat A_1).
$$

`W` quyết định vị trí nào phải bám chặt vào chunk cũ, vị trí nào chỉ bám nhẹ, và vị trí nào được sinh tự do.

Correction được backpropagate qua denoiser bằng VJP:

$$
\Delta v = J^\top e.
$$

Hai dòng `jax.vjp(...)` và `vjp_fun(...)` hiện thực trực tiếp phép tính này tại
[`model.py` dòng 235–246](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L235).

**Velocity cuối cùng là:**

$$
v_{\text{RTC}} = v_\theta + \lambda(\tau) \Delta v,
$$

trong đó:

- $v_\theta$: velocity gốc do flow policy dự đoán từ observation và noisy action chunk.
- $\Delta v$: correction được tính từ VJP để điều chỉnh velocity về phía action chunk trước.
- $\lambda(\tau)$: guidance scale phụ thuộc flow time, quyết định correction mạnh hay yếu.

Thay $\Delta v$ vào, ta được:

$$
v_{\text{RTC}} = v_\theta + \lambda(\tau) J^\top \left[ W \odot (Y - \hat A_1) \right].
$$

Ý nghĩa các thành phần:

- $Y$: action chunk trước (đã được căn chỉnh với chunk hiện tại).
- $\hat A_1$: **clean chunk estimate**, tức action chunk hoàn chỉnh mà model dự đoán sẽ thu được nếu tiếp tục flow đến $t=1$.
- $Y - \hat A_1$: sai số giữa kế hoạch cũ và kế hoạch mới.
- $W$: **soft-mask weight**, quyết định timestep nào phải bám chặt vào kế hoạch cũ và timestep nào được thay đổi tự do.
- $e = W \odot (Y - \hat A_1)$: weighted error chỉ giữ lại phần sai số cần được ưu tiên sửa. Coi như là hướng chỉnh của velocity.
- $J = \dfrac{\partial \hat A_1}{\partial x_t}$: Jacobian của denoiser, biểu diễn độ nhạy của clean chunk đối với thay đổi của noisy chunk. Coi như là một mapping giữa $\hat A_1$ và $x_t$.
- $J^\top e$: **VJP (Vector-Jacobian Product)**, backpropagate weighted error từ clean chunk về noisy chunk để tạo correction cho velocity.
- $\lambda(\tau)$: hệ số guidance được tính từ

$$
\lambda(\tau)
=
\min\left(
 c(\tau)\,r^{-2}(\tau),
 \beta
\right),
$$

trong đó:

- $c(\tau) = \dfrac{1-\tau}{\tau}$: correction factor lớn ở đầu quá trình sampling và giảm dần khi tiến tới endpoint.
- $r^{-2}(\tau) = \dfrac{\tau^2 + (1-\tau)^2}{(1-\tau)^2}$: hệ số chuẩn hóa được suy ra từ bài toán inverse guidance của Flow Matching.
- $\beta$: ngưỡng trên (maximum guidance weight), dùng để tránh correction quá lớn và giữ quá trình sampling ổn định.

Phần tính guidance scale và clipping nằm tại [`model.py` dòng 247–251](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L247).
Evaluation mặc định đặt `β = 5` tại [`eval_flow.py` dòng 29–33](../../../../../third_party/01-real-time-chunking-kinetix/src/eval_flow.py#L29).

Euler update sau đó dùng `v_RTC` thay cho velocity gốc.

Điểm quan trọng: code không dựng Jacobian đầy đủ và không tính pseudoinverse tường minh.
`jax.vjp` chỉ tính trực tiếp vector-Jacobian product `J^\top e`.

---

## Soft mask `W`

`W` là vector trọng số theo từng action index. Nó được tính từ:

- inference delay `d`;
- prefix attention horizon `e`;
- schedule được chọn.

Ba vùng của chunk mới là:

| Vùng         | Weight   | Ý nghĩa                              |
| ------------ | -------- | ------------------------------------ |
| `i < d`      | `1`      | Action đã cam kết, phải giữ chắc      |
| `d <= i < e` | giảm dần | Soft overlap, ưu tiên tính liên tục   |
| `i >= e`     | `0`      | Không còn guidance, sinh tự do        |

Các schedule được hiện thực trong
[`get_prefix_weights`, `model.py` dòng 40–63](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L40).

`W` không được học bởi model. Nó được tính trước và giữ cố định trong toàn bộ quá trình sinh
một chunk.

### Ví dụ

Giả sử:

```text
H = 8
d = 2
e = 6
```

Linear schedule:

```text
W = [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0]
```

Exponential schedule:

```text
W ≈ [1, 1, 0.571, 0.287, 0.115, 0.026, 0, 0]
```

Hard-prefix schedule:

```text
W = [1, 1, 0, 0, 0, 0, 0, 0]
```

All-ones schedule:

```text
W = [1, 1, 1, 1, 1, 1, 0, 0]
```

Exponential giảm nhanh hơn linear: nó giữ chắc các action gần hiện tại nhưng giải phóng nhanh
các action xa hơn.

---

## Ví dụ weighted error

Giả sử:

```text
Y - \hat A_1 =
[0.10, 0.10, 0.20, -0.30, 0.50, -0.20, 0.80, -0.10]
```

và:

```text
W =
[1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0, 0]
```

Khi đó:

```text
e = W ⊙ (Y - \hat A_1)

  = [0.10, 0.10, 0.16, -0.18, 0.20, -0.04, 0, 0]
```

Hai action đầu giữ nguyên toàn bộ sai số. Các action trong overlap bị giảm dần. Hai action cuối
không tạo correction.

Đây chính là intuition của soft mask:

> Không hỏi action này có bị khóa hay không, mà hỏi action này nên bám kế hoạch cũ mạnh đến mức nào.

---

## Ví dụ velocity trước và sau correction

Giả sử tại một flow timestep:

```text
v_\theta       = [0.4, 0.2, -0.2]
J^\top e      = [0.1, 0.05, 0]
\lambda(\tau)     = 2
```

Correction là:

```text
\lambda J^\top e = [0.2, 0.1, 0]
```

Velocity sau RTC:

```text
v_{RTC} = v_\theta + \lambda J^\top e
     = [0.6, 0.3, -0.2]
```

Hai action đầu được đẩy về phía chunk cũ. Action cuối không đổi vì weight của nó bằng `0`.

RTC không trực tiếp nội suy action theo dạng:

```text
new = W * old + (1 - W) * generated
```

Thay vào đó, nó dùng `W` để tạo weighted error, rồi backpropagate error đó để sửa **velocity
của flow**. Phép sửa này nằm trong cùng đoạn
[`model.py` dòng 235–251](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L235).

---

## Correction xảy ra ở từng flow timestep

Với `N = 5`, sampler chạy tại:

```text
τ = 0.0, 0.2, 0.4, 0.6, 0.8
```

Ở mỗi bước, RTC tính lại:

1. velocity mới;
2. clean chunk estimate mới;
3. residual mới với chunk cũ;
4. VJP correction mới;
5. guidance scale mới theo `τ`.

Vòng Euler có guidance được triển khai trong
[`FlowPolicy.realtime_action`, `model.py` dòng 219–265](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L219).

Quy trình:

```text
x0 → correction0 → x1
x1 → correction1 → x2
x2 → correction2 → x3
x3 → correction3 → x4
x4 → correction4 → x5
```

`Y` và `W` giữ nguyên trong một lần sinh chunk, nhưng correction thay đổi vì `xτ`, velocity,
clean estimate và Jacobian đều thay đổi theo từng flow step.

---

## Vì sao không copy cứng toàn bộ overlap?

Giả sử robot đang với tới một chiếc cốc.

Kế hoạch cũ:

```text
reach → reach → close gripper → lift
```

Trong lúc robot thực thi hai action đầu, chiếc cốc bị di chuyển.

Nếu copy cứng toàn bộ overlap:

```text
reach → reach → close gripper → lift
```

robot có thể đóng gripper vào vị trí cũ.

Với soft inpainting:

```text
reach → reach → adjust hand → close gripper → lift
```

Các action gần hiện tại vẫn liên tục, nhưng phần xa hơn có thể đổi theo observation mới.

Đây là trade-off chính:

- weight lớn: chuyển động mượt hơn nhưng phản ứng chậm hơn;
- weight nhỏ: phản ứng nhanh hơn nhưng dễ đổi chiến lược đột ngột;
- soft decay: cân bằng giữa continuity và reactivity.

---

## Liên hệ với image editing

Sự tương đồng có thể tóm tắt như sau:

| Image inpainting   | RTC                  |
| ------------------ | -------------------- |
| Ảnh gốc            | Chunk cũ             |
| Known pixels       | Action đã cam kết    |
| Soft boundary      | Soft overlap         |
| Missing region     | Free postfix         |
| Denoising guidance | Flow VJP guidance    |
| Pixel consistency  | Action continuity    |

Vì vậy, intuition đúng là:

> RTC dùng một kỹ thuật giống image editing/inpainting, nhưng áp dụng trên chuỗi action theo thời gian thay vì trên pixel không gian.

Điểm khác biệt quan trọng là ảnh thường có mask không gian, còn RTC có mask thời gian với
trọng số giảm dần theo tương lai.

---

## Hai branch trong `realtime_action`

| Điều kiện                     | Cách sampling                                   |
| ----------------------------- | ---------------------------------------------- |
| `simulated_delay is None`     | Dùng inference-time VJP guidance                |
| `simulated_delay is not None` | Dùng hard-prefix sampling, không gọi `jax.vjp`   |

Hai branch được chọn tại
[`model.py` dòng 253–260](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L253).

---

## Giới hạn

- Soft mask chỉ được dùng trong inference-time RTC guidance.
- Training-time RTC vẫn condition trên hard prefix.
- `W` là schedule thiết kế thủ công, không adaptive theo uncertainty hoặc scene change.
- `β = 5` là thiết lập thực nghiệm trong paper, không phải hằng số phổ quát.
- Khi thay đổi số flow steps, policy hoặc overlap horizon, có thể cần tinh chỉnh lại guidance.

---

## Nguồn

- [`src/model.py`](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py):
  - time embedding: [dòng 21–34](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L21);
  - prefix schedules: [dòng 40–63](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L40);
  - model input và broadcast time: [dòng 140–158](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L140);
  - sampler tiêu chuẩn: [dòng 160–171](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L160);
  - RTC VJP guidance: [dòng 219–265](../../../../../third_party/01-real-time-chunking-kinetix/src/model.py#L219).
- [`eval_flow.py` dòng 29–33](../../../../../third_party/01-real-time-chunking-kinetix/src/eval_flow.py#L29)
  cho thiết lập evaluation mặc định.
- *Real-Time Execution of Action Chunking Flow Policies*, Mục 3.1–3.3 và Phụ lục A.2, A.4:
  [PDF cục bộ](<../../../../papers/02-realtime-chunking/Real-Time Execution of Action Chunking Flow Policies.pdf>).