# SC-VLA — residual RL online với reward nội sinh từ sparse world imagination

## 1. Nguồn

- Tiêu đề: *Self-Correcting VLA: Online Action Refinement via Sparse World
  Imagination*
- Tác giả: Chenyv Liu, Wentao Tan (đồng tác giả chính), Lei Zhu, Fengling Li,
  Jingjing Li, Guoli Yang, Heng Tao Shen (Tongji University; University of
  Technology Sydney; UESTC; Advanced Institute of Big Data)
- arXiv: [2602.21633v1](https://arxiv.org/abs/2602.21633), 25 Feb 2026
- Venue: **Preprint** (bản PDF ghi rõ "Preprint Version.", format ICML)
- Code: https://github.com/Kisaragi0/SC-VLA (**Verified** ở mức URL tồn tại,
  push gần nhất 14 Apr 2026; chưa clone hay chạy)
- PDF trong repo: [docs/papers/06-retry-handle/03_self_correcting_vla_sparse_world_imagination.pdf](../../../papers/06-retry-handle/03_self_correcting_vla_sparse_world_imagination.pdf)
- Phân loại: **training** (cơ chế: online refinement bằng RL) — thêm một policy
  residual học online bằng RL trên nền một base policy đóng băng, với reward sinh
  từ chính dự đoán của model.

⚠️ **Lưu ý ngữ nghĩa:** "self-correcting" ở đây **không** có nghĩa phát hiện lỗi
rồi phục hồi. Nó nghĩa là *policy tự tinh chỉnh action của mình online*. Hai paper
ở [failure_adaptation/](../failure_adaptation/) mới nói về failure/recovery theo
nghĩa đen; SC-VLA thì không. Xếp vào `training/` vì cái nó tạo ra là **một policy
mới** — thứ tồn tại sau khi train xong, không phải một bộ phận chạy lúc deploy.

## 2. Câu hỏi nghiên cứu

RL cho VLA hầu như luôn cần **reward ngoài**: hoặc do người thiết kế tay, hoặc do
một VLM/world model bên ngoài sinh ra. Cả hai đều tách rời khỏi trạng thái nội
tại của policy và tốn thêm hạ tầng.

World action model có sẵn khả năng dự báo tương lai, nhưng dự báo đó là biểu diễn
ngầm, không có cơ chế nào biến nó thành tín hiệu cải thiện policy.

Câu hỏi: có thể lấy **chính dự đoán tương lai của model** làm reward để tinh
chỉnh action online, bỏ hẳn reward model bên ngoài?

## 3. Đóng góp

1. **Sparse World Imagination (SPI)**: hai head phụ dự báo tiến độ task và biến
   thiên trạng thái ngắn hạn, buộc policy mã hoá tiến hoá vật lý trước khi sinh
   action.
2. **Online Action Refinement (OAR)**: residual RL với dense reward xây từ chính
   dự đoán của SPI — reward **nội sinh**, không cần supervision ngoài.
3. Đánh giá trên 4 task ManiSkill3 và robot ARX5 thật.

## 4. Method

### 4.1 Nền

Base policy là flow matching (GR00T N1.5, DiT backbone). Đường nội suy
$x_t = t x_1 + (1-t)x_0$, target velocity $x_1 - x_0$, loss MSE. Inference giải
ODE bằng Euler.

Điều kiện đa phương thức: ảnh multi-view qua SigLIP-2, ghép instruction, đưa qua
Eagle-2; lấy hidden state ở **lớp trung gian** $l$ làm $o_{mid}$.

### 4.2 Sparse World Imagination

Query sequence mở rộng:

$$q_{input} = [s_t,\ q_{p_t},\ q_{\Delta s_t},\ q_a]$$

- $s_t$: state embodiment hiện tại (1×D)
- $q_a$: 16 query sinh action
- $q_{p_t}$: dự báo **tiến độ task** $p_t$
- $q_{\Delta s_t}$: dự báo **biến thiên trạng thái ngắn hạn**

$\Delta s_t \in \mathbb{R}^7$ được tính trong **hệ toạ độ cục bộ hiện tại**, tại
thời điểm tương lai $t' = t + H + \delta$ với $H$ là execution horizon và
$\delta \sim U(-\Delta, \Delta)$ là offset ngẫu nhiên cho robustness:

$$\Delta s_t = \big[R_t^\top (P_{t'} - P_t),\ \mathrm{Euler}(R_t^\top R_{t'}),\ g_{t'} - g_t\big]$$

Hai head MLP nhẹ đọc hidden $h^{(m)}$ từ block trung gian $m$ (block cuối lo
action, block giữa còn giữ biểu diễn world state):

$$\hat{p}_t = f_{prog}(h^{(m)}),\qquad \widehat{\Delta s}_t = f_{\Delta s}(h^{(m)})$$

$$\mathcal{L}_{total} = \mathcal{L}_{FM} + \lambda_1 \mathcal{L}_{prog} + \lambda_2 \mathcal{L}_{\Delta s_t}$$

Cả hai loss phụ là MSE. Đây là điểm khác với world model kiểu pixel (GR-MG) hay
latent (FLARE của Zheng et al.): dự báo **thưa và có ngữ nghĩa vật lý đọc được**
(dịch chuyển, xoay, gripper), không phải ảnh hay latent.

### 4.3 Online Action Refinement

Base policy **đóng băng**. Action cuối:

$$a_t = a^{base}_t + \lambda a^{res}_t$$

Residual policy $\pi_{res}$ là Gaussian MLP nhẹ (theo Policy Decorator), train
bằng SAC, quan sát **không** phải ảnh thô mà là

$$o_w = (s_t, \hat{p}_t, \widehat{\Delta s}_t) \in \mathbb{R}^{16}$$

Đây là lựa chọn thiết kế then chốt: residual explore trong không gian 16 chiều đã
được base policy tóm tắt, không phải trong không gian quan sát thô.

**Dense reward nội sinh.** Lấy 3 thành phần tịnh tiến của $\widehat{\Delta s}_t$
làm mục tiêu ngắn hạn $P_{goal} = P_t + \widehat{\Delta s}^{pos}_t$, rồi thưởng
theo cosine giữa dịch chuyển thực tế và hướng dự báo:

$$r^{guide}_t = \frac{(P_{t+n} - P_t) \cdot (P_{goal} - P_t)}{\lVert P_{t+n} - P_t \rVert \lVert P_{goal} - P_t \rVert + \epsilon}$$

**Dynamic weight scheduling.** Prior dự báo chỉ tốt ở giai đoạn đầu; giai đoạn
contact tinh thì nó cản trở. Dùng chính $\hat{p}_t$ làm tín hiệu lập lịch:

$$r^{final}_t = \eta(\hat{p}_t)\cdot w_{guide}\cdot r^{guide}_t + r^{env}_t - c$$

với $\eta$ giảm đơn điệu theo tiến độ, $c$ là phạt thời gian mỗi bước.

### 4.4 Giao thức train ba pha

1. **Buffer warm-up**: $\lambda = 0$, chỉ chạy base policy để nạp replay buffer.
2. **Residual injection**: $\lambda$ tăng tuyến tính lên mức đích, tránh residual
   ngẫu nhiên phá vòng điều khiển.
3. **Main training**: $\lambda$ cố định.

Các đường cong trong Fig. 4 **chỉ vẽ pha 3**; tác giả nói rõ đã loại bỏ hai pha
đầu khỏi hình.

### 4.5 Chi phí

Stage I: 50.000 iteration, batch 32, lr 1e-4, AdamW, 1× L40.
Stage II (SAC): buffer = tổng số bước, $\gamma$ 0.97, $\tau$ 0.01, batch 1024,
lr 1e-4, $\alpha$ 0.2, UTD 0.5, $w_{guide}$ 0.6. Số bước môi trường theo task:
StackCube/PlaceSphere/LiftPegUpright 500k–600k, **PegInsertion 3.000.000**.

## 5. Claim → Evidence

### 5.1 ManiSkill3 (100 demo/task, 50 episode đánh giá)

| Model | StackCube | PlaceSphere | LiftPegUpright | PegInsertion | Avg |
|---|---|---|---|---|---|
| DP‡ (multi-task) | 0.46 | 0.90 | 0.10 | 0.00 | 0.36 |
| DP† (specialist) | 0.88 | 1.00 | 0.80 | 0.40 | 0.77 |
| ACT‡ | 0.50 | 0.88 | 0.60 | 0.12 | 0.52 |
| ACT† | 0.64 | 0.90 | 0.46 | 0.04 | 0.51 |
| π0 | 0.66 | 0.86 | 0.48 | 0.22 | 0.55 |
| GR00T N1.5 (base) | 0.78 | 1.00 | 0.72 | 0.40 | 0.72 |
| **SC-VLA (SPI)** | 0.96 | 1.00 | 0.82 | 0.50 | **0.82** |
| **SC-VLA (SPI+OAR)** | 1.00 | 1.00 | 0.88 | 0.56 | **0.86** |

SPI một mình đã +10 điểm so với base GR00T N1.5. OAR thêm +4.

### 5.2 Độ dài episode thành công (throughput)

| Model | Avg length |
|---|---|
| π0 | 276 |
| GR00T N1.5 | 195 |
| SC-VLA (SPI) | 187 |
| DP† | 172 |
| **SC-VLA (SPI+OAR)** | **157** |

OAR giảm 16% so với SPI; PegInsertion giảm mạnh nhất (262 → 173, −34%).

**Mâu thuẫn cách trình bày:** abstract ghi "16% fewer steps ... than the
best-performing baselines". Con số 16% là delta **SPI → OAR trong chính method**,
không phải so với baseline. So với baseline mạnh nhất (DP† 172) thì mức giảm là
8.7%. Phần "9% higher success rate" (0.86 so với DP† 0.77) thì đúng.

### 5.3 Ablation SPI

| Variant | Avg |
|---|---|
| w/o $\Delta s$ | 0.78 |
| w/o progress | 0.80 |
| w/o cả hai | 0.72 |
| SC-VLA (SPI) đầy đủ | 0.82 |

Bỏ cả hai cho đúng 0.72 = base GR00T N1.5, xác nhận toàn bộ mức tăng của Stage I
đến từ hai head phụ. Hai thành phần **bổ trợ nhau**: bỏ riêng mất 2–4 điểm, bỏ cả
hai mất 10 điểm. $\Delta s$ quan trọng hơn progress, đặc biệt ở task nhạy contact
(PegInsertion 0.50 → 0.42).

### 5.4 Ablation reward và scheduling

- Bỏ imagination reward: PlaceSphere gần như không đổi (base policy đã mạnh);
  PegInsertion mắc kẹt — dense reward kéo số bước trung bình từ 800 xuống 650.
  Vai trò chính của reward này là **phá cold-start**, không phải nâng trần.
- Thay dynamic weight bằng hằng số: task chính xác cao suy giảm ở giai đoạn cuối
  (số bước phân kỳ hoặc dừng ở nghiệm dưới tối ưu). Xác nhận prior tĩnh xung đột
  với điều khiển tinh.

### 5.5 Robot thật (ARX5, 60 demo/task, 20 trial)

| Model | StackCube | PlaceSphere | PushCube | PegInsertion | Avg |
|---|---|---|---|---|---|
| DP‡ | 0.30 | 0.40 | 0.45 | 0.00 | 0.28 |
| GR00T N1.5 | 0.75 | 0.45 | 0.80 | 0.30 | 0.57 |
| **SC-VLA (SPI)** | 0.85 | 0.60 | 1.00 | 0.40 | **0.71** |

## 6. Giới hạn và điểm chưa rõ

- **OAR chưa từng chạy trên robot thật.** Bảng real-world chỉ có SC-VLA **(SPI)**.
  Nửa "online refinement" — tức đóng góp mang tên bài báo — mới chỉ được xác minh
  trong simulator. Tác giả nêu lý do "khó thiết kế reward thật" nhưng chính paper
  lại tuyên bố reward là nội sinh; hai điều này không nhất quán.
- **Chi phí online rất lớn.** PegInsertion cần **3 triệu** bước môi trường cho
  residual. Đây là con số chỉ khả thi trong sim.
- **Không xử lý failure theo nghĩa recovery.** Không phát hiện lỗi, không phân
  loại lỗi, không reset. Nếu mục tiêu là retry/recovery thì SC-VLA giải bài toán
  khác — nó nâng độ chính xác để lỗi ít xảy ra hơn, chứ không xử lý lỗi đã xảy ra.
- **Residual scale phải chỉnh theo task** (0.01–0.1 train, 0.005–0.03 eval) theo
  nguyên tắc nghịch với chất lượng base policy. Tức là cần biết trước base policy
  mạnh hay yếu ở task đó — một dạng tuning thủ công mà paper không tính vào chi
  phí.
- **4 task, một benchmark, một base model.** Không có bằng chứng SPI chuyển được
  sang backbone khác GR00T N1.5.
- **Reward nội sinh có rủi ro tự khẳng định.** Nếu $\widehat{\Delta s}_t$ sai,
  reward sẽ thưởng cho việc đi theo hướng sai. Dynamic weight scheduling giảm nhẹ
  điều này ở giai đoạn cuối, nhưng paper **không** đo độ chính xác của head
  $\Delta s$ một cách độc lập. Đây là khoảng trống đánh giá quan trọng nhất.
- **Baseline π0 yếu bất thường** (0.55, thấp hơn cả DP† 0.77). Với 100 demo/task
  thì π0 có thể chưa được finetune đủ; so sánh này nên đọc dè dặt.

## 7. Liên hệ với workspace

- SPI là phần **rẻ và dễ mượn nhất**: hai head MLP đọc hidden state trung gian,
  hai loss MSE. Không cần RL, không cần simulator, không cần dữ liệu mới — nhãn
  $p_t$ và $\Delta s_t$ **tính được từ chính trajectory demo đã có**.
- Đây là điểm nối rõ với [05-long-horizon/future_prediction](../../05-long-horizon/future_prediction/):
  SPI thuộc cùng họ với Seer (dự báo latent ảnh tương lai) và ACoT-VLA (dự báo
  action tương lai), nhưng dự báo đại lượng thưa nhất — 7 số. Nếu đã có báo cáo
  so sánh modality dự báo thì SC-VLA bổ sung điểm cực rẻ nhất của phổ đó.
- Với `vla-data-tools`: sinh nhãn SPI chỉ cần pose end-effector và gripper theo
  timestep, cộng chỉ số bước / độ dài episode để tính $p_t$. **Đây là nhãn duy
  nhất trong cả tập 4 paper mà schema hiện tại đủ để tạo, không cần trường mới.**
- OAR thì không: cần vòng lặp môi trường online, tức simulator hoặc robot chạy
  triệu bước. Ngoài phạm vi workspace hiện tại.

## 8. Thử nghiệm tiếp theo

1. **Planned — tạo nhãn SPI offline và kiểm tra tính khả thi.** Từ một dataset
   trong `dataset/`, tính $p_t$ và $\Delta s_t$ theo công thức mục 4.2 cho toàn
   bộ frame. Kiểm tra: pose có đủ độ chính xác không, $\Delta s$ có bị nhiễu chi
   phối ở $H$ đang dùng không. Rẻ, không cần GPU.
2. **Planned — đo độ chính xác head $\Delta s$ riêng.** Chỗ paper để trống. Train
   một head dự báo trên dữ liệu local, đo error theo horizon. Nếu error lớn thì
   reward nội sinh của OAR không đáng tin và phần OAR nên bỏ.
3. **Planned — nếu tái lập, chỉ tái lập Stage I.** Bảng 5.1 cho thấy SPI mang 10
   trong 14 điểm cải thiện, với chi phí bằng vài phần trăm của OAR (50k iteration
   trên một GPU so với 0.5–3 triệu bước môi trường).
