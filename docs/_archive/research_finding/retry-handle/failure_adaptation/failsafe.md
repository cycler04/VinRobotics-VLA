# FailSafe — sinh failure trong sim kèm action recovery đã kiểm chứng

## 1. Nguồn

- Tiêu đề: *FailSafe: Reasoning and Recovery from Failures in
  Vision-Language-Action Models*
- Tác giả: Zijun Lin, Jiafei Duan, Haoquan Fang, Dieter Fox, Ranjay Krishna,
  Cheston Tan, Bihan Wen (NTU; A*STAR CFAR; Allen Institute for AI; UW)
- arXiv: [2510.01642v4](https://arxiv.org/abs/2510.01642), bản v4 ghi 7 Jul 2026
- Trang dự án: https://jimntu.github.io/FailSafe/
- PDF trong repo: [docs/papers/retry-handle/failsafe_reasoning_recovery_from_failures.pdf](../../../papers/retry-handle/failsafe_reasoning_recovery_from_failures.pdf)
- Venue: **Unknown** — định dạng hai cột kiểu IEEE (ICRA/IROS) nhưng PDF không
  ghi venue. Chỉ mục citation của người dùng ghi 21 citation.
- Code: **chưa có**. Paper viết "We plan to release the FailSafe code"; trang dự
  án hiện không có link GitHub (kiểm tra 31 Jul 2026).
- Phân loại: **failure_adaptation** (cơ chế: sinh dữ liệu failure–recovery) —
  can thiệp ở tầng *dữ liệu* (sinh cặp failure–action) cộng một *external
  assistant* chạy song song VLA lúc deploy.

## 2. Câu hỏi nghiên cứu

Các pipeline sinh dữ liệu failure hiện có (AHA, RoboFAC, REFLECT) chỉ tạo **mô tả
văn bản**: "gripper nên dịch sang trái để thẳng hàng với tâm khối lập phương".
Câu đó không có độ lớn, không có scale, không có điểm kết — VLA **không thực thi
được**.

Câu hỏi: có thể tự động sinh đồng thời *reasoning mức cao về lỗi* và *action sửa
lỗi mức thấp thực thi được ngay*, ở quy mô lớn, hay không?

## 3. Đóng góp

1. Pipeline tự động sinh failure + recovery action **7-DoF thực thi được**, gắn
   vào bất kỳ simulator nào có motion planning.
2. Bằng chứng dữ liệu này cho phép VLM thường học reasoning về failure và nâng
   ba VLA khác nhau, với overhead nhỏ.
3. Bằng chứng tổng quát hoá qua camera angle, object category và embodiment.

## 4. Method

### 4.1 Sinh failure

Ba loại failure cơ bản:

- **Translation failure**: nhiễu theo trục Cartesian $(x, y, z)$, biên độ ±0.1.
- **Rotation failure**: lệch góc roll/pitch/yaw, biên độ ±1 radian.
- **No-ops failure**: cánh tay đứng yên một khoảng thời gian.

Trong ManiSkill, motion planner đưa tay qua chuỗi pose theo từng stage. FailSafe
bọc một environment wrapper và một file YAML (loại failure, dải nhiễu, stage được
phép chèn), làm lệch pose của **đúng một** stage: $B \to B'$. Rollout thành
$A \to B' \to C \to D$. Nếu task thật sự fail thì ghi lại observation, trajectory
và loại failure.

Lập luận về độ phủ: nhiều failure nhiều bước (vật trượt khỏi gripper khi di
chuyển) truy về được một grasp sai ban đầu do lệch translation/rotation. Pipeline
cố ý giữ các ca **delayed failure** — task fail vài bước sau khi lỗi gốc xảy ra.

### 4.2 Thu action sửa lỗi

Không thể lấy thẳng giá trị nhiễu làm delta action ngược lại — sẽ va chạm gripper
với vật. Thay vào đó:

- Duyệt pose lệch $P_d$ trong failure trajectory **từ bước 10 tới hết**.
- Ánh xạ mỗi $P_d$ tới một pose sửa $P_c$ trong correct trajectory, giới hạn cửa
  sổ từ 10 bước sau khi bắt đầu tới 3 bước trước khi kết thúc (tránh va chạm).
- Với no-ops: $P_c$ lấy ngẫu nhiên 3–10 bước sau $P_d$.
- $\Delta A$ = hiệu 7-DoF giữa hai pose.

Điểm quan trọng: **$\Delta A$ không 1-sparse.** Loại failure chỉ dùng để đặt tên
nguồn lỗi chính; $\Delta A$ vẫn chỉnh cả 7 chiều. Thiết kế này phù hợp thực tế
nhiều lỗi xảy ra đồng thời.

### 4.3 Systematic verification

Phát lại trajectory theo $A \to P_d \to P_c \to B \to C \to D$. **Chỉ giữ cặp
failure–action nếu task vốn fail nay thành công.** Đây là điểm tách FailSafe khỏi
các dataset failure chỉ có nhãn văn bản: mỗi mẫu đã được chứng minh là recover
được trong simulator.

### 4.4 Dataset

Ba task ManiSkill: pick cube, push cube, stack cube.

| Loại | Số entry |
|---|---|
| No-ops | 26.235 |
| Trans x / y / z | 24.480 / 29.034 / 2.385 |
| Rot x / y / z | 27.807 / 17.736 / 3.363 |
| **Tổng failure** | **131k** |
| Ground truth (không lỗi) | 55.961 |

Tỉ lệ failure:success = **2.3:1**. Mỗi entry gồm 10 frame liên tiếp và 3 view
(front, side, hand).

Format hỏi–đáp: hỏi "task là X, xác định subtask hiện tại, rồi trả lời có/không
có failure, nếu có thì xuất action sửa"; đáp gồm subtask, yes/no, loại failure,
$\Delta A$.

### 4.5 FailSafe-VLM

Full instruction finetune LLaVA-OneVision-7B, co-train với RoboPoint VQA mixture.
1 epoch, **32× H100**, DeepSpeed ZeRO-3. Backbone Qwen2-7B-Instruct, vision tower
SigLIP, projector MLP 2 lớp GELU. lr 1e-5 (vision tower 2e-6), cosine decay,
warmup 3%, bf16/TF32.

### 4.6 Deploy

**Mỗi 10 bước**, FailSafe-VLM giành quyền điều khiển: nhìn observation, hỏi có
failure sắp xảy ra không. Nếu "yes" thì xuất $\Delta A$ cho robot thực thi trực
tiếp, rồi trả quyền cho VLA. Cửa sổ = 10.

Thiết lập đánh giá cố ý bất lợi cho FailSafe-VLM: camera view dùng khi test là
view **VLA đã được train**, nhưng **mới** với FailSafe-VLM. Lý do: ngoài đời khó
có camera riêng cho module giám sát.

## 5. Claim → Evidence

### 5.1 Nâng ba VLA (ManiSkill, Franka Panda, test seed khác train)

| VLA | Pick | Push | Stack | Avg | Δ |
|---|---|---|---|---|---|
| π0-FAST | 88 → 88 | 52 → 64 | 96 → 96 | 78.7 → 82.7 | **+4.0** |
| OpenVLA | 28 → 48 | 4 → 24 | 12 → 40 | 14.7 → 37.3 | **+22.6** |
| OpenVLA-OFT | 84 → 96 | 88 → 100 | 100 → 100 | 90.7 → 98.7 | **+8.0** |

**Đọc kỹ con số headline.** "+22.6%" của abstract là trên OpenVLA, baseline
**14.7%** — gần như không làm được task. Trên hai baseline mạnh, mức tăng là
+4.0 và +8.0. Số đại diện trung thực cho method là 4–8 điểm, không phải 22.6.

### 5.2 Object mới (OpenVLA-OFT)

Sphere 44 → 68, Place Sphere 36 → 52, Pick Charger 80 → 92. Avg 53.3 → 70.7
(**+17.4**). FailSafe-VLM chưa từng thấy các object này.

### 5.3 Embodiment mới (xArm 6)

VLA finetune lại trên 1.000 trajectory/task của xArm 6; **FailSafe-VLM giữ nguyên
checkpoint train trên Franka**. Stack cube 56 → 76; hai task còn lại đã 100%.
Avg 85.3 → 92.0 (+6.7). Cross-embodiment generalization là do failure/recovery
được định nghĩa trong không gian pose end-effector, độc lập embodiment.

### 5.4 So với VLM tổng quát (1.712 entry từ 20 test seed)

| Model | Binary success | Accuracy (đúng loại + trục) | Cosine similarity |
|---|---|---|---|
| Qwen2.5-VL | 0.2401 | 0.2401 | 0.0000 |
| Gemini-2.5-flash | 0.6229 | 0.1412 | −0.0121 |
| GPT-4o | 0.7007 | 0.1960 | 0.0117 |
| **FailSafe-VLM** | **0.9094** | **0.8368** | **0.6522** |

Qwen2.5-VL luôn trả "no failure" + action toàn 0 — vô dụng cho recovery. GPT-4o
và Gemini phát hiện *có lỗi* tạm được nhưng **không** biết lỗi loại gì và sinh
action gần như ngẫu nhiên (cosine ≈ 0). Đây là bằng chứng mạnh: prompt VLM
thương mại không thay thế được finetune trên dữ liệu failure có nhãn action.

Tác giả cũng lưu: cosine ~0.65 đã đủ cải thiện; không cần khớp gần tuyệt đối vì
nhiều $\Delta A$ khác nhau đều recover được.

### 5.5 Overhead

| VLA | Không FailSafe | Có FailSafe | Δ |
|---|---|---|---|
| π0-FAST | 43.3 s | 47.2 s | +3.9 s |
| OpenVLA | 112.1 s | 121.2 s | +9.1 s |
| OpenVLA-OFT | 28.8 s | 32.6 s | +3.8 s |

Phần lớn delay đến từ **simulator replanning** sau khi nhận corrective action,
không phải từ inference của VLM.

## 6. Giới hạn và điểm chưa rõ

- **Chỉ motion-level.** Tác giả tự nhận không xử lý được object-level error (vật
  đã đổ, đã rơi khỏi bàn). Đây đúng là ranh giới mà FLARE gọi là OOD error.
- **Không có thí nghiệm robot thật.** Toàn bộ kết quả trong ManiSkill. Mọi tuyên
  bố "real-world settings" chỉ là mô phỏng thiết lập real-world (chia sẻ camera
  view), không phải chạy phần cứng.
- **Ba task đều là cube.** pick/push/stack cube. Độ phủ failure mode rộng nhưng
  độ phủ task rất hẹp.
- **Overhead đo bằng thời gian episode trong sim**, phụ thuộc replanning của
  simulator. Không có số latency thuần của FailSafe-VLM mỗi lần gọi, nên không
  ước lượng được chi phí trên robot thật.
- **Cửa sổ 10 bước là hằng số**, không có ablation. Không rõ 5 hay 20 thì sao.
- **Failure được inject ở đúng một stage.** Multi-stage failure thật sự (hai
  stage lệch độc lập) không nằm trong dữ liệu, dù paper lập luận là các ca đó
  truy về được lỗi gốc đơn.
- **Code chưa release.** Với 131k mẫu và 32× H100 để train, không tái lập được
  nếu không có code và dataset.

## 7. Liên hệ với workspace

- FailSafe là **đối lập trực tiếp của RaC** ([long-horizon/recovery_data](../../long-horizon/recovery_data/01_rac.md)):
  RaC thu recovery bằng người can thiệp trên robot thật; FailSafe sinh recovery
  hoàn toàn tự động trong sim rồi verify bằng replay. Chi phí người ~0, nhưng
  giới hạn ở những gì motion planner mô hình hoá được.
- Ý tưởng dùng lại được ngay cả khi không dùng FailSafe-VLM: **systematic
  verification bằng replay**. Bất kỳ recovery action nào cũng có thể được xác
  nhận bằng cách phát lại trajectory qua pose sửa và kiểm tra task có thành công
  không. Đây là một dạng nhãn tự động chất lượng cao, khác hẳn nhãn văn bản.
- Với `vla-data-tools`: schema hiện tại **không** biểu diễn được entry của
  FailSafe. Một entry cần: 10 frame × 3 view, nhãn `failure_type`, nhãn
  `subtask`, và vector $\Delta A$ 7 chiều gắn với timestep. Đây là trường
  annotation ở mức *frame*, không phải mức episode.
- Kiến trúc "assistant chạy song song, giành quyền theo chu kỳ" tách rời khỏi
  VLA — có thể ghép với bất kỳ policy nào, kể cả policy không có code training.

## 8. Thử nghiệm tiếp theo

1. **Planned — tái lập systematic verification, không cần train.** Trong một
   simulator có motion planning, chèn nhiễu vào một stage, sinh candidate $P_c$,
   replay, đếm tỉ lệ pass. Con số này cho biết pipeline sinh dữ liệu có khả thi
   trước khi bỏ compute train VLM.
2. **Planned — đo baseline VLM trước.** Chạy GPT-4o/Gemini trên vài chục
   trajectory failure local với đúng prompt template của paper. Nếu cosine
   similarity cũng ≈ 0 như paper báo cáo thì xác nhận cần finetune; nếu không
   thì tiết kiệm được 32× H100.
3. **Planned — ablation cửa sổ.** Nếu tái lập được, thử window ∈ {5, 10, 20} để
   tìm điểm cân bằng giữa overhead và tỉ lệ bắt được lỗi. Paper để trống chỗ này.
