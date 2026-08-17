# FLARE — taxonomy ID/OOD, retry bằng augmentation, reset bằng skill riêng

## 1. Nguồn

- Tiêu đề: *FLARE: A Failure-Aware Framework for Autonomous Correction and
  Recovery in Visual-Language Robotic Manipulation*
- Tác giả: Ganlong Zhao, Zijia Tang, Xingping Chen, Zhanghui Kuang, Ye Tian,
  Guanbin Li (CUHK; CPII; Duke; Sun Yat-sen University; TengenX;
  Tencent Robotics X; Shenzhen Loop Area Institute)
- Venue: **CVPR 2026** (pp. 22391–22401, bản CVF Open Access) — venue chắc chắn
  nhất trong corpus bốn paper ban đầu.
- PDF trong repo: [docs/papers/retry-handle/flare_failure_aware_correction_recovery.pdf](../../../papers/retry-handle/flare_failure_aware_correction_recovery.pdf)
- Nguồn: https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.pdf
- Code: **Unknown** — paper không ghi repo hay project page.
- Phân loại: **failure_adaptation** (cơ chế: augmentation + reset skill riêng) —
  can thiệp chính ở tầng dữ liệu (perturbation & bridging augmentation + reset
  demo), cộng orchestration bằng MLLM lúc deploy.

⚠️ **Trùng tên:** có một paper khác cũng tên FLARE — *FLARE: Robot Learning with
Implicit World Modeling* (Zheng et al., arXiv 2505.15659), được chính SC-VLA
trích dẫn. Đó là paper world model, **không** liên quan retry/recovery. Đừng lẫn.

## 2. Câu hỏi nghiên cứu

Vì sao VLA không biết "làm lại"? Paper không quy lỗi cho kiến trúc hay control
policy mà quy về **chế độ dữ liệu**:

Demo của người là **trajectory-monotonic** — robot luôn đi trên một dải pose hẹp
tương quan chặt với tiến độ task. Model học tương quan giả giữa **pose của chính
nó** $s^r_t$ và tiến độ, thay vì suy tiến độ từ **trạng thái môi trường**
$s^e_t$. Hệ quả: khi một nhiễu nhỏ tạo ra $s^e_t$ hợp lệ nhưng $s^r_t$ lạ, policy
hiểu nhầm là OOD và fail — dù task vẫn cứu được.

Ví dụ trong paper: pose robot đang ở vị trí "đặt vật", nên policy suy ra task đã
tới giai đoạn đặt, dù trạng thái môi trường cho thấy vật **chưa từng được gắp**.

## 3. Đóng góp

1. **Taxonomy ID/OOD error** cho failure của VLA, tách thành hai cơ chế phục hồi
   khác nhau.
2. **Perturbation & bridging augmentation** tách rời pose robot khỏi trạng thái
   môi trường, tạo khả năng retry bẩm sinh.
3. Hệ **dual-loop dùng MLLM**: offline làm nhà phân tích failure để bootstrap
   reset skill, online làm monitor điều phối vòng kín.

## 4. Method

### 4.1 Taxonomy

Trạng thái $s_t = (s^e_t, s^r_t)$. Gọi $S^e_{task}$ là tập trạng thái môi trường
hợp lệ trên mọi đường thành công.

- **ID Error**: $s^e_t \in S^e_{task}$ nhưng $s^r_t$ nằm vùng xác suất thấp của
  $P(s^r_t \mid s^e_t, \mathcal{D}_{demo})$. Task vẫn cứu được từ $s^e_t$.
  → cần **Retry**.
- **OOD Error**: $s^e_t \notin S^e_{task}$ (ly đã đổ). Không action nào của task
  policy cứu được. → cần **Reset skill** riêng.

Đây là đóng góp khái niệm rõ ràng nhất trong corpus bốn paper ban đầu: nó nói tại sao
"làm lại" và "dọn hiện trường" là hai bài toán khác nhau, không thể gộp.

### 4.2 Retry — Perturbation & Bridging

1. **Trajectory composition**: ghép trajectory mới từ thư viện segment subtask do
   người demo, biến đổi động học cho khớp pose object mới (dùng MimicGen).
2. **Chèn segment** $d_i$ giữa các segment task:
   $T_{aug} = (d_{init}, a_0, d_0, a_1, d_1, \dots, a_N)$.
   - $d^A_i$ (**perturbation**): chuyển động ngẫu nhiên đưa tay tới pose OOD tuỳ
     ý. Phá tương quan pose–state.
   - $d^B_i$ (**bridging**): đưa robot từ pose bị nhiễu về pose khởi đầu hợp lệ
     của segment kế $a_{i+1}$. Sinh bằng nội suy động học không cần planner —
     nội suy tuyến tính vị trí + SLERP cho xoay.
3. **Sinh training data**: $d^A_i$ **không** làm target huấn luyện (không phải
   action có nghĩa). Chỉ thêm vào dataset:
   - chuỗi thành công gốc $(a_i, a_{i+1}, \dots)$
   - chuỗi bridging-to-task $(d^B_i, a_{i+1}, a_{i+2}, \dots)$

Train trên dạng thứ hai buộc VLA học thực thi đúng $a_{i+1}$ từ rất nhiều pose
khởi đầu khác nhau. Đây chính là cơ chế phá tương quan giả.

Tác giả chấp nhận một phần trajectory sinh ra là không hợp lệ: "lợi ích của một
dataset retry lớn và đa dạng lớn hơn nhiễu từ các trajectory hiếm và không hợp
lệ".

### 4.3 Reset — khai thác failure bằng MLLM

1. **Thu failure**: chạy model đã có retry hàng trăm episode, lưu toàn bộ video
   và trajectory (cả thành công lẫn thất bại).
2. **Phân tích offline** bằng Gemini-2.5-Pro, xuất JSON có cấu trúc:
   - `error type`: ID-Retryable / OOD-Reset Required
   - `reset target`: tên object cần reset
   - `error group`: các object tạo thành lỗi (ví dụ pod kẹt trong holder)
   - `failure timestamp`
3. **Trích state chính xác**: từ timestamp lấy trạng thái MuJoCo đầy đủ.
4. **Demo người**: 10–20 demo cho mỗi reset skill hướng object ("reset the cup"),
   bắt đầu từ đúng các error state đã khai thác.
5. **Augment lên 500**: đặt **nguyên cụm `error group`** (giữ pose tương đối bên
   trong) vào cảnh mới, rồi áp cùng perturbation & bridging. Nhờ vậy skill học
   phục hồi từ *bản chất ngữ nghĩa của lỗi* ("đang bị kẹt") chứ không từ vị trí
   cụ thể trong cảnh.

### 4.4 Huấn luyện và inference vòng kín

Không train một model đơn khối. Dùng **LoRA adapter** trên backbone π0.5 dùng
chung:

- một adapter task chính $\pi^{task}_{LoRA}$ train trên $D_{task\_aug}$
- một adapter cho **mỗi** reset skill $\pi^{reset,j}_{LoRA}$

Lý do: tránh gradient xung đột giữa các skill khác nhau; thêm skill mới = thêm
adapter, không đụng adapter cũ.

Vòng kín: MLLM monitor quan sát; **ID error → không can thiệp** (VLA tự retry);
**OOD error → MLLM đổi adapter** sang reset skill tương ứng, chạy xong xác nhận
môi trường đã hợp lệ, nạp lại adapter task chính.

### 4.5 Cấu hình

500 demo augment/task từ 10 demo người (MimicGen). Perturbation tối đa 45° xoay
và 0.5 m tịnh tiến. 20 demo người cho mỗi reset skill → augment lên 500. Finetune
language model + action expert của π0.5 bằng LoRA, Adam lr 2.5e-4 hằng số.
50 trial/task khi đánh giá.

**Mâu thuẫn nội bộ:** Implementation Details ghi perturbation "max 45°, 0.5m",
nhưng Fig. 4 kết luận cấu hình tốt nhất là $r = 30°$, $t = 0.7$ — vượt 0.5 m ở
chiều tịnh tiến. Hai chỗ không khớp.

## 5. Claim → Evidence

### 5.1 RoboMimic, 9 task contact-rich

| Method | Mean |
|---|---|
| OpenVLA | 38.0% |
| Task-conditioned DP | 41.8% |
| Subgoal-conditioned DP | 43.8% |
| Motion-conditioned DP | 46.9% |
| Subgoal Self-reflection | 48.0% |
| Phoenix | 57.8% |
| π0.5 (backbone) | 72.2% |
| Phoenix-Human (*upper bound có người sửa*) | 78.9% |
| **FLARE** | **84.0%** |

Tốt nhất ở 8/9 task. Thua ở Threading D0 (72% so với 80% của Subgoal
Self-reflection và 100% của Phoenix-Human).

**Đọc đúng mức đóng góp:** 84.0% so với Phoenix 57.8% là **+26.2**, nhưng phần
lớn khoảng cách đó đến từ backbone. So với chính π0.5 thì FLARE thêm **+11.8**.
Tác giả trình bày cả hai số — đây là điểm trung thực của paper.

Chi tiết đáng chú ý: mức cải thiện lớn hơn ở phiên bản **D1** (randomization
rộng, khó hơn) so với D0 — khớp với luận điểm decoupling pose–state.

### 5.2 Ablation reset skill

| Method | Coffee D0 | Coffee D1 | ThreePiece D0 | ThreePiece D1 |
|---|---|---|---|---|
| Ours w/o Reset | 92 | 74 | 60 | 54 |
| Ours Reset-Only | 88 | 64 | 60 | 50 |
| **Ours** | 96 | 78 | 62 | 58 |
| Ours-Oracle (người ra lệnh) | 100 | 90 | 68 | 64 |

Bỏ reset skill mất trung bình **3.5%**. Oracle với chỉ dẫn người thêm **7%** —
tức MLLM monitor còn cách xa trần, và MLLM mạnh hơn sẽ còn cải thiện.

**Retry gánh phần lớn công.** "w/o Reset" (chỉ có perturbation & bridging) đã đạt
92/74/60/54, tức riêng augmentation đã vượt π0.5. Reset skill chỉ thêm vài điểm.

### 5.3 Chất lượng reset skill — điểm yếu lộ rõ nhất

| | Reset SR (Coffee) | Reset SR (ThreePiece) | Gen. efficiency (Coffee) | Gen. efficiency (ThreePiece) |
|---|---|---|---|---|
| Object 1 (lid / T-block) | 84% | 88% | 83.7% | 48.6% |
| Object 2 (pod / U-block) | **24%** | **20%** | **11.6%** | **5.9%** |

Reset object lớn dễ cầm thì tốt; reset object nhỏ đã đổ (dựng lại pod cà phê) thì
gần như thất bại. Tác giả thừa nhận cần khả năng in-hand pose refinement khéo
hơn. Nghĩa là: **nửa "Reset" của paradigm Retry/Reset mới chỉ chạy được với
những lỗi dễ.**

### 5.4 Độ chính xác phân tích failure của Gemini-2.5-Pro (50 video có nhãn tay)

| Task | Reset/Retry | Reset Object | Timestamp |
|---|---|---|---|
| Coffee | 88% | 88% | 78% |
| ThreePiece Assembly | 96% | 78% | 66% |

Phân loại ID/OOD tốt; xác định **thời điểm** lỗi yếu nhất (66–78%).

### 5.5 Robot thật (Piper arm + RealSense D435i, 40 trial/task)

| Task | π0.5 | FLARE |
|---|---|---|
| Stack Three Blocks | 62.5% | 75.0% |
| Insert U-shaped Block | 45.0% | 55.0% |

Chỉ 10 demo người → augment lên 50. Không có state đặc quyền của simulator: dùng
Any6D để ước lượng pose object phục vụ augmentation. Tác giả nhấn mạnh việc dựng
lại failure **không** cần khớp toạ độ chính xác — thu các state "đại khái giống"
(ly đổ) là đủ, vì augmentation lo phần tổng quát hoá.

Đây là paper duy nhất trong corpus bốn paper ban đầu có **cả** kết quả sim đầy đủ **và** kiểm chứng
robot thật cho đúng cơ chế được đề xuất.

## 6. Giới hạn và điểm chưa rõ

- **Không có code.** Với pipeline gồm MimicGen + LoRA trên π0.5 + Gemini-2.5-Pro
  + Any6D, đây là hệ khó tái lập nhất trong corpus bốn paper ban đầu.
- **Reset skill chỉ thành công với object dễ** (xem 5.3). Đóng góp khái niệm mạnh
  hơn nhiều so với kết quả thực nghiệm của phần Reset.
- **Chi phí MLLM online không được báo cáo.** Không có số latency, tần suất gọi
  Gemini, hay chi phí. Với một monitor phải theo dõi liên tục, đây là khoảng
  trống lớn cho triển khai thật.
- **Perturbation cần state đặc quyền hoặc pose estimator.** Trong sim là MuJoCo
  state; thật là Any6D. Chất lượng augmentation phụ thuộc chất lượng pose
  estimation — không có ablation về điểm này.
- **Số liệu perturbation không nhất quán** (45°/0.5 m so với 30°/0.7).
- **RoboMimic là benchmark sim tương đối cũ**, và các baseline diffusion policy
  lấy lại từ Phoenix chứ không train lại.
- **Unknown:** tỉ lệ ID error so với OOD error trong thực tế. Nếu OOD hiếm thì
  toàn bộ nửa Reset (đắt: demo người + adapter riêng + MLLM monitor) có thể không
  đáng, và ablation "w/o Reset" −3.5% ủng hộ nghi ngờ này.

## 7. Liên hệ với workspace

- **FLARE trả lời được câu hỏi RaC để mở.** RaC ([long-horizon/recovery_data](../../long-horizon/recovery_data/01_rac.md))
  kết luận thành phần dữ liệu quan trọng hơn kiến trúc nhưng chưa thử trên VLA
  generalist và cần người can thiệp từng vòng. FLARE làm đúng phần đó: cùng luận
  điểm dữ liệu, trên VLA generalist (π0.5), với augmentation tự động thay cho
  người ngồi giám sát.
- **Perturbation & bridging là kỹ thuật rẻ nhất để mượn.** Nó không cần simulator
  vật lý cho phần bridging — chỉ cần nội suy tuyến tính + SLERP giữa hai pose.
  Nếu có trajectory đã segment theo subtask thì sinh được dữ liệu retry ngay ở
  tầng dataset.
- Với `vla-data-tools`: cần trường **ranh giới subtask** trong episode để cắt
  segment $a_i$, cộng nhãn phân biệt frame `perturbation` (loại khỏi target),
  `bridging` (giữ làm target) và `task`. Đây là cùng loại khoảng trống mà RaC chỉ
  ra — provenance mức segment, không phải mức episode.
- Taxonomy ID/OOD dùng được ngay làm khung phân loại failure khi inspect dataset,
  độc lập với việc có triển khai FLARE hay không.

## 8. Thử nghiệm tiếp theo

1. **Planned — chỉ tái lập nửa Retry.** Bỏ hẳn phần Reset (MLLM + skill bank).
   Sinh dữ liệu perturbation & bridging từ demo có sẵn, finetune một policy, đo
   xem success rate có tăng không. Bảng 5.2 cho thấy riêng phần này đã mang phần
   lớn lợi ích, mà chi phí bằng một phần nhỏ.
2. **Planned — đo phân bố ID/OOD trước.** Chạy một policy sẵn có, gán nhãn tay
   vài chục failure theo taxonomy ID/OOD. Nếu OOD < ~10% thì bỏ luôn nửa Reset là
   quyết định đúng.
3. **Planned — quét siêu tham số perturbation.** Paper báo cáo tối ưu tại
   $r=30°$, $t=0.7$ nhưng đánh đổi với hiệu suất sinh demo hợp lệ. Trên dữ liệu
   khác, đường cong này sẽ khác; phải quét lại, không dùng thẳng số của paper.
