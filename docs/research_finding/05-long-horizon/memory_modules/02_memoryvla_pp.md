# MemoryVLA++ — memory cộng imagination, mô hình hoá thời gian đầy đủ

## 1. Nguồn

- Tiêu đề: *MemoryVLA++: Temporal Modeling via Memory and Imagination in
  Vision-Language-Action Models*
- Tác giả: Hao Shi, Weiye Li, Bin Xie, Yulin Wang, Renping Zhou, Tiancai Wang,
  Xiangyu Zhang, Ping Luo, Gao Huang (Tsinghua BNRist / HKU / Dexmal / StepFun)
- arXiv: [2606.09827v1](https://arxiv.org/abs/2606.09827), 8 Jun 2026
- Venue: **preprint** (định dạng journal IEEE). Là bản mở rộng của
  [MemoryVLA](01_memoryvla.md) (ICLR 2026).
- Trang dự án: https://shihao1895.github.io/MemoryVLA-PP-Web
- PDF trong repo: [docs/papers/05-long-horizon/12_memoryvla_pp_memory_and_imagination.pdf](../../../papers/05-long-horizon/12_memoryvla_pp_memory_and_imagination.pdf)
- Phân loại: **memory module** + imagination. Đọc sau
  [01_memoryvla.md](01_memoryvla.md); phần memory không đổi.

## 2. Câu hỏi nghiên cứu

MemoryVLA giải "không biết đã làm tới đâu". Nhưng có một lớp task thứ hai mà
memory **không** giải được: task cần **dự đoán vật sẽ ở đâu**. Ví dụ chuẩn của
paper: *Dynamic-Conveyor Grasping* — vật di chuyển trên băng chuyền, nắm sớm quá
hoặc muộn quá đều hỏng.

Hai chế độ hỏng, hai cơ chế:

| Chế độ hỏng | Ví dụ | Cần gì |
|---|---|---|
| Thiếu memory | Button Pressing — ảnh trước/sau khi nhấn gần như giống hệt | Nhớ quá khứ |
| Thiếu imagination | Conveyor Grasping — nắm quá sớm / quá muộn | Dự đoán tương lai |

Câu hỏi: có thể ghép cả hai vào một VLA mà không trả giá tính toán quá lớn không?

## 3. Đóng góp so với bản hội nghị

1. Mở rộng từ **past-only memory** sang **full temporal modeling** (quá khứ +
   hiện tại + tương lai).
2. **Latent-space imagination**: dùng world model video sinh, **partial
   denoising**, không giải mã ra RGB.
3. **Memory-guided imagination integration**: memory quyết định phần nào của
   tưởng tượng đáng dùng.
4. Bổ sung 2 benchmark (Calvin, Libero-Plus), robot thứ ba (Dual-ARX5), loại task
   thứ ba (imagination-dependent), và **bảng đo hiệu năng inference** — thứ mà
   bản hội nghị thiếu.

## 4. Method — phần mới

Phần memory (PCMB, retrieval, gate fusion, token-merge consolidation) **giữ
nguyên** so với [01_memoryvla.md](01_memoryvla.md). Chỉ tóm lại điểm khác: nay hỗ
trợ nhiều camera $I = \{I^v\}_{v=1}^V$ và robot hai tay.

### 4.1 World model

Instantiate bằng **Stable Video Diffusion** (1.5B), pretrain trên video Internet
quy mô lớn. Theo VPP, điều kiện trên cả quan sát hiện tại $I$ và lệnh $L$ ($L$ mã
hoá bằng CLIP, tiêm vào spatio-temporal UNet qua cross-attention). Sau đó adapt
trên video thao tác robot:

$$
\mathcal{L}_{wm} = \mathbb{E}_{(I,L,x_0), \epsilon, \tau}\Big[\big\lVert W_\phi(x_\tau, \tau, I, L) - x_0 \big\rVert_2^2\Big]
$$

### 4.2 Latent imagination — không giải mã pixel

Đây là lựa chọn thiết kế cốt lõi. Thay vì decode frame RGB tương lai:

1. **Partial denoising**, lấy đặc trưng UNet trung gian đa tỉ lệ $\{U_s\}_{s=1}^S$.
2. FPN gộp thành latent token $z = \mathrm{FPN}(\{U_s\}) \in \mathbb{R}^{K \times N_z \times d_p}$,
   cộng temporal embedding học được.
3. Imagination former: learnable query $q$ làm **spatial attention** trên từng
   bước tưởng tượng, rồi **temporal attention** qua $K$ bước:

$$
\hat{q}_k = \mathrm{SpatAttn}(q_k, \bar{z}_k, \bar{z}_k), \qquad
z_{img} = \mathrm{FFN}\big(\mathrm{TempAttn}(\hat{q}_{1:K})\big)
$$

Lập luận: dự đoán pixel tốn kém và **tập trung vào độ trung thực pixel thay vì
động lực học liên quan tới điều khiển**, đồng thời truyền lỗi dự đoán thị giác
xuống sinh action. Latent tránh cả ba.

### 4.3 Memory-guided integration

Tưởng tượng vẫn có thể chứa nội dung sai hoặc không liên quan tới quyết định.
Memory-augmented perceptual token $\tilde{p}$ **truy vấn** tưởng tượng, rồi gate
quyết định trộn bao nhiêu:

$$
h = \mathrm{FFN}\big(\mathrm{CrossAttn}(\tilde{p}, z_{img}, z_{img})\big)
$$
$$
g = \sigma\big(\mathrm{MLP}(\mathrm{concat}[h, \tilde{p}])\big), \qquad
\bar{p} = g \odot \tilde{p} + (1-g) \odot h
$$

Token full temporal-aware: $F_{temp} = \{\bar{p}, \tilde{c}\}$.

Thứ tự nhân quả đáng chú ý: **memory lọc imagination**, không phải ngược lại.
Ablation xác nhận lựa chọn này (mục 5.4).

## 5. Claim → Evidence

### 5.1 Simulation

| Benchmark | MemoryVLA++ | MemoryVLA | Baseline mạnh nhất |
|---|---|---|---|
| LIBERO 5 suite | **98.4** | 96.5 | π0 94.2; CogACT 93.2 |
| — Spatial / Object / Goal | 99.8 / 100.0 / 98.2 | 98.4 / 98.4 / 96.4 | — |
| — Long-10 / Long-90 | 96.0 / 97.8 | 93.4 / 95.6 | CogACT 88.8 / 92.1 |
| SimplerEnv-Bridge | **73.9** | 71.9 | π0 68.4; CogACT 57.3 |
| Mikasa-Robo | **44.4** | 41.2 | π0 29.4 |
| Calvin ABC→D (Avg Len) | **4.29** | 4.09 | π0 3.92; UniVLA 3.80; CLOVER 3.53 |
| Libero-Plus zero-shot | **73.1** | 70.2 | RIPT-VLA 68.4; OpenVLA-OFT 67.9 |
| Libero-Plus SFT | **82.7** | 81.9 | OpenVLA-OFT 79.6 |

Calvin theo bước: 95.6 / 90.2 / 85.7 / 81.7 / 76.1 — khoảng cách nới rộng theo độ
dài chuỗi.

**Không phải mọi chỗ đều tốt hơn.** Trên Mikasa-Robo, MemoryVLA++ **tụt** ở
RememberColor5 (19 vs 30) và RememberColor9 (16 vs 20) so với MemoryVLA, dù trung
bình cao hơn nhờ ShellGameTouch (97 vs 88). Trên Libero-Plus zero-shot, nó tụt ở
Camera (36.4 vs 42.7) và Background (90.6 vs 95.0); ở chế độ SFT tụt ở Language
(71.0 vs 79.4). Paper không bàn các hồi quy này.

### 5.2 Real world — 3 robot (Franka, Dual-ARX5, WidowX)

| Nhóm task | MemoryVLA++ | MemoryVLA | CogACT | π0 | OpenVLA |
|---|---|---|---|---|---|
| General (6 task) | — | **85** | 76 | 72 | 31 |
| Long-horizon **memory**-dependent (6 task) | — | **83** | 57 | 52 | 9 |
| Long-horizon **imagination**-dependent (5 task) | **77** | 65 | 49 | 49 | — |

Task imagination-dependent: Conveyor Pick-Low/Mid/High, Conveyor Scan-Pick,
Bag Pack & Zip. So với CogACT: **+28** trung bình, lớn nhất ở Bag Pack & Zip
(+36) và Conveyor Scan-Pick (+33). So với MemoryVLA: **+12**.

Chú ý: MemoryVLA++ **chỉ được đánh giá trên nhóm imagination-dependent**; hai
nhóm còn lại vẫn dùng số của MemoryVLA. Nghĩa là chưa biết imagination có làm hại
task general hay memory-dependent hay không.

### 5.3 Ablation memory — giải quyết nghịch lý $L$

Bản hội nghị chỉ đo trên SimplerEnv và cho thấy $L = 64$ tệ như $L = 4$. Bản này
đo trên **ba** setting với **thang $L$ khác nhau cho mỗi setting**:

| Length | SimplerEnv (4/16/64) | Long-90 (8/16/32) | Real-Temporal (64/256/512) |
|---|---|---|---|
| Small | 67.7 | 94.2 | 78 |
| **Default** | **71.9** | **95.6** | **84** |
| Large | 67.7 | 95.6 | 81 |

Kết luận đúng là: **$L$ tối ưu phụ thuộc task**, không phải "memory dài thì tệ".
Real-Temporal cần $L = 256$, gấp 16 lần SimplerEnv. Nghịch lý ghi ở mục 5.3 của
[../01_tong_quan.md](../01_tong_quan.md) được giải quyết ở đây.

Các ablation memory còn lại (nay đo trên cả ba setting, cùng chiều với bản hội
nghị):

| Chiều | Biến thể | SimplerEnv | Long-90 | Real-Temporal |
|---|---|---|---|---|
| Fusion | Add / **Gate** | 67.7 / **71.9** | 93.8 / **95.6** | 78 / **84** |
| Consolidation | FIFO / **Token Merge** | 66.7 / **71.9** | 94.9 / **95.6** | 76 / **84** |
| Retrieval | không / **có** timestep PE | 69.8 / **71.9** | — | — |
| Memory type | Cognitive / Perceptual / **Both** | 63.5 / 64.6 / **71.9** | — | — |

### 5.4 Ablation imagination (Mikasa-Robo)

| Chiều | Biến thể | Avg Succ |
|---|---|---|
| Denoise step | **1** / 3 / 5 | **44.4** / 44.6 / 43.6 |
| Imagination horizon | 4 / 8 / **16** | 43.4 / 43.8 / **44.4** |
| World model | không freeze / **freeze** | 42.8 / **44.4** |
| Integration | Add / **Mem-Guided** | 41.2 / **44.4** |

**Một bước denoise là đủ.** Đây là phát hiện có giá trị thực dụng cao nhất của
paper: nó xác nhận rằng cái cần không phải là ảnh tương lai đẹp, mà là *tín hiệu
động lực học*. Tăng lên 3 bước gần như không đổi (44.6), 5 bước còn tệ hơn.

**Đóng băng world model tốt hơn** cập nhật nó trong lúc train policy — prior động
lực học thị giác nên được giữ nguyên.

### 5.5 Hiệu năng inference — bảng bù cho khoảng trống của bản hội nghị

Đo trên 300 lần chạy, bfloat16, single-view, action chunk 16.

| Method | Latency (RTX 4090) | Throughput | Latency (H20) | Throughput | GPU Memory |
|---|---|---|---|---|---|
| Baseline (CogACT) | 0.187 s | 85.6 Hz | 0.236 s | 67.8 Hz | 15.8 GB |
| MemoryVLA | 0.194 s | 82.5 Hz | 0.246 s | 65.0 Hz | 16.6 GB |
| **MemoryVLA++** | 0.241 s | 66.4 Hz | 0.301 s | 53.2 Hz | 21.7 GB |

Memory bank tốn **~4%** latency và 0.8 GB. Thêm world model đưa tổng lên **~29%**
so với baseline và +5.9 GB. Vẫn nằm trong ngân sách real-time trên GPU consumer —
tương phản mạnh với 11.10 s/bước của
[ReflectVLM](../future_prediction/02_reflective_planning.md).

### 5.6 Chất lượng world model và backbone mạnh hơn

Chất lượng sinh video (chế độ full-denoising) trung bình: PSNR 20.36, SSIM 0.794,
LPIPS 0.216, FVD 105.00, Flow-EPE 0.8829. Kém nhất trên Bridge (PSNR 17.44) và
Real-Bag Pack (PSNR 16.94, EPE 1.7672).

Đổi backbone LLaMA2 + CogACT pretraining sang **Qwen2.5 + Dexbotic pretraining**:
SimplerEnv 71.9 → **84.4**; LIBERO 96.7 → 97.0. Mức tăng trên SimplerEnv (+12.5)
**lớn hơn** toàn bộ đóng góp của cơ chế imagination (+2.0 trên cùng benchmark).

## 6. Giới hạn và điểm chưa rõ

- **Không có mục Limitations.** Với một bài journal-format 34 trang, đây là thiếu
  sót đáng kể và là lý do đủ để hạ mức tin cậy.
- **Preprint, chưa peer-review.**
- Các **hồi quy** ở mục 5.1 (RememberColor5/9, Libero-Plus Camera / Background /
  Language) không được thảo luận. Vì tất cả đều liên quan tới nhiễu thị giác hoặc
  trí nhớ màu, giả thuyết hợp lý là imagination đưa thêm nhiễu thị giác vào
  perceptual stream — nhưng paper không kiểm tra.
- **MemoryVLA++ không được đánh giá trên nhóm general và memory-dependent** ở
  real world. Không loại trừ được khả năng imagination làm hại các nhóm đó.
- Mục 5.6 cho thấy **đổi backbone mang lại lợi ích gấp 6 lần cơ chế được đề
  xuất** trên SimplerEnv. Paper trình bày đó như một "analysis" phụ, nhưng nó đặt
  câu hỏi nghiêm túc về tỉ lệ đóng góp giữa cơ chế và pretraining.
- Ablation imagination chỉ chạy trên **một** benchmark (Mikasa-Robo).
- **Missing baseline**: vẫn không so với hệ hai tầng
  ([π0.5](../hierarchical_agent/01_pi0_5.md),
  [LoHo-Manip](../hierarchical_agent/04_loho_manip.md)) trên cùng task temporal.
  Đây là lỗ hổng chung của cả dòng MemoryVLA.
- World model là SVD 1.5B **đóng băng** — nghĩa là tổng tham số hệ thống là
  7B (LLM) + 1.5B (SVD) + 0.3B (action expert). Con số này không được nêu tổng
  hợp ở đâu.

## 7. Liên hệ với workspace

- Cập nhật hai điểm trong [../01_tong_quan.md](../01_tong_quan.md): giải quyết
  nghịch lý $L$ (mục 5.3) và cung cấp số liệu latency còn thiếu (mục 9).
- Kết quả "**1 bước denoise là đủ**" là ý rẻ nhất tách ra được từ paper: nó nói
  rằng dùng world model làm *trích xuất đặc trưng động lực học* rẻ hơn nhiều so
  với dùng nó làm *bộ sinh ảnh*. Áp dụng được cho bất kỳ hệ nào định ghép video
  diffusion vào policy — bao gồm cả hướng đối lập của
  [ReflectVLM](../future_prediction/02_reflective_planning.md) (sinh ảnh đầy đủ,
  11.10 s/bước).
- Về yêu cầu dữ liệu: vẫn nhẹ (RGB + instruction), nhưng nay cần thêm **video
  thao tác để adapt world model**. Video này không cần nhãn action — dùng lại
  được chính episode đã có.
- Ba paper trong tập nay dùng world model theo ba cách khác nhau, đáng đặt cạnh
  nhau khi thiết kế:

| Paper | Sinh cái gì | Ở đâu | Tần suất | Chi phí |
|---|---|---|---|---|
| [Seer](../future_prediction/01_seer.md) | Ảnh $t+n$ | Latent, fuse vào action | Mỗi bước | Rẻ (65M trainable) |
| [ReflectVLM](../future_prediction/02_reflective_planning.md) | Ảnh pixel | Ngoài policy, để VLM phản tỉnh | Mỗi $H$ bước | 11.10 s/bước |
| MemoryVLA++ | Latent UNet | Trong policy, memory lọc | Mỗi bước | +29% latency |

## 8. Thử nghiệm tiếp theo

1. **Kiểm tra hồi quy trên nhóm memory-dependent**: chạy MemoryVLA++ trên đúng 6
   task memory-dependent mà chỉ MemoryVLA được đo. Nếu điểm tụt thì imagination
   và memory xung đột, và hệ nên bật/tắt imagination theo loại task.
2. **Tách đóng góp backbone khỏi đóng góp cơ chế**: chạy MemoryVLA (không
   imagination) trên backbone Qwen2.5 + Dexbotic. Nếu đạt gần 84.4 thì phần lớn
   khoảng cách đến từ pretraining, không từ imagination.
3. **World model làm feature extractor thuần**: bỏ hẳn SVD, thay bằng một
   predictor latent nhỏ train từ đầu trên chính dữ liệu robot (kiểu
   [Seer](../future_prediction/01_seer.md)). Nếu giữ được hiệu năng thì tiết kiệm
   1.5B tham số và phần lớn của +29% latency.
