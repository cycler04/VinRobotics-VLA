# MemoryVLA — Perceptual-Cognitive Memory Bank cho phụ thuộc thời gian

> **[SOTA-CODE]** Paper thuộc danh sách [sota_with_code.txt](../sota_with_code.txt) —
> nhóm có mã nguồn công khai. Code: https://github.com/shihao1895/MemoryVLA ·
> Chỉ mục nhóm: [../02_sota_co_code.md](../02_sota_co_code.md)

## 1. Nguồn

- Tiêu đề: *MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action
  Models for Robotic Manipulation*
- Tác giả: Hao Shi (Tsinghua), Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu,
  Tiancai Wang, Erjin Zhou, Haoqiang Fan (Dexmal / MEGVII), Xiangyu Zhang,
  Gao Huang (corresponding)
- arXiv: [2508.19236v2](https://arxiv.org/abs/2508.19236), 30 Jan 2026
- Venue: ICLR 2026
- PDF trong repo: [docs/papers/05-long-horizon/04_memoryvla_perceptual_cognitive_memory.pdf](../../../papers/05-long-horizon/04_memoryvla_perceptual_cognitive_memory.pdf)
- Phân loại: **memory module** (module bộ nhớ gắn thêm, không thay đổi phân tầng).

## 2. Câu hỏi nghiên cứu

Manipulation là bài toán **non-Markovian**: hai quan sát giống hệt nhau có thể
đòi hỏi hành động khác nhau tuỳ vào việc trước đó đã làm gì. Ví dụ chuẩn của
paper: task *Push Buttons* — ảnh trước và sau khi nhấn nút gần như không khác
nhau, model không biết "mình nhấn chưa". VLA chỉ nhìn frame hiện tại thì giải
quyết thế nào?

Hai cách ngây thơ và lý do chúng hỏng:
1. Nối nhiều frame liên tiếp làm input VLM — self-attention có độ phức tạp bậc
   hai, giới hạn độ dài context.
2. Input dạng chuỗi frame **lệch phân phối** so với pretraining single-frame của
   chính model đó.

## 3. Đóng góp

1. Khung **Cognition-Memory-Action** lấy cảm hứng từ hệ nhớ kép của người:
   working memory (hoạt động thần kinh tức thời) + episodic memory (hippocampus,
   lưu cả *verbatim* chi tiết và *gist* ngữ nghĩa).
2. **Perceptual-Cognitive Memory Bank (PCMB)** hai luồng, với ba thao tác:
   retrieval, gate fusion, consolidation.
3. Action expert diffusion **có điều kiện trên memory**.
4. Đánh giá rộng: 3 robot, 6 benchmark, 150+ task, 500+ biến thể.

## 4. Method

### 4.1 Working memory

- Vision: DINOv2 và SigLIP song song trên **một** ảnh third-person RGB 224×224;
  nối feature thành raw visual token.
- Nén: module SE-bottleneck ép thành perceptual token $p \in \mathbb{R}^{N_p \times d_p}$
  với $N_p = 256$.
- Cognition: raw visual token chiếu tuyến tính vào không gian embedding ngôn ngữ,
  nối với instruction đã tokenize, đưa vào LLaMA-7B. Lấy output ở vị trí EOS làm
  **một** cognitive token $c \in \mathbb{R}^{1 \times d_c}$.
- Working memory $M_{wk} = \{p, c\}$. Backbone là Prismatic VLM 7B đã pretrain
  thêm trên Open-X Embodiment.

Đối xứng đáng chú ý: 256 token cho chi tiết tri giác, **1** token cho ngữ nghĩa.
Đây là lựa chọn thiết kế mạnh và paper không ablate tỉ lệ này.

### 4.2 Memory Retrieval

Mỗi entry được gắn timestep qua sinusoidal embedding $TE(\cdot)$, cộng vào **key
nhưng không vào value**:

$$
K^x = [\,m^x_1 + TE(t_1);\ \dots;\ m^x_L + TE(t_L)\,], \qquad
V^x = [\,m^x_1;\ \dots;\ m^x_L\,]
$$

$$
\hat{H}^x = \mathrm{softmax}\!\left(\frac{q^x (K^x)^\top}{\sqrt{d_x}}\right) V^x,
\qquad q^x \in \{p, c\},\ x \in \{per, cog\}
$$

Hai layer transformer (attention + FFN) cho ra $H^p$ và $H^c$.

### 4.3 Memory Gate Fusion

$$
g^x = \sigma\big(\mathrm{MLP}(\mathrm{concat}[x, H^x])\big), \qquad
\tilde{x} = g^x \odot H^x + (1 - g^x) \odot x
$$

Gate học được, không phải cộng thẳng — ablation cho thấy khác biệt 4.2 điểm.

### 4.4 Memory Consolidation

Khi số entry vượt $L$, trong **mỗi luồng** tính cosine similarity giữa các entry
kề nhau và gộp cặp giống nhau nhất bằng trung bình:

$$
i^*_x = \arg\max_{i=1..L-1} \cos(\tilde{x}_i, \tilde{x}_{i+1}), \qquad
m^x_{i^*_x} \leftarrow \tfrac{1}{2}\big(\tilde{x}_{i^*_x} + \tilde{x}_{i^*_x+1}\big)
$$

Đây là *token merge*, không phải FIFO — nó giữ lại các mốc thay đổi và nén các
đoạn tĩnh.

### 4.5 Action expert

DiT ~300M, DDIM 10 bước, CFG scale 1.5, sinh $T = 16$ action 7-DoF. Mỗi bước
denoise: token action nhiễu + sinusoidal encoding của denoising timestep, nối
với $\tilde{c}$; một **cognition-attention layer** đưa ngữ nghĩa cấp cao vào, một
**perception-attention layer** bổ sung chi tiết từ $\tilde{p}$; rồi FFN. Loss MSE.

Train: 8×A100, PyTorch FSDP, 32 sample/GPU (global batch 256), LR $2\times10^{-5}$.
**Chỉ dùng một camera third-person, không wrist view, không proprioception.**

## 5. Claim → Evidence

### 5.1 Simulation

| Benchmark | MemoryVLA | Baseline mạnh nhất | Chênh |
|---|---|---|---|
| SimplerEnv-Bridge (WidowX) | **71.9** | π0-Beta* 68.4; CogACT-Large 57.3 | +14.6 so với CogACT |
| SimplerEnv-Fractal overall | **72.7** | CogACT 68.1 | +4.6 |
| — Visual Matching | 77.7 | CogACT 74.8 | +2.9 |
| — Visual Aggregation | 67.7 | CogACT 61.3 | +6.4 |
| LIBERO (5 suite) | **96.5** | π0* 94.2; CogACT 93.2 | +3.3 so với CogACT |
| — Long-10 / Long-90 | 93.4 / 95.6 | CogACT 88.8 / 92.1 | — |
| Mikasa-Robo | **41.2** | π0 29.4 | +11.8 |
| — ShellGameTouch | 88 | OpenVLA-OFT 47 | +41 |

Điểm đáng chú ý: mức tăng lớn nhất nằm ở **Visual Aggregation** (đổi background,
ánh sáng, distractor, texture bàn) và ở **Mikasa-Robo** (benchmark thiết kế riêng
cho memory). Đây là kiểu bằng chứng nhất quán với cơ chế được đề xuất, không phải
tăng đều đặn mọi nơi.

### 5.2 Real world (12 task, Franka + WidowX, chỉ RGB third-person)

| Nhóm | MemoryVLA | CogACT | π0 | OpenVLA |
|---|---|---|---|---|
| General (6 task) | **85** | 76 | 72 | 31 |
| Long-horizon Temporal (6 task) | **83** | 57 | 52 | 9 |

Per-task ở nhóm temporal: Seq. Push Buttons 58 vs 15 (**+43**), Change Food 85 vs
47 (+38), Guess Where 72 vs 40 (+32), Clean Table & Count 84 vs 67 (+17), Pick
Place Order 100, Clean Restaurant Table 96.

Khoảng cách general (+9) so với temporal (+26) là bằng chứng mạnh nhất của paper:
lợi ích tỉ lệ với mức độ phụ thuộc thời gian của task.

### 5.3 Ablation (SimplerEnv-Bridge, avg success %)

| Chiều ablate | Biến thể | Kết quả |
|---|---|---|
| Loại memory | Cognitive only / Perceptual only / **Both** | 63.5 / 64.6 / **71.9** |
| Độ dài $L$ | 4 / **16** / 64 | 67.7 / **71.9** / 67.7 |
| Retrieval | không / **có** timestep PE | 69.8 / **71.9** |
| Fusion | Add / **Gate** | 67.7 / **71.9** |
| Consolidation | FIFO / **Token merge** | 66.7 / **71.9** |

Hai luồng cộng lại (71.9) **vượt tổng phần riêng lẻ** so với từng luồng (63.5,
64.6) — chúng bổ sung nhau chứ không dư thừa.

## 6. Giới hạn và điểm chưa rõ

- **$L = 64$ tệ hơn $L = 16$** (67.7 vs 71.9). Paper không giải thích. Đây là dấu
  hiệu memory dài gây nhiễu chứ không đơn thuần "nhớ nhiều hơn thì tốt hơn" —
  mâu thuẫn nhẹ với động lực long-horizon của chính paper. Cần đọc kỹ trước khi
  suy ra "cứ tăng memory".
- Ablation chỉ chạy trên **một** benchmark (Bridge). Không rõ $L=16$ có tối ưu
  cho task dài hơn không.
- **Không có ablation tỉ lệ token** 256 perceptual : 1 cognitive.
- Backbone 7B + action expert 300M — chi phí inference không được báo cáo. Không
  có số liệu latency, khác hẳn Hi Robot.
- **Missing baseline**: không so với hệ hai tầng (π0.5, Hi Robot) trên cùng task
  temporal. Câu hỏi "memory bank hay hierarchical planning giải quyết
  long-horizon tốt hơn" chưa được trả lời.
- Real-world dùng step-wise scoring với 10–15 trial mỗi task — cỡ mẫu nhỏ.
- Consolidation chỉ gộp entry **kề nhau**; sự kiện quan trọng lặp lại cách xa
  nhau về thời gian không được gộp hay ưu tiên.

## 7. Liên hệ với workspace

- Đây là câu trả lời trực tiếp cho giới hạn mà **cả π0.5 lẫn Hi Robot tự nêu**:
  "context ngắn, không có memory". Ba paper ghép lại thành một luận điểm rõ:
  hierarchy giải quyết *phân rã task*, memory giải quyết *xác định trạng thái đã
  làm tới đâu*. Chúng trực giao.
- Với dataset tooling: MemoryVLA chỉ cần **một camera third-person + instruction**
  — nhẹ nhất trong cả tập paper về yêu cầu dữ liệu. Không cần nhãn subtask, không
  cần wrist camera, không cần proprioception. Đây là điểm mạnh thực dụng cho
  workspace vì canonical episode v0.1 hiện có đủ trường cần thiết.
- Cơ chế PCMB độc lập backbone; về nguyên tắc có thể gắn vào action expert flow
  matching đang mô tả ở [03-vla-core](../../03-vla-core/).

## 8. Thử nghiệm tiếp theo

1. **Kiểm chứng nghịch lý $L$**: quét $L \in \{4,8,16,32,64,128\}$ trên
   LIBERO-Long-10 (dài hơn Bridge). Nếu điểm tối ưu dịch theo độ dài task thì
   $L$ là siêu tham số phụ thuộc task; nếu vẫn tụt ở $L$ lớn thì cơ chế
   consolidation là nút thắt, không phải capacity.
2. **Đối chứng memory vs hierarchy**: chạy MemoryVLA và một hệ hai tầng kiểu π0.5
   trên cùng bộ task "temporal" (Seq. Push Buttons, Clean Table & Count). Giả
   thuyết cần bác bỏ: hierarchy đủ để thay memory.
3. **Ablate tỉ lệ token**: giảm $N_p$ từ 256 xuống 64 và tăng cognitive token từ
   1 lên 8. Nếu hiệu năng không đổi thì chi phí retrieval (attention trên
   $L \cdot N_p$ token) đang bị lãng phí.
