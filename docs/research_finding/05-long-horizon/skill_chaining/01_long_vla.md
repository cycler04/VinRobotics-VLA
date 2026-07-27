# Long-VLA — phase-aware input masking để giải bài toán skill chaining

> **[SOTA-CODE — honorable mention]** Paper thuộc danh sách
> [sota_with_code.txt](../sota_with_code.txt), nhưng ở mục *"Code not yet
> released"*. Project: https://long-vla.github.io/ · Code: chưa phát hành ·
> Chỉ mục nhóm: [../02_sota_co_code.md](../02_sota_co_code.md)

## 1. Nguồn

- Tiêu đề: *Long-VLA: Unleashing Long-Horizon Capability of Vision Language
  Action Model for Robot Manipulation*
- Tác giả: Yiguo Fan, Pengxiang Ding (project lead), Shuanghao Bai, Xinyang Tong,
  Yuyang Zhu, Hongchao Lu, Fengqi Dai, Wei Zhao, Yang Liu, Siteng Huang, Zhaoxin
  Fan, Badong Chen, Donglin Wang (Westlake / Zhejiang / Xi'an Jiaotong /
  Beijing Advanced Innovation Center / UESTC)
- arXiv: [2508.19958](https://arxiv.org/abs/2508.19958)
- Venue: CoRL 2025
- PDF trong repo: [docs/papers/05-long-horizon/05_long_vla_long_horizon_capability.pdf](../../../papers/05-long-horizon/05_long_vla_long_horizon_capability.pdf)
- Phân loại: **skill chaining** (xử lý điểm nối giữa các subtask, không phải xử
  lý việc phân rã task).

## 2. Câu hỏi nghiên cứu

Phân rã task thành subtask làm giảm độ phức tạp học của từng hành vi, nhưng
**không** mô hình hoá chuyển tiếp và phụ thuộc giữa các subtask. Hệ quả là
*skill chaining problem*: dynamic coupling và lan truyền lỗi qua biên subtask.
Các cách sửa hiện có (online adaptive optimization cần reward; modular
architecture) đều **không tương thích** với paradigm offline end-to-end của VLA.
Có cách nào sửa skill chaining mà vẫn giữ được scalability và data efficiency của
VLA không?

## 3. Đóng góp

1. Nhận xét nền: mỗi subtask nên được chia nhỏ thêm thành **moving phase** và
   **interaction phase**, vì hai pha này cần nguồn thị giác khác nhau.
2. **Phase-aware input masking**: điều chỉnh input ở mức attention mask thay vì
   bỏ hẳn modality — nhờ đó vẫn train được một model hợp nhất, end-to-end.
3. **L-CALVIN**: mở rộng chuỗi task CALVIN từ 5 lên 10 bước.
4. Module **architecture-agnostic**: chứng minh trên cả MDT và HULC.

## 4. Method

### 4.1 Nghiên cứu sơ bộ: phân rã có thật sự cần không?

Tác giả tách CALVIN thành pha movement và pha interaction (điểm cắt đặt **10–15
frame trước khi trạng thái vật thể thay đổi**; trích chuỗi 64 frame; nhãn ngôn
ngữ được bổ sung lệnh chuyển động dựa trên vật thể và vị trí phát hiện được),
rồi train một **moving policy** riêng thay cho IK.

| Method | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| MDT | 93.3 | 82.4 | 71.9 | 60.9 | **51.1** |
| MDT + Moving Policy | **95.8** | **91.7** | **87.5** | **66.7** | 34.2 |

Đọc kỹ bảng này: hai model rời cải thiện mạnh ở bước 1–4 nhưng **sụp ở bước 5**
(34.2 so với 51.1). Chính đây là bằng chứng cho luận điểm của paper — phân rã
thành hai model tách rời không phải lời giải, vì nó chia nhỏ dữ liệu và không cho
train chung.

### 4.2 Long-VLA

**Phase identifier.** Mỗi trajectory được cắt tại $d$:
$\tau = [(s^M, a^M)_t,\ t \in (0,d];\ (s^Z, a^Z)_t,\ t \in [d+1, T]]$.
Action token mở rộng thêm một chiều $s_p$:

$$
[\,x,\ y,\ z,\ eu_x,\ eu_y,\ eu_z,\ s_g,\ s_p\,]
$$

$s_p = -1$ ở moving phase, $+1$ ở interaction phase. Lúc inference khởi tạo
$s_p = -1$.

**Masking.** Mỗi token có mask nhị phân $m_i \in \{0,1\}$, mở rộng thành ma trận
$M_{ij} = m_i \cdot m_j$. Với $P = QK^\top/\sqrt{C}$:

$$
A_{ij} = \frac{\exp(P_{ij})\, M_{ij}}{\sum_k \exp(P_{ik})\, M_{ik}}, \qquad 1 \le i,j \le N
$$

Quy tắc: **moving phase dùng camera third-person** (định vị vật thể chính xác;
camera gripper lúc này gần như vô nghĩa); **interaction phase dùng camera
gripper** (giảm distribution shift thị giác, thao tác chính xác).

Điểm mấu chốt về mặt kỹ thuật: masking **không đổi cấu trúc input**, nên vẫn
train chung được toàn bộ dữ liệu của cả hai pha trong một model — khác với việc
xoá hẳn modality (làm giảm dữ liệu khả dụng mỗi pha).

**Loss.**

$$
L_{diff} = \mathbb{E}_{a \sim p, \epsilon \sim \mathcal{N}(0, \sigma^2 I)}
\big\lVert D_\theta(\hat{a}_t, e_{post}, \sigma_t) - \epsilon \big\rVert,
\qquad
L = L_{diff} + \alpha L_{Goal},\ \alpha = 0.1
$$

$L_{Goal}$ là InfoNCE giữ visual goal nhất quán ngữ nghĩa với instruction.

### 4.3 Kiến trúc

| Thành phần | Lựa chọn |
|---|---|
| Observation encoder | ResNet-18 trainable cho gripper cam và static cam |
| Goal encoder | CLIP đông lạnh (text và image); dùng $s^{t+n}$ làm visual goal khi thiếu nhãn |
| Detection | Grounding DINO + LoRA finetune trên tập con CALVIN; bounding box → positional encoder → **FiLM** modulate feature static cam |
| Multimodal encoder | GPT-2 style transformer |
| Action decoder | Conditional diffusion, DDIM sampling, MLP 2 lớp GELU ở đầu ra |

## 5. Claim → Evidence

### 5.1 L-CALVIN, 10 task liên tiếp

**D→D** (dữ liệu hạn chế):

| Method | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Avg Len |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GR-1 | 0.83 | 0.58 | 0.48 | 0.35 | 0.24 | 0.17 | 0.13 | 0.09 | 0.05 | 0.04 | 2.96 |
| RoboVLMs | 0.81 | 0.60 | 0.44 | 0.34 | 0.28 | 0.15 | 0.10 | 0.08 | 0.05 | 0.03 | 2.88 |
| Base (MDT) | 0.86 | 0.64 | 0.53 | 0.47 | 0.37 | 0.31 | 0.28 | 0.21 | 0.13 | 0.11 | 4.11 |
| **Long-VLA** | 0.92 | 0.74 | 0.65 | 0.50 | 0.43 | 0.39 | 0.36 | 0.30 | 0.26 | 0.20 | **4.75–4.81** |

Mức cải thiện tương đối **tăng theo độ dài chuỗi**: +7% ở bước 1, +100% ở bước 9,
+81% ở bước 10. Đây chính là chữ ký của việc giảm lan truyền lỗi, không phải của
việc cải thiện kỹ năng đơn lẻ.

**ABCD→D** (dữ liệu đầy đủ): base 1.00→0.45, Long-VLA 1.00→0.56 ở bước 10. Mức
cải thiện nhỏ hơn khi dữ liệu dồi dào.

### 5.2 Real world

Sorting 8 bước (UR robot, RealSense L515 + USB camera), 3 điều kiện unseen:

| Điều kiện | Method | Bước 1 | 5 | 7 | 8 |
|---|---|---|---|---|---|
| Random localization | Base | 0.70 | 0.20 | 0.00 | 0.00 |
| | **Long-VLA** | 0.95 | 0.50 | 0.50 | 0.45 |
| Unseen lighting | Base | 0.50 | 0.15 | 0.00 | 0.00 |
| | **Long-VLA** | 0.80 | 0.45 | 0.30 | 0.25 |
| Visual distraction | Base | 0.55 | 0.05 | 0.00 | 0.00 |
| | **Long-VLA** | 0.85 | 0.50 | 0.40 | 0.35 |

Base policy về **0** sau bước 7 trong cả ba điều kiện; Long-VLA giữ ~25–45%.

Cleaning 4 bước (nhiều loại action hơn: nhấn, nắm, đặt; nhiễu thị giác nhiều
hơn): base 12/20 → 3/20; Long-VLA 18/20 → 11/20 (random localization). Mức cải
thiện lớn hơn ở cleaning so với sorting — tác giả quy cho khả năng chống nhiễu
thị giác của phase-aware masking.

### 5.3 Ablation (Avg Len)

| Dec. | Inp. | Uni. | Real sorting | Real cleaning | Sim D→D |
|---|---|---|---|---|---|
| ✗ | ✗ | ✓ | 2.3 | 1.4 | 4.11 |
| ✓ | ✗ | ✓ | 3.6 (+1.3) | 1.7 (+0.3) | 4.42 (+0.31) |
| ✓ | ✓ | ✗ | 4.1 (+1.8) | 2.0 (+0.6) | 4.76 (+0.65) |
| ✓ | ✓ | ✓ | **5.5 (+3.2)** | **2.8 (+1.4)** | **4.81 (+0.70)** |

Đọc bảng: ở simulation, unified model chỉ thêm +0.05 so với input adaptation
không unified. Nhưng ở **real world** thì chênh 4.1 → 5.5 và 2.0 → 2.8 — rất lớn.
Giả thuyết đọc được: unified model quan trọng khi dữ liệu ít (real world), vì nó
cho phép chia sẻ dữ liệu giữa hai pha.

### 5.4 Tính phổ quát

| Base | Avg Len gốc | + Long-VLA |
|---|---|---|
| HULC | 2.65 | 3.30 (+0.65) |
| MDT | 4.11 | 4.81 (+0.70) |

## 6. Giới hạn và điểm chưa rõ

- **Tác giả tự nêu**: phân rã pha vẫn **thủ công** (có thể tự động hoá bằng VLM);
  phạm vi task long-horizon còn hẹp; giảm được initial state gap nhưng **không**
  xử lý được thất bại thi hành khi điều kiện ban đầu đã chính xác; độ dài chuỗi
  thử nghiệm còn giới hạn.
- **Heuristic điểm cắt** "10–15 frame trước khi trạng thái vật thể thay đổi" đòi
  hỏi biết trước khi nào vật thể thay đổi trạng thái — trong CALVIN thì có task
  detector, ngoài đời thì không. Đây là rào cản thực tế lớn nhất.
- **Quy tắc camera cứng** (moving → third-person, interaction → gripper) là giả
  định về hình học cảm biến. Không rõ có tổng quát cho setup nhiều camera hoặc
  robot mobile không.
- Bản trích PDF từ file scan có OCR lỗi ở một số bảng (ví dụ Table 2 thiếu dòng
  base policy cho D→D; Figure 4 và Table 2 báo Avg Len hơi khác nhau: 4.75 vs
  4.81). **Không nên trích số lẻ từ paper này mà không kiểm tra lại bản gốc.**
- **Missing baseline**: không so với hierarchical VLA hiện đại (π0.5, Hi Robot).
  So sánh với π0 chỉ có ở real world và không có bảng số chi tiết trong bản trích.
- Chưa rõ detection branch (Grounding DINO + LoRA + FiLM) đóng góp bao nhiêu —
  nó bị gộp vào "input-level adaptation" trong ablation, không tách riêng.

## 7. Liên hệ với workspace

- Đây là paper duy nhất trong tập tấn công đúng **điểm nối giữa các subtask**,
  trong khi π0.5/Hi Robot/LoHoVLA tấn công **việc chọn subtask nào**. Hai vấn đề
  khác nhau và các paper kia không giải quyết skill chaining.
- Cơ chế masking là loại thay đổi **rẻ nhất để tích hợp** trong cả tập: không đổi
  backbone, không đổi input pipeline, chỉ đổi attention mask và thêm một chiều
  vào action vector. Nếu workspace có sẵn một policy đa camera, đây là thử nghiệm
  chi phí thấp nhất.
- Với `vla-data-tools`: yêu cầu dữ liệu là **nhãn pha theo timestep** — một
  trường nhị phân trên mỗi frame. Nhẹ hơn nhiều so với nhãn subtask dạng câu của
  π0.5/LoHoVLA. Đáng cân nhắc thêm vào canonical episode nếu đi hướng này.
- Cảnh báo về `s_p` khi inference: model tự chuyển pha bằng cách dự đoán $s_p$
  như một chiều của action. Không có cơ chế đảm bảo chuyển pha đúng lúc — đây là
  chỗ có thể hỏng âm thầm.

## 8. Thử nghiệm tiếp theo

1. **Tự động hoá điểm cắt pha**: thay heuristic "10–15 frame trước thay đổi trạng
   thái" bằng bộ phân loại pha học từ dữ liệu (hoặc từ tín hiệu gripper
   open/close + vận tốc end-effector). Đo mức mất mát so với nhãn oracle. Nếu gần
   bằng thì rào cản triển khai lớn nhất biến mất.
2. **Tách riêng detection branch**: chạy Long-VLA không có Grounding DINO/FiLM.
   Nếu phần lớn +0.65 của "input-level adaptation" đến từ detection chứ không từ
   masking thì thông điệp của paper cần sửa lại.
3. **Ghép với memory**: Long-VLA giảm lan truyền lỗi giữa các subtask nhưng không
   biết đã hoàn thành subtask nào. Ghép $s_p$ với PCMB của
   [MemoryVLA](../memory_modules/01_memoryvla.md) trên L-CALVIN 10 bước để xem
   hai cơ chế có cộng dồn không.
