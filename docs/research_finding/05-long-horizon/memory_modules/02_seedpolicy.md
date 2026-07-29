# SeedPolicy — trạng thái ẩn tự tiến hoá, giải nút thắt horizon của Diffusion Policy

> **[SOTA-CODE]** Paper thuộc danh sách [sota_with_code.txt](../sota_with_code.txt) —
> nhóm có mã nguồn công khai. Code: https://github.com/Youqiang-Gui/SeedPolicy
> (bản PDF trỏ tới link ẩn danh `anonymous.4open.science/r/SeedPolicy-64F0/`) ·
> Chỉ mục nhóm: [../02_sota_co_code.md](../02_sota_co_code.md)

## 1. Nguồn

- Tiêu đề: *SeedPolicy: Horizon Scaling via Self-Evolving Diffusion Policy for Robot Manipulation*
- Tác giả: Youqiang Gui (Sichuan University), Yuxuan Zhou (independent), Shen
  Cheng, Haoqiang Fan (Dexmal Inc.), Xinyang Yuan, Peng Cheng (Sichuan
  University), Shuaicheng Liu (UESTC)
- arXiv: [2603.05117v3](https://arxiv.org/abs/2603.05117), 8 May 2026
- Venue: **preprint**
- PDF trong repo: [docs/papers/05-long-horizon/07_seedpolicy_self_evolving_diffusion.pdf](../../../papers/05-long-horizon/07_seedpolicy_self_evolving_diffusion.pdf)
- Phân loại: **memory module**. Paper tự đối chiếu trực tiếp với
  [MemoryVLA](01_memoryvla.md) — xem mục 5.4.

## 2. Câu hỏi nghiên cứu

Xuất phát từ một quan sát rất cụ thể và ít được để ý: **Diffusion Policy tụt hiệu năng khi tăng observation horizon xếp chồng.** \Chính tác giả DP có ghi nhận hiện tượng phản trực giác này trong phụ lục, nhưng **không giải thích cũng không giải quyết**.

SeedPolicy chẩn đoán và sửa. Chuỗi lập luận rất sạch, đáng ghi lại nguyên vẹn:

1. Xếp chồng frame **không** bắt được phụ thuộc thời gian phức tạp; càng nhiều
   frame càng rõ.
2. Thêm temporal self-attention trên đặc trưng đã xếp chồng → cải thiện. Xác nhận
   rằng mô hình hoá thời gian tường minh khai thác lịch sử tốt hơn xếp chồng thuần.
3. Nhưng chi phí attention tăng **bậc hai** theo horizon, và kéo dài horizon cho
   lợi ích giảm dần → không scale được.
4. → Cần cơ chế **cập nhật kiểu đệ quy**: một trạng thái ẩn tiến hoá theo thời
   gian, nén liên tục ngữ cảnh lịch sử.

Vấn đề phụ: thông tin có giá trị **thưa theo thời gian**. Không phải quan sát nào
cũng đóng góp. Nhiễu thị giác (nền đổi, che khuất) mà tích hợp bừa sẽ làm ô nhiễm
ngữ cảnh lịch sử → cần cổng lọc.

## 3. Đóng góp

1. **SEGA** (Self-Evolving Gated Attention): module thời gian kết hợp attention
   với cổng động, duy trì trạng thái ẩn nén, bắt phụ thuộc dài hạn đồng thời lọc
   nhiễu thời gian.
2. **Đảo ngược xu hướng suy giảm** theo horizon của diffusion policy: hiệu năng
   nay **tăng** khi horizon hiệu dụng tăng.
3. **SeedPolicy**: SOTA trong nhóm imitation learning trên RoboTwin 2.0, với ít
   hơn tới 36× tham số so với VLA tỉ tham số.

## 4. Method

### 4.1 Khung tổng thể

$$
O_t = \text{Encoder}(I_t, P_t) \quad (1) \qquad\qquad S_t = \text{Update}(S_{t-1}, O_t) \quad (2)
$$

$$
EObs_t = \text{Retrieve}(O_t, S_{t-1}) \quad (3) \qquad A_t = \text{Diffusion}(EObs_t) \quad (4)
$$

Trạng thái ẩn $S_{t-1} \in \mathbb{R}^{N_s \times D}$. **Cửa sổ quan sát mỗi bước
giữ nguyên cố định**; trường tiếp nhận thời gian hiệu dụng được mở rộng bằng cách
tích luỹ vào trạng thái tiến hoá. Đây là khác biệt cốt lõi so với mọi cách xếp
chồng frame.

### 4.2 SEGA — thiết kế hai luồng song song

**Luồng State Update** (cập nhật trạng thái):

$$
S'_{t-1} = S_{t-1} + \text{MSA}(S_{t-1}), \qquad O'_t = O_t + \text{MSA}(O_t)
$$

$$
(\text{Inter} \cdot S_t,\ A) = \text{CA}(S'_{t-1},\ O'_t,\ O'_t)
$$

Trạng thái làm **Query**, quan sát làm Key/Value.

**Luồng State Retrieval** (truy hồi, vai trò đảo ngược):

$$
EObs_t = \text{CA}(O'_t,\ S'_{t-1},\ S'_{t-1})
$$

Quan sát làm **Query**, trạng thái làm Key/Value. Cho phép khôi phục thông tin đã
mất do phụ thuộc dài hạn.

### 4.3 Self-Evolving Gate (SEG) — điểm mới thật sự

Cổng **không** học từ một MLP riêng mà **đọc trực tiếp logit cross-attention
trước softmax** như tín hiệu độ liên quan:

$$
R = \frac{1}{L \cdot H \cdot N_o} \sum_{l=1}^{L} \sum_{h=1}^{H} \sum_{j=1}^{N_o} A^{(l,h)}_{:,j},
\qquad G_t = \sigma(R)
$$

$$
S_t = G_t \odot (\text{Inter} \cdot S_t) + (1 - G_t) \odot S_{t-1}
$$

So sánh với [MemoryVLA](01_memoryvla.md): công thức fusion **giống hệt về hình
thức** ($g \odot H + (1-g) \odot x$), nhưng nguồn của $g$ khác hẳn — MemoryVLA
tính $g$ bằng `MLP(concat[x, H])`, SeedPolicy lấy $g$ từ chính logit attention.
Ablation ở mục 5.5 cho thấy khác biệt này quan trọng.

Tác giả nhấn mạnh: **không tạo nút thắt inference**. Trong điều khiển robot, quan
sát đến tuần tự; ở bước $t$, $S_{t-1}$ đã tính và cache sẵn.

### 4.4 Cấu hình

$T_{obs} = 3$ bước quan sát, RGB $320 \times 240$ + joint pose 14-DoF. Trạng thái
ẩn $N_s = 60$, $D = 256$. AdamW, batch 128, lr $10^{-4}$ cosine, warmup 500 step.

**Toàn bộ thí nghiệm chạy trên một GPU RTX 4090D.**

## 5. Claim → Evidence

### 5.1 RoboTwin 2.0 — 50 task, 50 demo/task, 600 epoch, 100 rollout/task, 3 lần chạy

| Method                           | Easy            | Hard            | Tham số |
| -------------------------------- | --------------- | --------------- | -------- |
| RDT (VLA)                        | 34.50           | **13.72** | 1.2 B    |
| ACT                              | 29.74           | 1.74            | 80 M     |
| DP-Transformer                   | 33.10           | 1.44            | 20.61 M  |
| DP-CNN                           | 28.04           | 0.64            | 96.80 M  |
| **SeedPolicy-Transformer** | 40.08           | 4.28            | 33.36 M  |
| **SeedPolicy-CNN**         | **42.76** | 1.54            | 147.26 M |

- Easy: **+7.0 tuyệt đối / +21.1% tương đối** (Transformer); **+14.72 / +52.5%**
  (CNN).
- Bằng hoặc hơn DP trên **45/50** task (Transformer) và **44/50** (CNN) —
  architecture-agnostic.
- Vượt RDT 1.2B ở Easy hơn 8 điểm với **ít hơn tới 36× tham số**.
- **Nhưng RDT thắng ở Hard** (13.72 vs 4.28). Tác giả thừa nhận thẳng: VLA hưởng
  lợi từ encoder thị giác/ngôn ngữ pretrain quy mô lớn cho khái quát mở. Trung
  thực và đúng — con số Hard tuyệt đối của mọi phương pháp IL ở đây đều rất thấp.

### 5.2 Lợi ích tăng theo độ dài task

| Nhóm độ dài | Gain Transformer | Gain CNN        |
| --------------- | ---------------- | --------------- |
| Short           | +2.9             | +13.6           |
| Medium          | +6.4             | +12.9           |
| **Long**  | **+16.0**  | **+21.9** |

Đây là bằng chứng trung tâm: policy cửa sổ cố định mất ngữ cảnh lịch sử trong task
kéo dài, còn trạng thái tiến hoá giữ được tiến độ.

### 5.3 Hai chế độ hỏng của Diffusion Policy được đặt tên

1. **Execution stagnation và state aliasing.** Trong task nhiều giai đoạn, demo
   thường chứa khoảng dừng hoặc trạng thái lặp lại về mặt thị giác. Điều này tạo
   **phase ambiguity** cho policy cửa sổ cố định: baseline có thể **đóng băng** sau
   khi quay về trạng thái nhìn giống lúc bắt đầu, hoặc overfit vào các khoảng dừng
   và rơi vào **vòng lặp vận tốc bằng không**.
2. **Sai định vị không gian do thiếu depth**: bốc hụt (air grab), va chạm. Định vị
   3D chính xác từ RGB một góc nhìn cố định là nhập nhằng; cửa sổ quan sát hẹp
   không giải được. Trạng thái tiến hoá tích luỹ manh mối chuyển động dài hạn nên
   suy ra không gian tốt hơn.

Chế độ (1) chính là vấn đề số 2 của [../01_tong_quan.md](../01_tong_quan.md)
("không biết mình đã làm tới đâu"), mô tả ở mức chi tiết hơn mọi paper khác trong
tập.

### 5.4 Đối chứng trực tiếp với các cơ chế memory khác (10 task)

| Task                  | DP + ARMT-style | DP +**MemoryVLA-style** | **SeedPolicy** |
| --------------------- | --------------- | ----------------------------- | -------------------- |
| Move Playingcard Away | 56              | 64                            | **68**         |
| Turn Switch           | 50              | 52                            | **54**         |
| Place Object Stand    | 20              | 26                            | **28**         |
| Dump Bin Bigbin       | 47              | 50                            | **52**         |
| Place Container Plate | 38              | 51                            | **60**         |
| Place Empty Cup       | 15              | 30                            | **32**         |
| Put Object Cabinet    | 15              | 29                            | **41**         |
| Stack Blocks Two      | 33              | 38                            | **47**         |
| Stack Bowls Two       | 56              | 60                            | **73**         |
| Put Bottles Dustbin   | 21              | 26                            | **48**         |

**Đây là bảng có giá trị nhất của paper trong corpus hiện tại**: ba cơ chế memory
khác nhau, cùng backbone DP, cùng giao thức huấn luyện. Không báo cáo còn lưu nào
khác trong corpus làm đối chứng cơ chế kiểu này.

Khác biệt tác giả nêu: ARMT tổ chức truy hồi và cập nhật trong **một chuỗi đệ quy
duy nhất**; MemoryVLA cũng truy hồi nhưng cập nhật chủ yếu là tích hợp biểu diễn
đã hợp nhất vào **memory bank ngoài**; SEGA thực hiện tiến hoá trạng thái **và**
làm giàu quan sát qua **hai luồng song song** ở mỗi bước.

Khoảng cách lớn nhất nằm đúng ở task dài (Stack Bowls Two 60 → 73; Put Bottles
Dustbin 26 → 48).

### 5.5 Ablation thành phần

| Cấu hình                        | Turn Switch (ngắn) | Place Empty Cup (vừa) | Stack Bowls Two (dài) |
| --------------------------------- | ------------------- | ---------------------- | ---------------------- |
| DP                                | 51                  | 24                     | 33                     |
| + Temporal Attention              | 51                  | 26                     | 48                     |
| + State (đệ quy, không cổng)  | 51                  | 28                     | **65**           |
| +**Gating CA** (SeedPolicy) | **54**        | **32**           | **73**           |
| + Gating FFN (MLP thường)       | 53                  | 21                     | 70                     |

Đọc theo cột "task dài": frame stacking 33 → temporal attention 48 → trạng thái đệ
quy 65 → thêm cổng 73. Mỗi bước trong chuỗi chẩn đoán ở mục 2 đều được xác nhận
định lượng.

**Cổng dựa trên cross-attention logit thắng cổng FFN thường**, rõ nhất ở task vừa
(32 vs 21). Kết luận: logit attention nội tại cho tín hiệu độ liên quan đáng tin
hơn cổng học thuần tuý.

### 5.6 Real world

Dexmal DOS W1, camera D435 RGB cố định phía trước, 50 demo/task, 600 epoch, 2 lần
chạy × 50 rollout.

Năm task được thiết kế **riêng cho state ambiguity**: Looping_Place-Retrieval,
Sequential_Picking, Bottle_Handover, Food_Replacement, Cover_and_Reveal.
Bottle_Handover: **16% → 54%**.

### 5.7 Trực quan hoá cổng

Giá trị cổng **đạt đỉnh tại điểm tương tác ngữ nghĩa** (nắm, xếp chồng) và **giảm
trong lúc di chuyển/tiếp cận**; giữ ổn định trạng thái bằng cách lọc frame không
liên quan khi bị che khuất. Đây là bằng chứng diễn giải được cho giả thuyết "thông
tin thưa theo thời gian" ở mục 2.

## 6. Giới hạn và điểm chưa rõ

- **Mục "Limitations and Future Work" không thật sự liệt kê giới hạn nào.** Nó chỉ
  nói Appendix A.4 có bằng chứng sơ bộ rằng SEGA tương thích kiến trúc VLA, và
  rằng World Action Models vẫn khó ở long-horizon. Paper thứ tư liên tiếp trong
  đợt đọc này có vấn đề tương tự.
- **Không có số liệu latency**, dù tuyên bố "không tạo nút thắt inference". Tuyên
  bố này hợp lý về mặt lập luận (trạng thái được cache) nhưng chưa đo. SEGA thêm 2
  MSA + 2 cross-attention mỗi bước.
- **Con số tuyệt đối ở chế độ Hard rất thấp** (4.28% và 1.54%). Mức cải thiện
  "169% tương đối" trong abstract là từ 1.44% lên 4.28% — đúng về số học nhưng dễ
  gây hiểu nhầm về ý nghĩa thực tế. Nên đọc con số tuyệt đối.
- **SeedPolicy-CNN thắng ở Easy nhưng thua Transformer ở Hard** (1.54 vs 4.28), với
  gấp 4.4× tham số. Không được thảo luận.
- Không phải VLA: **không có backbone ngôn ngữ**, không có khái quát mở. Đây là
  imitation learning chuyên biệt. So sánh với RDT/VLA được tác giả tô xám trong
  bảng vì "không so trực tiếp được" — đúng, nhưng abstract vẫn nêu so sánh đó.
- $N_s = 60$ và $T_{obs} = 3$ được chọn cố định, **không quét**. Đây là hai siêu
  tham số trung tâm của tuyên bố "horizon scaling".

## 7. Liên hệ với workspace

- **Đây là một trong các paper nhẹ nhất để tái lập trong corpus hiện tại.** Toàn bộ thí nghiệm
  chạy trên **một RTX 4090D**, model 33M tham số, không cần VLM backbone, không
  cần pretrain, không cần nhãn nào ngoài chính demo. Nếu workspace muốn một thử
  nghiệm long-horizon chạy được end-to-end trên phần cứng đang có, đây là ứng viên
  số một — vượt cả [Seer](../future_prediction/01_seer.md) (65M, cần pretrain).
- **Bảng 5.4 lấp một lỗ hổng đã ghi ở mục 9 của [../01_tong_quan.md](../01_tong_quan.md)**:
  "đối chứng chéo giữa các nhóm gần như không tồn tại". SeedPolicy là paper duy
  nhất chạy ba cơ chế memory trên cùng backbone và cùng giao thức. Kết quả:
  MemoryVLA-style thắng ARMT-style, SEGA thắng cả hai — nhưng chú ý đây là bản
  **tái hiện** MemoryVLA trong DP, không phải MemoryVLA đầy đủ với VLM 7B.
- Với `vla-data-tools`: **không cần nhãn thêm gì cả**. Cùng nhóm nhẹ nhất với
  [MemoryVLA](01_memoryvla.md), [Seer](../future_prediction/01_seer.md) và
  [ACoT-VLA](../future_prediction/03_acot_vla.md).
- Chẩn đoán ở mục 2 (frame stacking → temporal attention → trạng thái đệ quy) là
  bản đồ thiết kế dùng lại được cho bất kỳ policy nào đang xếp chồng frame. Nếu
  workspace gặp hiện tượng "tăng history làm giảm hiệu năng", đây là tài liệu tham
  chiếu trực tiếp.

## 8. Thử nghiệm tiếp theo

1. **Tái lập đường cong horizon scaling** (Fig. 1) trên code công khai: đo hiệu
   năng theo $T_{obs}$ và $N_s$ tăng dần, cho cả DP và SeedPolicy. Chạy được trên
   một 4090. Xác nhận hoặc bác bỏ tuyên bố trung tâm với chi phí thấp nhất trong
   cả tập.
2. **Quét $N_s$** — chưa ai quét. Nếu $N_s$ cũng phụ thuộc task theo cùng kiểu
   thì đó là quy luật chung của memory bank, không phải đặc thù một kiến trúc.
3. **Đo latency và bộ nhớ của SEGA** để hoàn thiện bảng chi phí ở mục 9 của
   [../01_tong_quan.md](../01_tong_quan.md). Ta tự đo được vì có code và phần cứng
   phù hợp.
