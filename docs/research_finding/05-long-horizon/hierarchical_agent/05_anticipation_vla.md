# Anticipation-VLA — sinh subgoal đệ quy, độ mịn thích ứng theo value

## 1. Nguồn

- Tiêu đề: *Anticipation-VLA: Solving Long-Horizon Embodied Tasks via
  Anticipation-based Subgoal Generation*
- Tác giả: Zhilong Zhang, Wenyu Luo, Haonan Wang, Yifei Sheng (đồng tác giả
  chính), Yidi Wang, Hanyuan Guo, Haoxiang Ren, Xinghao Du, Yuhan Che, Tongtong
  Cao, Lei Yuan, Yang Yu (Nanjing University — National Key Lab for Novel
  Software Technology; Huawei 2012 Labs)
- arXiv: [2605.01772v1](https://arxiv.org/abs/2605.01772), 3 May 2026
- Venue: **preprint**
- PDF trong repo: [docs/papers/05-long-horizon/11_anticipation_vla_subgoal_generation.pdf](../../../papers/05-long-horizon/11_anticipation_vla_subgoal_generation.pdf)
- Phân loại: **hierarchical agent**. Thành phần sinh ảnh subgoal cắt sang
  [future_prediction](../future_prediction/).

## 2. Câu hỏi nghiên cứu

Mọi phương pháp phân rã hiện có dùng **độ mịn subgoal cố định**. Hệ quả: subgoal
quá mịn thì thêm phức tạp không cần thiết; quá thưa thì không đủ dẫn hướng
policy. Có thể để **độ mịn tự thích ứng theo tiến độ thực tế** không?

## 3. Đóng góp

1. **Anticipation Model** $\mathcal{G}: \mathcal{S} \times \mathcal{G} \to \mathcal{G}$
   — sinh subgoal **đệ quy**: subgoal sinh ra có thể quay lại làm goal đầu vào để
   phân rã tiếp.
2. Nền tảng lý thuyết: điều kiện **optimal decomposition** dựa trên value.
3. **Goal stack động** với ba điều kiện kích hoạt (đạt / trì trệ / tiến bộ), cho
   phép backtrack.
4. Một **UMM duy nhất** đóng bốn vai: policy ngôn ngữ, forward dynamics, inverse
   dynamics, và value model.

## 4. Method

### 4.1 Không gian goal đa phương thức

$$
\mathcal{G} = (\mathcal{L} \cup \{\emptyset_\mathcal{L}\}) \times (\mathcal{S} \cup \{\emptyset_\mathcal{S}\}) \setminus \{(\emptyset_\mathcal{L}, \emptyset_\mathcal{S})\}
$$

Một goal là cặp (ảnh, lệnh) với **ít nhất một** thành phần khác rỗng. Cho phép
goal thuần ngôn ngữ ("Make a coffee") hoặc thuần thị giác (ảnh tách cà phê).

### 4.2 Điều kiện phân rã tối ưu

$$
V^*(s_0, g) = V^*(s_0, g') + V^*(s_{g'}, g)
$$

Reward tối đa từ $s_0$ tới goal cuối phải phân rã đúng bằng tổng hai chặng qua
waypoint $g'$. Điều này đòi hỏi **cấu trúc reward kiểu shortest-path**. Đây là
mục tiêu huấn luyện của bộ sinh subgoal — điểm khác biệt so với các paper còn
lại trong tập, vốn chỉ bắt chước nhãn subtask do người/luật gán.

### 4.3 Value model: hồi quy → phân loại 3 lớp

Trong thực tế không có dense reward, chỉ có tín hiệu sparse mức trajectory, nên
TD learning bất ổn. Tác giả **đổi bài toán**: value model $V_\theta$ nhận
$(s, s_{prev}, g)$ và phân loại vào $\{$*goal achieved*, *progress stagnated*,
*progress improved*$\}$.

Đây là đơn giản hoá thực dụng đáng chú ý: hệ chỉ cần biết "đạt chưa / có tiến
không", không cần giá trị tuyệt đối.

### 4.4 Goal stack (mỗi $K$ bước kiểm tra một lần)

| Điều kiện | Hành động |
|---|---|
| $\lvert V^*(s,g) - V^*(g,g)\rvert < \delta$ | Subgoal xong → **pop** khỏi stack |
| Tiến bộ không đáng kể và stack chưa đầy | $\mathcal{G}$ sinh subgoal mịn hơn → **push** |
| Tiến bộ không đáng kể và stack **đã đầy** | **Backtrack** về trạng thái ban đầu (policy đang kẹt cục bộ) |
| Tiến bộ đáng kể | **Giữ** subgoal hiện tại |

### 4.5 Chống ảo giác subgoal: bottleneck ngữ nghĩa + tự kiểm

Sinh trực tiếp $g'$ từ $(s, g)$ hay bị ảo giác. Tác giả tách hai bước:

$$
\mathcal{G}_\theta = (P_\theta \circ l_\theta,\ l_\theta)
$$

1. $l_\theta: \mathcal{S} \times \mathcal{G} \to \mathcal{L}$ sinh **lệnh subgoal
   dạng text** trước.
2. $P_\theta: \mathcal{S} \times \mathcal{L} \to \mathcal{S}$ mới sinh **ảnh
   subgoal** có điều kiện trên lệnh đó.

Text đóng vai trò bottleneck ngữ nghĩa ép quá trình sinh.

**Self-discriminative regularization** (theo Uni-Plan): sau khi sinh $s_{g'}$,
áp inverse dynamics $P_\theta^{-1}: \mathcal{S} \times \mathcal{S} \to \mathcal{L}$
để suy ngược lệnh $l'_{inv}$ dẫn từ $s$ tới $s_{g'}$. Nếu $l'_{inv}$ tương đương
ngữ nghĩa với $l_{g'}$ thì giữ; không thì bỏ và sinh lại.

### 4.6 Loss hợp nhất

$$
\mathcal{L}(\theta) = \lambda_1 \mathcal{L}_{policy} + \lambda_2 \mathcal{L}_{dyna} + \lambda_3 \mathcal{L}_{inverse} + \lambda_4 \mathcal{L}_{value}
$$

CE cho policy / inverse / value; MSE flow-matching cho forward dynamics. Tất cả
trong **một** kiến trúc UMM (Bagel).

### 4.7 Low-level policy

π0.5 goal-conditioned. Chuỗi input: quan sát hiện tại từ mọi camera $s_o^t$, rồi
**ảnh subgoal** $s_g^t$, rồi cấu hình robot $q$, rồi **lệnh subgoal** $\ell_g^t$.

$$
\pi_\theta(a_{t:t+h} \mid s^t_o, g) = \pi_\theta(a_{t:t+h} \mid s^t_o, g_t) \cdot \mathcal{G}_\theta(g_t \mid g)
$$

Chống nhiễu từ subgoal sinh ra: **mask ngẫu nhiên token của ảnh goal** trong lúc
train.

Siêu tham số VLA: 4× H100 80GB, peak LR 2.5e-5 (warmup 1k, cosine decay 29k
xuống 2.5e-6), batch 64, 5k step LIBERO / 10k VLABench / 10k real, ảnh 224×224,
**không dùng observation history**, action chunk 10 (sim) và 20 (real).

## 5. Claim → Evidence

### 5.1 LIBERO, chế độ one-trajectory SFT (chỉ 40 trajectory tổng cộng)

| Model | Spatial | Object | Goal | **Long** | Avg |
|---|---|---|---|---|---|
| UniVLA | 26.0 | 40.0 | 18.0 | 1.8 | 21.5 |
| DreamVLA | 38.0 | 34.0 | 16.6 | 20.6 | 27.3 |
| π0 | 70.2 | 80.0 | 70.6 | 37.6 | 64.6 |
| π0.5 + VLM | 82.0 | 88.0 | 80.8 | 53.2 | 76.0 |
| π0.5 | 78.2 | 88.6 | 85.8 | 54.6 | 76.8 |
| **Anticipation-VLA** | 81.8 | **91.6** | **86.6** | **63.2** | **80.8** |

**Kết quả quan trọng nhất của paper nằm ở hai dòng giữa**: gắn thêm một VLM
lập kế hoạch **tĩnh** vào π0.5 (Qwen2.5-7B fine-tune sinh subgoal text) cho
76.0 — **thấp hơn** π0.5 trần (76.8). Chỉ cơ chế anticipation **thích ứng** mới
cải thiện. Tác giả kết luận thẳng: "effective long-horizon reasoning relies not
on static external modules but on an adaptive multimodal anticipation mechanism."

### 5.2 VLABench — task *Hammer Nail & Hang Picture* (100 trajectory)

| Model | Process Reward | Success Rate |
|---|---|---|
| DreamVLA | 7.3 | 0.0 |
| UniVLA | 28.1 | 1.0 |
| π0 | 39.6 | 1.0 |
| π0.5 | 42.7 | 2.1 |
| π0.5 + VLM | 47.9 | 2.1 |
| **Anticipation-VLA** | **56.3** | **4.2** |

Success rate tuyệt đối vẫn rất thấp (4.2%) — task này gần như không model nào
giải được. Process reward là chỉ số có ý nghĩa hơn ở đây.

### 5.3 Real world (Arx-X5 mobile manipulator)

Hai task với **hai kiểu goal khác nhau**: *Rearrange Objects* (goal là **ảnh**,
100 demo) và *Spell Words* (goal là **ngôn ngữ**, 200 demo). 40 rollout mỗi task
(20 seen, 20 unseen).

- Cải thiện +60% ở cấu hình seen, **+107% ở unseen**.
- Là model **duy nhất đạt success rate khác 0** trên *Spell Words* unseen.
- Stage-wise: baseline sụp nhanh từ stage 3 ở *Rearrange Objects* unseen;
  Anticipation-VLA còn 0.41 tại stage 3.

### 5.4 Ablation (real world)

Ba biến thể: bỏ ảnh subgoal, bỏ text subgoal, bỏ đệ quy (thay bằng sinh một mức
cố định). Bản đầy đủ thắng ở **cả bốn** cấu hình (seen/unseen × 2 task), cả
success rate lẫn score.

Phân tích theo giai đoạn: biến thể **w/o recursive** tụt rõ ở **các stage muộn** —
đúng dự đoán rằng sinh cố định độ mịn thiếu khả năng thích ứng cho phân rã sâu.
Bỏ ảnh hoặc bỏ text đều gây sụp nhanh, cho thấy hai kênh bổ sung nhau: ảnh cho
định vị vật lý, text cho neo ngữ nghĩa.

### 5.5 Chất lượng anticipation (400 mẫu held-out mỗi task)

| Benchmark | Subtask pred. acc. | PSNR | SSIM | FID |
|---|---|---|---|---|
| Libero | 84.4 | 20.4 | 0.85 | 31.0 |
| VLABench | 88.8 | **15.5** | **0.76** | **55.1** |
| Rearrange Objects | 88.1 | 28.0 | 0.93 | 45.1 |
| Spell Words | **98.9** | 26.4 | 0.92 | 34.7 |

Chất lượng ảnh subgoal **kém nhất ở simulation** (VLABench), tốt nhất ở real
world. Tác giả giải thích: dataset sim nhỏ hơn, tranh ảnh khó và chưa từng thấy,
và UMM **không được pretrain trên dữ liệu simulation**. Đây là nghịch lý ngược so
với thường lệ và đáng ghi nhận.

### 5.6 Generalization

Hai chế độ: **Object** (Spell Words chỉ train trên chữ cái, yêu cầu ghép chuỗi
chữ-số như "H2O") và **Background** (đổi texture bề mặt và điều kiện chiếu sáng).

Trên *Rearrange Objects*: score 0.58 (object) và 0.64 (background), **xấp xỉ mốc
unseen chuẩn 0.59** — suy giảm không đáng kể. Trên *Spell Words*: model duy nhất
có success khác 0; các baseline hỏng hoàn toàn.

## 6. Giới hạn và điểm chưa rõ

- **Tác giả tự nêu**: vẫn cần vài demo có nhãn subgoal để fine-tune; sinh ảnh
  subgoal **tốn kém về tính toán, gây tạm dừng inference không liên tục**
  (occasional inference pauses).
- **Preprint, chưa peer-review.**
- Không có bảng latency. "Occasional inference pauses" là mô tả định tính duy
  nhất về chi phí — với một pipeline phải chạy UMM sinh ảnh, đây là thiếu sót
  đáng kể. Đối chiếu:
  [ReflectVLM](../future_prediction/02_reflective_planning.md) đo được 11.10 s/bước
  cho một vòng sinh ảnh bằng diffusion.
- **Cơ sở lý thuyết chưa được kiểm chứng**: điều kiện optimal decomposition đòi
  hỏi cấu trúc reward shortest-path, và value model thật lại **không** ước lượng
  $V^*$ mà chỉ phân loại 3 lớp. Khoảng cách giữa lý thuyết và hiện thực không
  được đo — không có thí nghiệm nào cho thấy subgoal sinh ra thực sự thoả
  $V^*(s_0,g) = V^*(s_0,g') + V^*(s_{g'},g)$.
- Không quét $K$ (chu kỳ kiểm tra) và không quét độ sâu tối đa của goal stack.
- Cơ chế **backtrack về trạng thái ban đầu** khi stack đầy: không rõ áp dụng thế
  nào cho task không đảo ngược được (đã đổ nước ra thì không backtrack được).
- **Missing baseline**: không so với [MemoryVLA](../memory_modules/01_memoryvla.md)
  hay [LoHo-Manip](04_loho_manip.md) dù cùng dùng π0.5 làm executor.

## 7. Liên hệ với workspace

- Cùng [LoHo-Manip](04_loho_manip.md), đây là paper thứ hai lấp lỗ hổng
  "một model vs hai model" của [../01_tong_quan.md](../01_tong_quan.md): executor
  là chính π0.5, cùng dữ liệu, và hệ hai tầng thắng.
- Nhưng nó **thu hẹp** kết luận đó theo một cách quan trọng: π0.5 + VLM tĩnh
  **không** thắng (76.0 vs 76.8). Vậy không phải "cứ tách hai tầng là tốt", mà
  là "tầng cao phải thích ứng theo tiến độ thực tế". Cả LoHo-Manip
  (receding-horizon remaining-plan) lẫn Anticipation-VLA (value-triggered
  recursive) đều thoả điều kiện này; một planner chạy một lần lúc đầu thì không.
- Với `vla-data-tools`: cần **dataset subgoal phân tầng** $D^h_{anti}$ với $H$
  mức độ mịn, cộng nhãn tiến độ 3 lớp $\{progress, achieve, no\ progress\}$ gán
  theo độ gần thời gian và hình học tới goal. Đây là yêu cầu nhãn nặng nhất trong
  cả 12 paper.
- Ý rẻ nhất tách ra được: **value model 3 lớp**. Nó không cần UMM, không cần sinh
  ảnh, và giải quyết đúng câu hỏi "khi nào nên replan" mà
  [LoHoVLA](03_loho_vla.md) đang xử lý bằng đếm lỗi $k > K$ và
  [PALM](../future_prediction/03_palm.md) xử lý bằng progress vô hướng. Ba cách
  cho cùng một chức năng, khác nhau về chi phí một bậc.

## 8. Thử nghiệm tiếp theo

1. **Chỉ giữ value model 3 lớp**: gắn bộ phân loại tiến độ vào một VLA phẳng để
   kích hoạt replan, bỏ hết phần sinh subgoal đa phương thức. Nếu thu được phần
   lớn khoảng cách thì UMM sinh ảnh là không cần thiết.
2. **Đo latency và tần suất tạm dừng**: đo phân phối thời gian mỗi bước, không
   chỉ trung bình. "Occasional pauses" nghĩa là phân phối có đuôi dài — đó mới là
   thứ quyết định dùng được trên robot thật hay không.
3. **Kiểm chứng điều kiện phân rã tối ưu**: trong simulator có full state, tính
   $V^*$ thật và đo sai số của
   $V^*(s_0,g) - [V^*(s_0,g') + V^*(s_{g'},g)]$ trên các subgoal do model sinh.
   Nếu sai số lớn thì phần lý thuyết chỉ là động lực, không phải cơ chế đang hoạt
   động.
