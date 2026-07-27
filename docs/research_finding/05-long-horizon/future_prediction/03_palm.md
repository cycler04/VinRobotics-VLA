# PALM — affordance foresight có cấu trúc + progress liên tục

## 1. Nguồn

- Tiêu đề: *PALM: Progress-Aware Policy Learning via Affordance Reasoning for
  Long-Horizon Robotic Manipulation*
- Tác giả: Yuanzhe Liu, Jingyuan Zhu, Yuchen Mo, Gen Li, Xu Cao, Jin Jin, Yifan
  Shen, Zhengyuan Li, Tianjiao Yu, Wenzhen Yuan, Fangqiang Ding, Ismini
  Lourentzou (UPenn / UIUC / NTU / Oxford / MIT)
- arXiv: [2601.07060v2](https://arxiv.org/abs/2601.07060), 4 Apr 2026
- Venue: CVPR 2026
- Trang dự án: https://plan-lab.github.io/palm
- PDF trong repo: [docs/papers/05-long-horizon/09_palm_progress_aware_affordance.pdf](../../../papers/05-long-horizon/09_palm_progress_aware_affordance.pdf)
- Phân loại: **future prediction** (dự báo affordance tại $t+n$). Thành phần
  progress cắt sang [skill_chaining](../skill_chaining/) và
  [memory_modules](../memory_modules/) — xem mục 7.

## 2. Câu hỏi nghiên cứu

Bốn chế độ hỏng đặc trưng của long-horizon mà tác giả nêu tên: **lặp lại hành
động, bỏ sót bước, kết thúc sớm, và tuyên bố thành công ở trạng thái sai**. Tác
giả quy chúng về hai thiếu sót: (1) không có biểu diễn phân biệt được *nên chạm
vật nào, ở đâu, đặt vào đâu, chuyển động thế nào* cho bước kế tiếp; (2) không có
ước lượng liên tục *đã đi được bao xa trong subtask hiện tại*.

Nếu bổ sung đúng hai thứ đó thì có đủ để ổn định thực thi dài không?

## 3. Đóng góp

1. **Affordance foresight có cấu trúc**: bốn loại affordance được dự báo tại
   $t+n$ dưới dạng latent query riêng biệt, thay vì dự báo ảnh tương lai dày đặc.
2. **Progress-aware inverse dynamics**: DiT sinh đồng thời chuỗi action và chuỗi
   giá trị progress vô hướng.
3. Công thức train hai pha trên bốn loại dữ liệu, gồm cả **video người**
   (EPIC-KITCHENS) và benchmark long-horizon (RoboCerebra).

## 4. Method

### 4.1 Bốn affordance query

Backbone GPT-2 style với causal + cross-modal attention. Encoder đông lạnh: CLIP
text, MAE cho ảnh + Perceiver Resampler, MLP cho robot state. Trên đó có hai tập
learnable query.

$$
\hat{F}_{t+n} = f_{aff}(l, o_t, s_t) \in \mathcal{F}
$$

| Query | Trả lời câu hỏi | Nguồn nhãn | Loss |
|---|---|---|---|
| `<Global>` | Vật nào liên quan tới lệnh, nằm đâu | Grounding DINO giải quyết referent + SAM segment thành mask nhị phân | Focal + Dice |
| `<Local>` | Chạm vào chỗ nào trên vật | GLOVER++ contact point chuyển thành Gaussian heatmap | Focal + KL trên bản đồ chuẩn hoá $\ell_1$ |
| `<Spatial>` | Đặt vào đâu | SpatialVLM chuyển lệnh thành ngữ nghĩa không gian + RoboPoint lấy mẫu toạ độ 2D khả thi | Set-matching (min qua $M$ ứng viên dự đoán) |
| `<Dynamic>` | Vùng nào sẽ chuyển động | Lưới $N \times N$ query point tại $t-\delta$, CoTracker theo dõi tiến, giữ quỹ đạo vượt ngưỡng dịch chuyển tích luỹ, rasterize tại $t+n$ | Masked reconstruction (mô hình biến ẩn, có số hạng KL trọng số $\beta$) |

`<Spatial>` cố tình dự đoán **một tập ứng viên** thay vì một toạ độ, để không học
thuộc một vị trí cụ thể — đây là chi tiết thiết kế chống overfit layout.

**Structured attention**: bốn affordance subquery chỉ attend tới context token
chung, **không** attend lẫn nhau, để giữ chúng tách bạch (disentangled).

### 4.2 Progress-aware policy

Thay vì planner riêng, PALM gắn thêm một vô hướng $p_t \in [0,1]$ vào output của
policy:

$$
(\hat{a}_{t:t+n-1},\ \hat{p}_{t:t+n-1}) = f_{inv}(l, o_t, s_t, \hat{F}_{t+n})
$$

huấn luyện bằng objective diffusion chuẩn:

$$
\mathcal{L}_{DiT} = \mathbb{E}_{t_d, \epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta(\tilde{y}_{t:t+n-1,t_d} \mid l, o_t, s_t, \hat{F}_{t+n}, t_d)\big\rVert_2^2\Big]
$$

Lập luận của tác giả: hai quan sát nhìn giống nhau có thể ứng với hành động khác
nhau tuỳ giai đoạn; $p_t$ khử nhập nhằng đó, khuyến khích latent tiến hoá đơn
điệu và làm mượt chuyển tiếp ở biên sub-policy — **mà không cần controller phân
tầng riêng**.

### 4.3 Chi tiết dễ bỏ sót: affordance head là train-only

Theo Fig. 2: khi inference, **bốn head giải mã affordance được gỡ bỏ**;
action-progress query vẫn attend tới affordance foresight latent. Nghĩa là chi
phí giải mã mask/heatmap/point/dynamic chỉ tồn tại lúc train. Latent vẫn được
tính. Đây là phiên bản nhẹ của mẫu hình "giám sát định hình latent, không cần
giải mã lúc chạy" — xem mục 5.1 của [../01_tong_quan.md](../01_tong_quan.md).

### 4.4 Dữ liệu

| Pha | Nguồn | Vai trò |
|---|---|---|
| Pre-training | DROID, BridgeData V2 | Robot in-the-wild |
| Pre-training | EPIC-KITCHENS, RoboCerebra | Video long-horizon, nhãn phân đoạn thời gian để học progress |
| Fine-tuning | **942 trajectory** tự thu, gán nhãn affordance + progress bán tự động | Adapt xuống robot |

## 5. Claim → Evidence

### 5.1 CALVIN ABC→D (Avg Len, top-3 checkpoint, 1000 rollout/task)

| Method | Loại | 5 task liên tiếp | Avg Len |
|---|---|---|---|
| RT-1 | Autoregressive | 1.3% | 0.90 |
| Diffusion Policy | Diffusion | 0.0% | 0.56 |
| OpenVLA | Autoregressive | 43.5% | 3.27 |
| 3D Diffuser Actor | 3D-aware | 41.2% | 3.27 |
| RoboUniview | 3D-aware | 50.7% | 3.65 |
| π0 | Diffusion | 59.9% | 3.92 |
| [Seer](01_seer.md) | Prediction | 64.3% | 3.98 |
| PALM (✗ progress) | Prediction | 67.0% | 4.02 |
| **PALM** | Prediction + Progress | **82.0%** | **4.48** |

+17.7 điểm tuyệt đối so với Seer ở mốc 5 task. Khoảng cách nới rộng theo độ dài
chuỗi — chữ ký của việc giảm lan truyền lỗi.

### 5.2 LIBERO (3 seed × 500 episode)

| Method | Avg | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| SpatialVLA | 69.0 | 88.2 | 89.9 | 78.6 | 55.5 |
| Diffusion Policy | 72.4 | 78.3 | 92.5 | 68.3 | 50.5 |
| OpenVLA | 76.5 | 84.7 | 88.4 | 79.2 | 53.7 |
| CoA-VLA | 79.8 | 85.3 | 93.1 | 85.8 | 55.0 |
| CoT-VLA | 81.1 | 87.5 | 91.6 | 87.6 | 69.0 |
| **PALM** | **94.5** | 95.2 | 96.7 | 94.3 | **91.8** |

+22.8 điểm trên LIBERO-LONG so với CoT-VLA.

### 5.3 Ablation bốn affordance (cộng dồn)

| Cấu hình | CALVIN Avg Len | LIBERO-LONG SR |
|---|---|---|
| VLA thuần | 3.58 | 77.0 |
| + `<Global>` | 3.96 | 84.0 |
| + `<Global,Local>` | 4.16 | **82.5** (tụt) |
| + `<Global,Local,Spatial>` | 4.34 | 86.5 |
| **PALM (đủ 4)** | **4.48** | **91.8** |

`<Local>` cải thiện CALVIN nhưng **làm giảm** LIBERO-LONG; tác giả quy cho bias
hình học do viewpoint ảnh hưởng đặc trưng cạnh chi tiết. `<Spatial>` khôi phục.
Đây là loại chi tiết trung thực đáng ghi nhận — không phải mọi thành phần đều
cộng dồn.

### 5.4 Ablation module theo pha (CALVIN Avg Len)

| Bỏ đi | Pre-training | Fine-tuning |
|---|---|---|
| — (PALM đầy đủ) | 4.48 | 4.48 |
| ✗ Affordance Foresight | 3.90 | **3.58** |
| ✗ Inverse Dynamic Prediction | 4.17 | 3.92 |
| ✗ Progress Prediction | **3.73** | 4.02 |

Đọc chéo bảng: **affordance quan trọng nhất ở fine-tuning**, **progress quan
trọng nhất ở pre-training**. Diễn giải của tác giả: dữ liệu long-horizon quy mô
lớn có giá trị chủ yếu ở chỗ dạy một prior về progress.

### 5.5 Ablation thành phần dữ liệu

| Bỏ đi | CALVIN Avg Len | LIBERO-LONG SR |
|---|---|---|
| — | 4.48 | 91.8 |
| ✗ In-the-Wild (DROID, Bridge V2) | 3.90 | **73.5** |
| ✗ Long-Horizon Video (EPIC-KITCHENS, RoboCerebra) | 3.73 | 84.5 |
| ✗ Human-Annotated (942 traj) | **3.58** | 76.5 |
| ✗ Simulation Data (pretrain) | 3.96 | 81.0 |

Chỉ 942 trajectory có nhãn tay mà bỏ đi thì mất nhiều nhất trên CALVIN (−0.90).
Củng cố cùng luận điểm với [RaC](../recovery_data/01_rac.md): chất lượng và loại
dữ liệu thắng số lượng.

### 5.6 Real world (UFACTORY xArm6 + Gripper G2, 2× RealSense D455)

Task: 6 subtask pick-and-place liên tiếp từ **một** lệnh cấp cao. Fine-tune 200
demo. 20 rollout, tối đa 3 lần thực thi mỗi rollout. Baseline được fine-tune trên
cùng dữ liệu, cùng số iteration, cùng checkpoint cuối.

| Điều kiện | OpenVLA | Octo | PALM |
|---|---|---|---|
| Random Localization | 0.95 | 0.65 | **3.05** |
| Visual Distraction | 1.60 | 0.95 | **3.80** |
| Unseen Lighting | 1.25 | 1.05 | **3.55** |

Cả hai baseline về **0.00** từ subtask 5–6 ở mọi điều kiện; PALM giữ 0.30–0.40.

## 6. Giới hạn và điểm chưa rõ

- **Tác giả tự nêu** (Appendix F): khả năng phục hồi trực tuyến còn hạn chế khi
  execution drift do partial observability, bất định tiếp xúc, che khuất, biến
  dạng hình học. Chuỗi affordance segmentation + state grounding + diễn giải ngữ
  nghĩa bằng VLM tạo **chi phí không nhỏ ở cả annotation, perception và
  inference**.
- **Không có số liệu latency.** Pipeline gán nhãn dùng Grounding DINO + SAM +
  GLOVER++ + SpatialVLM + RoboPoint + CoTracker — sáu mô hình ngoài. Dù chỉ chạy
  lúc train, đây là rào cản tái lập rất lớn và paper không định lượng nó.
- **Missing baseline**: không so với hệ hai tầng ([π0.5](../hierarchical_agent/01_pi0_5.md),
  [Hi Robot](../hierarchical_agent/02_hi_robot.md)) hay với
  [MemoryVLA](../memory_modules/01_memoryvla.md). PALM khẳng định progress vô
  hướng thay được "separate planners or hierarchical controllers" nhưng **không
  đo** đối chứng đó.
- **Threat to validity**: 942 trajectory gán nhãn "bán tự động" — mức can thiệp
  của người không được định lượng, nên không rõ chi phí tái lập.
- **Chưa rõ**: $n$ (khoảng nhìn trước của affordance) không được quét — cùng lỗ
  hổng với [Seer](01_seer.md).

## 7. Liên hệ với workspace

- PALM cắt ngang ba nhóm trong taxonomy: affordance foresight thuộc **future
  prediction**, progress vô hướng phục vụ cả **chuyển tiếp subtask**
  ([skill_chaining](../skill_chaining/01_long_vla.md)) lẫn **biết đã làm tới đâu**
  ([memory_modules](../memory_modules/01_memoryvla.md)). Nó là bằng chứng cho
  thấy vấn đề 2 và 3 của [../01_tong_quan.md](../01_tong_quan.md) có thể được
  đánh cùng một cơ chế rất rẻ: **một số thực trên mỗi bước**.
- Với `vla-data-tools`: yêu cầu nhãn là **một giá trị progress liên tục mỗi
  frame** — nhẹ tương đương nhãn pha nhị phân của Long-VLA và nhẹ hơn nhiều nhãn
  subtask dạng câu. Ngược lại, nhãn affordance thì rất nặng.
- Nếu chỉ lấy một ý từ paper này để thử: lấy **progress head**, bỏ affordance.
  Ablation cho thấy nó một mình đóng góp +0.46 Avg Len (4.02 → 4.48) và chỉ tốn
  một chiều output.

## 8. Thử nghiệm tiếp theo

1. **Chỉ progress, không affordance**: thêm một progress head vào một VLA có sẵn
   và không làm gì khác. Nếu thu được phần lớn khoảng cách thì toàn bộ hạ tầng
   gán nhãn affordance (6 mô hình ngoài) là không cần thiết cho đa số ứng dụng.
2. **Đối chứng progress vô hướng vs closed-loop phân tầng**: chạy PALM-style
   progress và [LoHoVLA](../hierarchical_agent/03_loho_vla.md)-style ngưỡng $K$
   trên cùng benchmark. Hai cơ chế cùng nhắm "biết khi nào chuyển bước" nhưng
   khác nhau về chi phí một bậc.
3. **Quét $n$ và kiểm tra `<Local>`**: tái lập cú tụt trên LIBERO-LONG khi thêm
   `<Local>`. Nếu nó tương quan với số lượng camera hoặc góc nhìn thì quy tắc
   chọn affordance phụ thuộc setup, không phổ quát.
