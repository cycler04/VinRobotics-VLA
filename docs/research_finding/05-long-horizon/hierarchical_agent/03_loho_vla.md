# LoHoVLA — VLA hợp nhất planning và control với closed-loop phân tầng

## 1. Nguồn

- Tiêu đề: *LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon
  Embodied Tasks*
- Tác giả: Yi Yang (Fudan), Jiaxuan Sun (ShanghaiTech), Siqi Kou (SJTU), Yihan
  Wang (Fudan), Zhijie Deng (SJTU, corresponding)
- arXiv: [2506.00411v1](https://arxiv.org/abs/2506.00411), 31 May 2025
- Venue: **preprint, chưa peer-review**
- PDF trong repo: [docs/papers/05-long-horizon/08_loho_vla_unified_long_horizon_embodied_tasks.pdf](../../../papers/05-long-horizon/08_loho_vla_unified_long_horizon_embodied_tasks.pdf)
- Phân loại: **hierarchical agent** (hợp nhất một model, nhưng vẫn phân tầng ở
  mức suy luận và ở mức vòng điều khiển).

## 2. Câu hỏi nghiên cứu

Kiến trúc hợp nhất (một VLM sinh cả subtask lẫn action token) có vượt được kiến
trúc phân tầng rời (planner + controller riêng) và VLA phẳng trên task
long-horizon không? Và cơ chế closed-loop nào xử lý được lỗi đến từ **cả hai**
tầng?

## 3. Đóng góp

1. **Một language head, hai loại output**: PaliGemma sinh cả câu subtask lẫn
   action token rời rạc, chia sẻ biểu diễn giữa planning và control.
2. **Hierarchical closed-loop control**: re-predict action thường xuyên, chỉ
   re-plan subtask khi số lần thất bại vượt ngưỡng $K$.
3. **LoHoSet**: dataset tổng hợp trên Ravens, 20 task long-horizon × 1000 demo,
   có nhãn subtask theo bước.
4. Chiến lược **two-stage training** tách việc học planning khỏi việc học action.

## 4. Method

### 4.1 Phân rã

$$
\pi_\theta(a_t, \hat{g}_t \mid o_t, g)
= \pi_\theta(a_t \mid o_t, g, \hat{g}_t) \cdot \pi_\theta(\hat{g}_t \mid o_t, g)
$$

So sánh ba paradigm mà paper đối chiếu:

| Paradigm | Công thức | Vấn đề |
|---|---|---|
| Vanilla VLA | $\pi_\theta(o_t, g) \to a_t$ | Subtask ngầm, không diễn giải được, overfit pattern |
| Hierarchical rời | $\pi^{planner}(o_t,g) \to \hat{g}_t$; $\pi^{controller}(o_t,\hat{g}_t) \to a_t$ | Phối hợp kém, khái quát hạn chế |
| LoHoVLA | công thức trên, **một** $\theta$ | — |

Lưu ý: khác π0.5, ở đây tầng thấp vẫn thấy $g$ (goal tổng) bên cạnh $\hat{g}_t$.

### 4.2 Kiến trúc

- Backbone: PaliGemma-3b-mix-224 (SigLIP image encoder + Gemma-2B decoder +
  linear projection). Image encoder và projection **đóng băng**, chỉ tune LLM.
- Action: rời rạc hoá giá trị đã chuẩn hoá thành **1024 bin đều**; action là
  $(T_{pick}, T_{place})$.
- Loss: $L = L_{text} + L_{action}$, cả hai đều cross-entropy từ cùng language
  head.

### 4.3 Hierarchical closed-loop control (Algorithm 1)

```
k ← 0
while not done:
    if t = 0 or r > 0 or k > K:      # thành công subtask, hoặc quá nhiều lỗi
        ĝ_t ~ π(ĝ_t | o_t, g);  k ← 0
    a_t ~ π(a_t | o_t, g, ĝ_t)
    execute a_t;  quan sát o_{t+1}, r
    if r = 0: k ← k + 1
```

Ba loại lỗi mà cơ chế này phân biệt: (1) plan sai subtask, (2) plan đúng nhưng
action sai, (3) plan và action đúng nhưng nhiễu ngoại cảnh. Chỉ loại (1) cần
re-plan. Thực nghiệm dùng $K = 2$.

### 4.4 LoHoSet

- Simulator Ravens, UR5e + suction gripper. Observation = RGB + depth
  top-down orthographic.
- Nhiễu quan sát + xác suất **rơi vật mỗi giây** — tạo ra lỗi loại (3).
- Object: block (2 kích cỡ), bowl, zone; 11 màu.
- 20 task long-horizon × 1000 demo; nhãn subtask sinh bằng luật, tận dụng full
  state của simulator. Thứ tự subtask ngẫu nhiên khi không có ràng buộc phụ
  thuộc.
- Kế thừa 10 task long-horizon + 3 task pick-and-place primitive từ LoHoRavens,
  cộng thêm 10 task mới **để chống overfit**.

### 4.5 Two-stage training

| Stage | Dữ liệu | Loss | Cấu hình |
|---|---|---|---|
| 1 | 14 task long-horizon × 1000 demo | chỉ $L_{text}$ | 3 epoch, LR 5e-5, LoRA rank 16 mọi linear layer, 8×RTX 4090, batch 2/device |
| 2 | + 10k demo mỗi pick-and-place primitive | $L_{text} + L_{action}$ | 1 epoch, LR 1e-5 |

## 5. Claim → Evidence

Metric: **average score** (tỉ lệ bước pick-and-place đúng, 0–100) và **success
rate**. Hai kiểu khớp: *pose match* (chính xác vị trí + hướng) và *zone match*
(diện tích chồng lấn vượt ngưỡng).

| Claim | Bằng chứng (score / SR, %) |
|---|---|
| Vượt vanilla VLA rất xa trên task cần lý luận | `put-block-into-matching-bowl`: 97.8/91.5 vs 14.9/**0.0**; `put-even-blocks-in-same-color-zone`: 85.1/81.0 vs 22.1/3.5 |
| Vượt LoHoRavens (planner+actor+reporter rời) | Task E: 85.1/81.0 vs 9.6 (explicit) và 8.2 (implicit) |
| Khái quát sang task chưa thấy | F 86.1/41.0, I 77.2/52.0, K 73.8/54.5 — đều cao nhất |
| **Không** thắng ở task nguyên tử | `pick-and-place-primitive`: LoHoVLA 77.5/77.5 < vanilla VLA 79.0/79.0 |
| Closed-loop phân tầng ≈ luôn re-plan nhưng rẻ hơn | Task B: (a) 89.5/74.0 với 5.7 lần plan; (b) 96.4/88.5 với 6.4; (c) 97.8/91.5 với 6.2 |
| Chỉ re-predict action là tệ nhất | Strategy (a) thấp nhất mọi task — kẹt vòng lặp khi plan sai |
| Mở rộng training set là bắt buộc để khái quát | Không mở rộng: sub-task planning success trên task F và K rơi về **0%**; có mở rộng đạt ~70–100% |
| Two-stage tốt hơn one-stage | Sub-task planning success ~85.4% vs ~80.9%, task completion ~44.2% vs ~40.8% (đọc từ Fig 3b) |

Phân tích định tính đáng giá: vanilla VLA thất bại vì **overfit pattern thị
giác** — trong `put-block-into-mismatching-bowl` nó vẫn bỏ block vào bát cùng
màu, bỏ qua goal ngôn ngữ. Đây là bằng chứng cho thấy giám sát subtask hoạt động
như một điều chuẩn (regularizer) buộc model đọc goal.

## 6. Giới hạn và điểm chưa rõ

- **Tác giả tự nêu**: action rời rạc hạn chế độ chính xác; giả định **mỗi subtask
  hoàn thành trong một timestep** — giả định này rất mạnh và không đúng ngoài
  Ravens.
- **Chỉ simulator**. Không có thí nghiệm robot thật. Ravens dùng ảnh top-down
  orthographic tái dựng, dễ hơn RGB từ camera thật rất nhiều.
- **Preprint, chưa peer-review** — nên xếp mức tin cậy thấp hơn 7 paper còn lại
  trong tập này.
- **Missing baseline**: không so với π0.5 hay Hi Robot, tức là không so với hệ
  "một model" và "hai model" hiện đại nhất. Baseline hierarchical (LoHoRavens)
  dùng CLIPort làm actor — yếu hơn nhiều so với một VLA flow-matching.
- **Threat to validity**: LoHoVLA và vanilla VLA được train số epoch khác nhau
  (2 stage 3+1 epoch vs 5 epoch) và LR khác nhau. Không phải so sánh compute-
  matched.
- **Chưa rõ**: metric `sub-task planning success rate` dùng một LLM để phán xét
  tương đương ngữ nghĩa với danh sách ground-truth do người liệt kê. Độ tin cậy
  của bộ phán xét này không được đo.

## 7. Liên hệ với workspace

- Cùng họ "một model, hai tầng" với [01_pi0_5.md](01_pi0_5.md) nhưng đi xa hơn ở
  chỗ **action cũng là token của language head** thay vì có action expert riêng.
  Đây là đánh đổi đáng cân nhắc khi đọc [03-vla-core](../../03-vla-core/): head
  rời rạc dễ train, nhưng chính paper thừa nhận nó giới hạn độ chính xác.
- Cơ chế closed-loop có ngưỡng $K$ là thứ dễ tái lập nhất trong cả tập paper này
  và không phụ thuộc kiến trúc — có thể bọc quanh bất kỳ policy nào có tín hiệu
  reward/thành-bại theo subtask.
- LoHoSet là ví dụ về dataset có **nhãn subtask theo timestep**; nếu cần thử
  nghiệm hierarchical VLA trong workspace thì đây là nguồn rẻ nhất (simulator,
  sinh nhãn bằng luật).

## 8. Thử nghiệm tiếp theo

1. **Tách riêng đóng góp của giám sát subtask khỏi đóng góp của kiến trúc hợp
   nhất**: train vanilla VLA với auxiliary loss dự đoán subtask nhưng **không**
   đưa subtask vào context lúc inference (tương đương "implicit HL" của π0.5).
   Nếu nó gần bằng LoHoVLA thì kiến trúc hợp nhất không phải nguyên nhân.
2. **Bỏ giả định một-subtask-một-timestep**: chạy trên LIBERO-Long hoặc CALVIN
   nơi subtask kéo dài nhiều bước, đo xem cơ chế đếm lỗi $k > K$ còn hoạt động
   không.
3. **Quét ngưỡng $K$**: paper chỉ báo $K=2$. Đo đường cong success rate và số
   lần plan theo $K \in \{1,2,4,8\}$ để biết cơ chế nhạy tới mức nào.
