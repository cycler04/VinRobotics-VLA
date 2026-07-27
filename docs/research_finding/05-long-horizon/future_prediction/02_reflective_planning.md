# ReflectVLM — tưởng tượng tương lai bằng diffusion để phản tỉnh kế hoạch

## 1. Nguồn

- Tiêu đề: *Reflective Planning: Vision-Language Models for Multi-Stage
  Long-Horizon Robotic Manipulation*
- Tác giả: Yunhai Feng (Cornell), Jiaming Han, Xiangyu Yue (CUHK), Zhuoran Yang
  (Yale), Sergey Levine, Jianlan Luo (UC Berkeley)
- arXiv: [2502.16707v1](https://arxiv.org/abs/2502.16707), 23 Feb 2025
- Venue: CoRL 2025
- Trang dự án: https://reflect-vlm.github.io
- PDF trong repo: [docs/papers/05-long-horizon/06_reflective_planning_multi_stage_long_horizon.pdf](../../../papers/05-long-horizon/06_reflective_planning_multi_stage_long_horizon.pdf)
- Phân loại: **future prediction** (dùng dynamics model sinh ảnh tương lai để
  đánh giá và sửa plan). Cũng có thể xếp là agent — nhưng cơ chế mới nằm ở phần
  tưởng tượng.

## 2. Câu hỏi nghiên cứu

VLM pretrain trên Internet có kiến thức rộng nhưng **kém lý luận vật lý** và
**kém lập kế hoạch dài**. Có thể bù bằng test-time computation thay vì retrain
quy mô lớn không? Bài toán thử: lắp ráp các mảnh **cài khoá lẫn nhau
(interlocking)** vào bảng — chỉ lắp được theo một thứ tự nhất định.

## 3. Đóng góp

1. Khung **test-time computation** gồm hai phần: look-ahead bằng diffusion
   dynamics model, và reflection để VLM tự phê bình rồi sửa action đã đề xuất.
2. **Interactive post-training** kiểu DAgger sinh đồng thời dữ liệu cho cả hai
   chế độ prompt (propose và reflect) từ cùng một rollout.
3. Bằng chứng thực nghiệm rằng cách này **vượt MCTS** cả về chất lượng lẫn chi
   phí, và MCTS thậm chí **làm hỏng** policy nền.

## 4. Method

### 4.1 Bài toán

POMDP $(S, A, T, O, Z)$. Action space là primitive cấp cao:
$\{\texttt{pick up}, \texttt{insert}, \texttt{reorient}, \texttt{put down}\} \times \{\text{objects}\}$,
mỗi primitive có tỉ lệ hỏng $\epsilon$. Policy chỉ thấy ảnh: $\pi(a_t \mid I_t, I_g)$.
Có expert oracle $\pi_E$ truy cập full state (success rate 97%) dùng làm nhãn.

### 4.2 Sinh dữ liệu cho reflection (điểm khéo nhất của paper)

Không cần thu thập riêng. Sau khi một trajectory kết thúc, **relabel** nó: ảnh
$I_{t+H}$ (quan sát thực tế sau khi chạy $a_{t:t+H-1}$) được thêm vào context tại
bước $t$, và model vẫn bị giám sát để xuất **cùng** action expert $a^*_t$.

Với mỗi timestep sinh ra hai ví dụ:

- **(Q1, A1)** propose: $(I_g, I_t) \to a^*_t$
- **(Q2, A2)** reflect: $(I_g, I_t, I_{t+H}, a_{t:t+H-1}) \to a^*_t$

Loss:

$$
\min_{\pi_{VLM}} \mathbb{E}_D\Big[
L_{CE}\big(\pi^{propose}_{VLM}(a_t \mid I_g, I_t),\, a^*_t\big)
+ L_{CE}\big(\pi^{reflect}_{VLM}(a_t \mid I_g, I_t, I_{t+H}, a_{t:t+H-1}),\, a^*_t\big)
\Big]
$$

Rollout dùng hỗn hợp: action của learner $a^\dagger_t$ với xác suất $p$, action
expert $a^*_t$ với xác suất $1-p$ (để hội tụ).

### 4.3 Diffusion Dynamics Model (DDM)

Dự đoán forward dynamics như bài toán image-to-image translation, khởi tạo từ
**InstructPix2Pix**:

- Latent encoder và text encoder **đóng băng**; Diffusion UNet và latent decoder
  được finetune.
- Hai pha train song song: UNet học $z_t \to z_{t+1}$ có điều kiện $z_{a_t}$;
  latent decoder được adapt riêng vì task đòi hỏi tái dựng chính xác các mảnh nhỏ.
- Dữ liệu thu bằng phiên bản có nhiễu của oracle policy để phủ rộng trạng thái;
  tác giả **có đưa một ít data point từ test vào** để tăng độ trung thực — đây là
  điểm cần lưu ý khi diễn giải kết quả.

### 4.4 Reflective Planning lúc inference (Algorithm 2)

```
Ĩ_t ← I_t
for k = 0 .. H-1:
    ã_{t+k}   ← π^propose(I_g, Ĩ_{t+k})     # đề xuất trong tưởng tượng
    Ĩ_{t+k+1} ← T̃(Ĩ_{t+k}, ã_{t+k})         # diffusion sinh ảnh kế tiếp
a_t ← π^reflect(I_g, I_t, Ĩ_{t+H}, ã_{t:t+H-1})   # phản tỉnh rồi quyết định
```

Chỉ **một vòng** reflection, và chỉ dùng **ảnh cuối** $\tilde{I}_{t+H}$ (không
dùng các bước trung gian) — do giới hạn context của VLM.

### 4.5 Cấu hình

- Policy: LLaVA-1.5-13B, pretrain bằng 5000 demo expert (1000 task × 5 cấu hình
  ban đầu).
- Post-training: mỗi iteration lấy ngẫu nhiên 200/1000 task, thu 1k trajectory,
  finetune trên dữ liệu tích luỹ. 3 iteration.
- Eval: 100 task **chưa từng thấy**, 5 seed (trừ VLM thương mại chạy 1 lần).

## 5. Claim → Evidence

### 5.1 Kết quả chính (success rate trên 100 task unseen)

| Method | SR (%) |
|---|---|
| LLaVA-OneVision (zero-shot) | 0.0 |
| Gemini-2.0-flash | 6.0 |
| GPT-4o | 6.0 |
| Gemini-2.0-flash-thinking | 8.0 |
| GPT-o1 | 15.0 |
| MCTS (dùng VLM pretrain làm base policy, oracle làm value) | 24.0 |
| BC (VLM policy pretrain) | 47.8 |
| Ours w/o reflect (train và test đều không) | 77.8 |
| Ours w/o reflect@test (có reflect khi train) | 82.2 |
| **Ours w/ diffusion** | **82.4** |
| Ours w/ sim (oracle dynamics, upper bound) | 85.4 |

### 5.2 Ba phát hiện đáng nhớ

1. **VLM thương mại tốt nhất chỉ đạt 15%.** GPT-o1 chỉ giải được các case đơn
   giản không đòi hỏi lý luận về cơ cấu cài khoá. Đây là bằng chứng sạch cho việc
   internet-scale knowledge **không** chuyển thành physical reasoning.
2. **MCTS làm giảm hiệu năng** (24.0 < 47.8 của chính base policy). Nguyên nhân
   tác giả phân tích: rất nhạy với chất lượng value function; lý luận vật lý tinh
   tế khó nhét vào value; và vì **từ bất kỳ trạng thái nào cũng có thể thành công
   bằng cách dọn sạch bảng làm lại**, nên chênh lệch value giữa các state gần như
   bằng 0. Đây là một quan sát có giá trị vượt ra ngoài paper.
3. **Prompt reflection khi train cải thiện cả khi không reflect lúc test**
   (77.8 → 82.2). Tức là học phản tỉnh làm model lý luận ngầm tốt hơn — hiệu ứng
   giống chain-of-thought distillation.

### 5.3 Động lực post-training (SR % theo iteration)

| Variant | Iter 1 | Iter 2 | Iter 3 |
|---|---|---|---|
| w/o reflect | 58.2 | 74.4 | 77.8 |
| w/o reflect@test | 64.4 | 76.0 | 82.2 |
| reflect w/ diffusion | 66.2 | 75.8 | 82.4 |
| reflect w/ sim | 66.8 | 75.4 | 85.4 |

### 5.4 Chi phí inference (mỗi bước, một A100)

| Method | Thời gian |
|---|---|
| Ours w/o reflect@test | 0.45 s |
| Ours w/ sim | 6.05 s |
| Ours w/ diffusion | 11.10 s |
| MCTS | 391.42 s |

Reflection rẻ hơn MCTS ~35 lần và cho SR gấp hơn 3 lần.

## 6. Giới hạn và điểm chưa rõ

- **Khoảng cách tuyệt đối giữa có và không reflection lúc test là nhỏ**: 82.2 →
  82.4 với diffusion, tức +0.2 điểm với chi phí **gấp 25 lần** (0.45 s → 11.10 s).
  Tác giả lập luận rằng các task thêm được là "qualitatively significant" (cần
  replan nhiều lần, tháo vật đã đặt ra để thử cách khác) nhưng **không định lượng
  lập luận này**. Đây là điểm yếu lớn nhất của paper: phần lớn lợi ích đến từ
  *dữ liệu reflection trong post-training*, không từ *reflection lúc chạy*.
  So sánh: π0.5 tìm thấy đúng kiểu hiệu ứng này với "implicit HL".
- **Dữ liệu train DDM có chứa một ít điểm từ test** để tăng độ trung thực. Tác
  giả nói rõ trong Appendix D.2, nhưng điều này làm nhiễu con số 82.4.
- **Chỉ simulator**, chỉ một họ task (lắp ráp interlocking). Không có robot thật.
- Action là primitive rời rạc chạy bằng script rule-based; không có low-level
  policy học được. Kết quả nói về **planning**, không nói gì về control.
- Reflection **một vòng**, chỉ dùng ảnh cuối. Tác giả nêu đây là hướng mở rộng.
- Oracle policy đạt 97% — trần thực tế là 97 chứ không phải 100.

## 7. Liên hệ với workspace

- Là paper duy nhất trong tập này đo **chi phí inference** một cách hệ thống —
  hữu ích khi cân nhắc test-time compute trong ngân sách real-time của
  [02-realtime-chunking](../../02-realtime-chunking/). Kết luận thực dụng: một
  vòng reflection ~11 s/bước là **không** dùng được cho điều khiển liên tục, chỉ
  dùng được ở tầng plan chạy tần số thấp.
- Kết quả MCTS âm là cảnh báo trực tiếp cho bất kỳ kế hoạch nào định thêm search
  vào VLA: nếu môi trường cho phép "làm lại từ đầu", value function gần như phẳng
  và search sẽ hại nhiều hơn lợi.
- Cùng nhóm "tưởng tượng tương lai" với [01_seer.md](01_seer.md) nhưng khác hoàn
  toàn về vị trí trong pipeline — xem bảng so sánh trong
  [../01_tong_quan.md](../01_tong_quan.md).

## 8. Thử nghiệm tiếp theo

1. **Định lượng lập luận "task thêm được là quan trọng"**: phân tầng 100 task
   test theo độ sâu của dependency graph, rồi báo SR theo tầng. Nếu reflection
   chỉ thắng ở tầng sâu nhất thì lập luận đứng vững; nếu không, +0.2 điểm là
   nhiễu.
2. **Loại bỏ rò rỉ test khỏi dữ liệu DDM** và chạy lại. Cần thiết để con số 82.4
   có ý nghĩa.
3. **Nhiều vòng reflection với context dài hơn**: dùng một VLM context lớn để
   đưa cả $\tilde{I}_{t+1}, \dots, \tilde{I}_{t+H}$ vào thay vì chỉ ảnh cuối. Đây
   là hướng tác giả để ngỏ và là cách rẻ nhất để kiểm tra xem trần hiệu năng do
   cơ chế hay do context.
