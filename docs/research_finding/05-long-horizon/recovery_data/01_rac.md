# RaC — scale recovery và correction thay vì scale demo

## 1. Nguồn

- Tiêu đề: *RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and
  Correction*
- Tác giả: Zheyuan Hu, Robyn Wu, Naveen Enock, Jasmine Li, Riya Kadakia,
  Zackory Erickson, Aviral Kumar (Carnegie Mellon University)
- arXiv: [2509.07953v1](https://arxiv.org/abs/2509.07953), 9 Sep 2025
- Venue: CoRL 2025 **workshop** + arXiv — **không** nằm trong main proceedings
- Trang dự án: https://rac-scaling-robot.github.io/
- PDF trong repo: [docs/papers/05-long-horizon/07_rac_recovery_and_correction.pdf](../../../papers/05-long-horizon/07_rac_recovery_and_correction.pdf)
- Phân loại: **recovery data** (can thiệp ở tầng *thành phần dữ liệu*, không đổi
  kiến trúc và không đổi objective).

## 2. Câu hỏi nghiên cứu

Imitation learning trên demo teleop đang chạm trần **thấp hơn hẳn mức hoàn hảo**:
ngay cả với hơn 5000 demo, model chuyên biệt SOTA chỉ treo được một chiếc áo lên
móc ở khoảng 75% thành công. Nguyên nhân nằm ở **phân phối dữ liệu** hay ở thuật
toán/model?

Luận điểm của paper: nằm ở dữ liệu. Demo thiên lệch về trajectory sạch và thành
công, nên không dạy policy cách xử lý lỗi tích luỹ trong task dài.

## 3. Đóng góp

1. Một **pha huấn luyện mới sau imitation pre-training**, dựa trên rollout có
   người can thiệp, nhấn mạnh hành vi **recovery** (lùi về trạng thái quen thuộc)
   bên cạnh **correction** (đẩy subtask tiến lên).
2. Hai **quy tắc thu thập** đơn giản chuẩn hoá can thiệp.
3. Bằng chứng về **test-time scaling** cho robot policy: success rate tăng tuyến
   tính theo số lần recovery mà policy thực hiện — tương tự long chain-of-thought
   ở LLM.
4. Hiệu quả dữ liệu cao hơn ~1 bậc so với SOTA cùng loại task.

## 4. Method

### 4.1 Định nghĩa hình thức

Cho trajectory $\tau = (s_0, a_0, \dots, s_t)$ với $s_t$ là trạng thái người can
thiệp. Chuỗi hành động người $(a^h_{t+1}, \dots, a^h_{t+k})$ là:

- **recovery segment** nếu trạng thái kết quả $s^h_{t+k}$ nằm trong phân phối
  trạng thái của **tiền tố** demo $\mathcal{D}_{full}[0:t]$;
- **corrective segment** nếu $s^h_{t+k}$ nằm trong phân phối trạng thái **sau**
  bước $t$: $\mathcal{D}_{full}[t+1:H]$.

Điểm gây ngạc nhiên: recovery segment **tự nó không tối ưu cho task** — nó có thể
huỷ tiến độ đã có. Đây là chỗ RaC tách khỏi HG-DAgger, vốn coi can thiệp của
người là "lời giải expert cần bắt chước".

### 4.2 Hai quy tắc

**Rule 1 — recover then correct.** Mỗi lần can thiệp gồm hai pha: người vận hành
đưa robot về vùng trạng thái quen thuộc, **rồi** thực hiện đoạn sửa để hoàn thành
subtask hiện tại. Đảm bảo mỗi can thiệp dạy cả "cách tự reset" lẫn "cách tiến
lên".

**Rule 2 — terminate after intervention.** Kết thúc episode ngay khi đoạn can
thiệp xong. Lý do: nếu để rollout chạy tiếp, các subtask sau sẽ nằm dưới phân
phối trạng thái **hỗn hợp** giữa policy và người — phân phối này không phải phân
phối mà policy tự sinh ra khi chạy độc lập, nên học trên đó tốn mẫu mà ít lợi.

### 4.3 Vì sao recovery có tác dụng — lập luận verification-generation gap

Với nhiều task long-horizon, **tập trạng thái khởi đầu hợp lệ thì rộng** (áo nằm
trên bàn, móc ở đâu đó trong tay robot phía trên áo) nhưng **tập trạng thái đích
thì hẹp** (cổ áo đã xuyên đúng qua móc). Do đó *quay về một trạng thái đã gặp*
dễ hơn nhiều so với *thực hiện đúng subtask*, và học được với ít mẫu hơn.

Hệ quả: xác suất **không bao giờ** thành công một subtask giảm theo hàm mũ theo
số lần thử lại. Policy có hai đường cải thiện độc lập — làm đúng ngay lần đầu,
hoặc reset rồi thử lại.

### 4.4 Bằng chứng rằng dữ liệu hiện có thiếu recovery

Phân tích 1% sub-sample của DROID: chỉ **3.68%** episode chứa ≥1 recovery,
**16.58%** chứa ≥1 correction. Dữ liệu can thiệp HG-DAgger của chính nhóm cũng
lệch mạnh về correction.

### 4.5 Hệ thống

- 2× xArm-7 (7 DoF) với soft gripper thu nhỏ; 60 Hz; RMPFlow làm IK generator
  (tránh va chạm real-time). 1 camera overhead + 2 camera cổ tay.
- **Giao diện clutch trên Oculus Quest**: giữ nút bên = người điều khiển, nhả =
  policy chạy. Đăng ký pose theo **local frame** với delta tương đối
  $T^{c_0}_{c}(t) = (T^v_{c_0})^{-1} T^v_{c_t}$, $\Delta p_k = p_k - p_{k-1}$,
  $\Delta R_k = R_{k-1}^\top R_k$ — người vận hành **không** phải căn chỉnh tư
  thế tay cầm với end-effector trước khi tiếp quản.
- **Công cụ hỗ trợ trực quan**: dùng SAM2 dựng heatmap tần suất robot gripper
  xuất hiện từ vòng demo đầu tiên, phủ lên cửa sổ camera overhead để chỉ cho
  người vận hành biết **vùng in-distribution nào cần recovery về**.

### 4.6 Policy

MM-DiT flow matching (paper ghi 300M trong phần method, Table 1 ghi 368M — **hai
con số không khớp**). ResNet-50 encoder cho 3 camera (ResNet-18 trong sim).
Action chunk $H = 60$ ≈ 1 giây ở 60 Hz; inference mỗi 0.5 s, chạy nửa đầu chunk
rồi replan; 10 bước Euler.

$$
\mathcal{L}_{Flow}(\theta) = \mathbb{E}\Big[\big\lVert v_\theta(\tau, o_t, x_\tau) - (A_t - x_0)\big\rVert_2^2\Big]
$$

**Objective imitation không đổi.** Toàn bộ cải thiện đến từ thành phần dữ liệu.

Lọc dữ liệu: **không** đưa transition từ rollout của chính robot vào training,
trừ khi trajectory hoàn thành toàn bộ task mà không có can thiệp nào.

## 5. Claim → Evidence

### 5.1 Hiệu quả dữ liệu — shirt hanging

| Method | Kiến trúc | Kích thước | Dữ liệu | SR |
|---|---|---|---|---|
| ALOHA Unleashed | Diffusion Transformer | 217M | ~89 giờ (5345 demo) | 75.0% |
| Seed GR-3 | VLA | 4B | 116 giờ demo + dữ liệu vision-language | ~63.6% |
| **RaC** | Flow-matching Transformer | 368M | **5 giờ** (expert + recovery + correction) | **78.3%** |

Đây là số liệu ấn tượng nhất của cả tập paper: **~1 bậc độ lớn** ít dữ liệu hơn
mà cao hơn.

### 5.2 Scaling so với baseline

Ba task thật: shirt-hanging ($K=6$ vòng), airtight-lid-sealing ($K=10$),
clamshell-takeout-box-packing ($K=9$); một task sim: bimanual-assembly. 60 trial
mỗi task thật, 100 trial sim.

- RaC vượt cả **batched full demonstration** và **HG-DAgger** về SR, task
  progress, và **độ dốc** đường scaling.
- Cải thiện hiệu quả dữ liệu ít nhất **2×** so với batched collection.
- HG-DAgger của nhóm này **không** phải baseline yếu: nó vượt batched collection
  ở cùng lượng dữ liệu người, đúng như prior work báo cáo.

### 5.3 Test-time scaling (phát hiện đắt giá nhất)

Hồi quy tuyến tính giữa số recovery trung bình mỗi trajectory thành công và
success rate:

| Task | Fit | $r$ |
|---|---|---|
| Shirt hanging | $y = 1.169x - 0.387$ | 0.714 |
| Airtight lid sealing | $y = 0.179x + 0.099$ | 0.803 |
| Takeout box packing | $y = 0.253x + 0.111$ | 0.877 |

Cả ba đều dương và tương quan khá mạnh. Đây là dạng test-time scaling "o1-style"
nhưng diễn ra **trong không gian action** chứ không phải không gian token.

### 5.4 Hồ sơ hành vi

- **Robustness của checkpoint trung gian**: tỉ lệ rollout dừng sớm / không tiến
  triển giảm nhanh qua các vòng RaC; batched full demonstration **không** có xu
  hướng đó, đặc biệt trong sim.
- **Độ dài rollout thành công** (sim): full demos median 43.0 s / mean 59.8 /
  23/100 thành công; HG-DAgger median 48.0 / mean 58.6 / 22/100; **RaC median
  53.0 / mean 67.2 / 61/100**. Policy RaC chạy lâu hơn vì nó recovery — và thành
  công nhiều hơn gấp gần 3 lần.

### 5.5 Ablation hai quy tắc (sim)

- **Thành phần can thiệp**: RaC giữ tỉ lệ recovery:correction khoảng **1:1 đến
  1:2** qua 4 vòng. Không có Rule 1 (tức HG-DAgger) thì tỉ lệ trượt về **1:3 rồi
  1:10**, phần recovery giảm mạnh ở các vòng sau.
- **Rule 2 riêng lẻ có tác dụng**: "Ours w/o Rule 1" (chỉ có Rule 2) scale tốt
  hơn "Ours w/o Rule 1&2" (HG-DAgger thuần). Cả hai quy tắc đều đóng góp.

## 6. Giới hạn và điểm chưa rõ

- **Venue yếu nhất trong tập**: workshop CoRL, không qua main proceedings. Xếp
  mức tin cậy thấp hơn các paper còn lại, dù nội dung thực nghiệm chắc.
- **Mâu thuẫn số liệu nội bộ**: 300M (mục 4.4) vs 368M (Table 1) cho cùng policy.
- **Không thử trên VLA generalist**. Tác giả nêu đây là hướng tương lai, và thừa
  nhận **chưa rõ hành vi recovery có tự phát sinh trong VLA hay không**. Toàn bộ
  kết quả là trên specialist policy train từ đầu.
- **Chi phí người**: RaC cần người ngồi giám sát rollout và can thiệp đúng lúc
  qua $K = 6$–$10$ vòng, xen kẽ với train lại policy mỗi vòng. So sánh "5 giờ dữ
  liệu vs 89 giờ" **không** tính chi phí thời gian chờ train và thời gian giám
  sát. Đây là điểm mà paper trình bày thuận lợi cho mình.
- **Test-time scaling là tương quan, không phải nhân quả**: đường hồi quy được vẽ
  từ *các checkpoint của các vòng khác nhau*, mà checkpoint sau vừa có nhiều
  recovery hơn vừa được train nhiều dữ liệu hơn. Không có can thiệp thực nghiệm
  nào ép số recovery ở một checkpoint cố định. $r \approx 0.71$–$0.88$ với cỡ mẫu
  vài điểm.
- Định nghĩa recovery/correction dựa trên "nằm trong phân phối trạng thái của
  $\mathcal{D}_{full}$" — trong thực tế được xấp xỉ bằng heatmap SAM2 do người
  nhìn. Không có bộ phân loại tự động.

## 7. Liên hệ với workspace

- RaC là paper duy nhất trong tập **không đụng vào kiến trúc**. Nó nói rằng với
  cùng model và cùng loss, đổi thành phần dữ liệu cho lợi ích lớn hơn mọi thay
  đổi kiến trúc trong 7 paper còn lại (78.3% với 5 giờ vs 75% với 89 giờ). Nếu
  đúng, thứ tự ưu tiên R&D của workspace nên đặt dữ liệu trước kiến trúc.
- Với `vla-data-tools`: cần trường **provenance mức segment** — mỗi transition
  phải biết mình thuộc loại nào (full demo / recovery / correction / rollout tự
  chủ) và có phải do người điều khiển không (`is_human`). Contract canonical v0.1
  chỉ có provenance mức episode. **Đây là khoảng trống cụ thể và đáng vá nhất mà
  tập paper này chỉ ra cho code hiện có.**
- Con số DROID 3.68% recovery / 16.58% correction là kiểm tra rẻ và tái lập được:
  có thể chạy phân tích tương tự trên bất kỳ dataset nào mà `vla-data-tools` đọc
  được, để định lượng "dataset này thiếu recovery tới mức nào".

## 8. Thử nghiệm tiếp theo

1. **Tái lập phép đo tỉ lệ recovery trên dataset local**: viết bộ phân loại
   segment (recovery vs correction vs forward progress) dựa trên khoảng cách
   trạng thái tới tiền tố/hậu tố của demo, chạy trên dataset trong `dataset/`. Đây
   là thử nghiệm rẻ nhất, không cần robot, và cho biết ngay dữ liệu đang thiếu gì.
   Cần ước lượng dung lượng và RAM trước theo
   [.agents/03_conventions.md](../../../../.agents/03_conventions.md).
2. **Kiểm tra nhân quả của test-time scaling**: ở **một** checkpoint cố định, ép
   số lần recovery (ví dụ bằng cách chèn thủ công nhiễu buộc policy phải reset)
   và đo SR. Nếu quan hệ biến mất thì đường hồi quy chỉ phản ánh chất lượng
   checkpoint, không phải scaling.
3. **RaC trên VLA generalist**: đúng câu hỏi tác giả để ngỏ. Finetune một VLA
   pretrained bằng dữ liệu RaC và vẽ đường test-time scaling. Nếu độ dốc gần 0
   thì hành vi recovery đã có sẵn trong VLA và RaC không cộng thêm.
