# LoHo-Manip — task manager VLM + visual trace làm giao diện với executor

## 1. Nguồn

- Tiêu đề: *Long-Horizon Manipulation via Trace-Conditioned VLA Planning*
- Tác giả: Isabella Liu, An-Chieh Cheng, Rui Yan, Geng Chen, Ri-Zhao Qiu, Xueyan
  Zou, Sha Yi (UC San Diego), Hongxu Yin, Xiaolong Wang, Sifei Liu (NVIDIA /
  UCSD, đồng hướng dẫn)
- arXiv: [2604.21924v1](https://arxiv.org/abs/2604.21924), 23 Apr 2026
- Venue: **preprint**
- Trang dự án: https://www.liuisabella.com/LoHoManip
- PDF trong repo: [docs/papers/05-long-horizon/10_loho_manip_trace_conditioned.pdf](../../../papers/05-long-horizon/10_loho_manip_trace_conditioned.pdf)
- Phân loại: **hierarchical agent** (hai model tách rời, giao diện là *ngôn ngữ +
  quỹ đạo 2D*).

## 2. Câu hỏi nghiên cứu

Ghép chặt planning vào executor gây hai vấn đề: (1) **fragility under drift** —
chuỗi dài khuếch đại lỗi nhỏ, kế hoạch một lần không lường được thất bại từng
phần, che khuất, vật di chuyển; (2) **poor modularity** — đổi embodiment, action
space hay training domain thì phải làm lại cả stack.

Có thể tách hẳn "còn phải làm gì" khỏi "làm thế nào", theo cách mà **cùng một
manager dùng lại được với nhiều executor khác nhau**?

## 3. Đóng góp

1. **Task manager tách rời**, dùng lại được trên nhiều VLA backbone.
2. **Remaining-plan prediction theo receding horizon**: mỗi lần gọi, manager dự
   đoán *phần còn lại* của kế hoạch từ quan sát hiện tại → tự động có progress
   tracking, replanning và recovery **mà không cần failure detector viết tay**.
3. **Visual trace làm prompt điều khiển**: quỹ đạo keypoint 2D được render vào
   ảnh quan sát, executor học kỹ năng chung "đi theo trace".

## 4. Method

### 4.1 Biểu diễn kế hoạch có nhận thức tiến độ

Chuỗi primitive nguyên tử $\bar{S} = [\bar{s}^{(1)}, \dots, \bar{s}^{(K)}]$. Tại
thời điểm $t$, thay vì chỉ phát ra "subtask kế tiếp", manager phát ra **cả tiền
tố đã xong lẫn hậu tố còn lại**:

$$
C^\star_t = [\bar{s}^{(1)}, \dots, \bar{s}^{(k(t)-1)}], \qquad
R^\star_t = [\bar{s}^{(k(t))}, \dots, \bar{s}^{(K)}]
$$

$C_t$ là **language memory** nén gọn; $R_t$ là kế hoạch còn lại.

Visual trace là phần tương lai của quỹ đạo pixel end-effector:

$$
\tau^\star_t = \{p_t, p_{t+1}, \dots, p_{t^e_K}\}
$$

lưu ở dạng waypoint đã resample, render thành visual prompt.

### 4.2 Shift-resilient language memory — lựa chọn quan trọng nhất

Manager **chỉ nhận frame hiện tại** trong cả training lẫn inference; lịch sử
được mã hoá thành **tóm tắt văn bản** $C_t$. Nhận đầu vào $(x, o_t, C_{t-1})$,
dự đoán $(C_t, R_t, \tau_t)$.

Lý do được nêu rõ: nạp lịch sử ảnh dài làm tăng latency và **đẩy manager vào
distribution shift** khi thực thi thật lệch khỏi demo mượt; còn phát ra mỗi một
"subtask kế tiếp" thì dòng lệnh mất ổn định khi thất bại lặp lại. Tóm tắt văn
bản giữ được bookkeeping tường minh mà tránh cả hai.

Đây là câu trả lời **thay thế** cho vấn đề mà
[MemoryVLA](../memory_modules/01_memoryvla.md) giải bằng memory bank latent: cùng
mục tiêu "biết đã làm tới đâu", nhưng một bên là bộ nhớ vector học được, một bên
là chuỗi ký tự do model tự viết lại mỗi bước.

### 4.3 Closed loop ngầm

Không có failure detector. Nếu subtask hỏng (cốc chưa được nắm), thế giới phản
ánh điều đó, manager tiếp tục liệt kê hạng mục chưa xong trong output tiếp theo,
kèm trace được cập nhật. Progress tracking, replanning và recovery đều là **hệ
quả** của việc dự đoán "phần còn lại", không phải module riêng.

### 4.4 Data pipeline

Từ video RGB thao tác: dùng VLM nền tảng làm frame grounding, object detection,
captioning để xác định sự kiện tương tác; cắt thành primitive nguyên tử với frame
bắt đầu/kết thúc; trích vị trí pixel end-effector thành trace. Toạ độ chuẩn hoá
về $[0, 1000]$, frame nào VLM không phát hiện được thì loại bỏ.

Nguồn: Bridge subset (định dạng OXE) + RoboVQA + EgoPlan-BenchIT.

**Dữ liệu failure-recovery tổng hợp**: lọc episode grasp-and-place, tìm frame
chuyển tiếp, rồi tạo dữ liệu hỏng "giả" bằng cách **thay vật đã nắm bằng một vật
graspable khác trong cảnh**. Model học phát hiện lỗi ngữ nghĩa (đang cầm sushi
thay vì bắp) và sinh subtask sửa ("Drop the sushi") kèm trace phục hồi.

### 4.5 Huấn luyện và triển khai

- Manager: khởi tạo từ VLM pretrained (Qwen3-VL họ 4B), đóng băng vision encoder,
  fine-tune language model. Kết quả là LoHo-Manip-4B.
- Executor: kiến trúc **π0.5**, khởi tạo từ checkpoint gốc, fine-tune để điều
  kiện trên trace đã render (và tuỳ chọn text subtask).
- Lịch chạy: manager được gọi **một lần mỗi 100 bước** của executor. Trên A6000,
  planner ~2 Hz, executor ~10 Hz.

## 5. Claim → Evidence

### 5.1 Manager như một model lập kế hoạch độc lập

| Benchmark | LoHo-Manip-4B | Đối thủ mạnh nhất |
|---|---|---|
| RoboVQA (BLEU avg) | **63.1** | RynnBrain-8B 62.1; Qwen3-VL-8B 60.8; Gemini-3.0-Flash 37.3; GPT-4V 26.8 |
| EgoPlan-Bench2 (acc %) | **56.7** | Gemini-3.0-Flash 48.8; ThinkAct-7B 48.2; GPT-4V 32.6 |
| EmbodiedBench EB-Alfred | **0.38** | GPT-4o mini 0.24; Qwen3-VL-4B 0.19 |
| EmbodiedBench EB-Habitat | **0.38** | GPT-4o mini 0.33; Qwen3-VL-4B 0.30 |

Trajectory prediction (thấp hơn là tốt):

| Benchmark | Method | DFD | HD | RMSE |
|---|---|---|---|---|
| ShareRobot-T | Embodied-R1-3B | 0.3426 | 0.3002 | 0.2388 |
| | **LoHo-Manip-4B** | **0.2309** | **0.2058** | **0.1559** |
| VABench-V | Hamster-13B | 0.2124 | 0.2045 | 0.1825 |
| | **LoHo-Manip-4B** | **0.2123** | **0.1821** | **0.1469** |

Model 4B vượt Gemini-3.0-Flash trên cả hai benchmark lập kế hoạch — chi tiết đáng
chú ý, và nhất quán với kết quả GPT-4o/GPT-4 kém ở
[Hi Robot](02_hi_robot.md) và [ReflectVLM](../future_prediction/02_reflective_planning.md).

### 5.2 End-to-end manipulation

VLABench (executor là π0.5 trong mọi trường hợp):

| Method | In-Dist. | Cross-Cat. | Common Sense | Semantic Instr. | Unseen Texture | Avg |
|---|---|---|---|---|---|---|
| π0-fast | 0.29 | 0.18 | 0.21 | 0.20 | 0.24 | 0.22 |
| π0.5 | 0.37 | 0.22 | 0.21 | 0.17 | 0.25 | 0.24 |
| **LoHo-Manip** | **0.54** | **0.23** | **0.36** | **0.42** | **0.39** | **0.39** |

LIBERO: 97.5 avg (Spatial 98.0, Object 98.6, Goal 98.0, **Long 95.2**), vượt
StarVLA 96.6, MolmoAct 86.6, GR00T-N1.5 86.5, π0-fast 85.5.

Real world: Franka + 2 RealSense (top-view + wrist), 100 demo teleop qua data
pipeline tự động. Vượt π0.5 fine-tune trên **cùng 100 mẫu** ở các tình huống OOD
(vật mới cho single-step; bố cục và tổ hợp ngôn ngữ mới cho multi-step). Bản
trích chỉ có biểu đồ, không có bảng số.

### 5.3 Manager dùng lại được với executor khác

| Method | VLABench Avg |
|---|---|
| StarVLA | 0.18 |
| **StarVLA + LoHo-Manip manager** | **0.24** |
| π0.5 | 0.24 |
| **π0.5 + LoHo-Manip manager** | **0.39** |

Bằng chứng cho tuyên bố modularity: cùng một manager checkpoint nâng được hai
executor khác nhau.

### 5.4 Ablation data curation

| Cấu hình | ShareRobot-T DFD | VABench-V DFD |
|---|---|---|
| Không có subtask/trace curated | 0.2437 | 0.2500 |
| **Có** | **0.2309** | **0.2123** |

### 5.5 Chi phí

| Cấu hình | Latency 1 episode (sim) |
|---|---|
| Không có task manager | ~72 s |
| Có task manager | ~86 s |

+19% cho toàn episode, vì manager chỉ chạy 1 lần mỗi 100 bước executor.

## 6. Giới hạn và điểm chưa rõ

- **Tác giả tự nêu** (Appendix E): phụ thuộc vào độ chính xác của subtask
  grounding và trace prediction; **trace 2D không diễn đạt được** thao tác đòi
  hỏi độ chính xác cao hoặc contact-rich.
- **Preprint, chưa peer-review.**
- Kết quả real-world chỉ ở dạng biểu đồ, **không có bảng số** trong bản trích —
  không trích được con số cụ thể.
- Trace được trích bằng cách hỏi VLM định vị end-effector từng frame rồi lấy tâm
  bounding box; **frame nào hỏng thì bỏ**. Tỉ lệ frame bị bỏ không được báo cáo.
- Dữ liệu failure-recovery là **giả tạo** (thay vật đã nắm bằng vật khác), khác
  hẳn cách tiếp cận of [RaC](../recovery_data/01_rac.md) vốn lập luận rằng lỗi
  tổng hợp từ trạng thái dàn dựng không phản ánh lỗi thật của policy. Hai paper
  mâu thuẫn trực tiếp ở điểm này và không paper nào trích dẫn paper kia.
- **Chưa rõ**: manager chạy 1 lần / 100 bước executor. Không có ablation chu kỳ
  này. Với task cần phản ứng nhanh, 100 bước có thể quá thưa.

## 7. Liên hệ với workspace

- **Đây là paper lấp đúng lỗ hổng "một model vs hai model"** đã ghi ở mục 5.2
  của [../01_tong_quan.md](../01_tong_quan.md): executor là chính π0.5, được
  fine-tune trên **cùng dữ liệu**, và hệ hai tầng thắng (VLABench 0.39 vs 0.24;
  real-world OOD). Kết luận cũ cần điều chỉnh — xem mục 5.2 đã cập nhật.
- Với `vla-data-tools`: yêu cầu **vị trí pixel end-effector mỗi frame** (để dựng
  trace) và **phân đoạn primitive nguyên tử**. Cả hai đều sinh được tự động từ
  video bằng VLM, không cần người gán nhãn — đây là hướng gán nhãn rẻ nhất trong
  toàn bộ 12 paper.
- Giao diện *rendered trace* rất đáng chú ý về mặt kỹ thuật: nó biến prompt cấp
  cao thành **pixel trong ảnh đầu vào**, nên không cần đổi kiến trúc executor,
  chỉ cần fine-tune. Có thể thử với bất kỳ policy nào nhận RGB.

## 8. Thử nghiệm tiếp theo

1. **Quét chu kỳ gọi manager**: 1/10/50/100/200 bước executor. Đo success rate và
   latency. Cần thiết để biết "closed loop ngầm" thực sự đóng lại nhanh tới đâu.
2. **Trace thật vs trace dự đoán**: chạy executor với trace ground-truth (oracle)
   để tách trần do manager khỏi trần do executor — tương tự baseline "human
   high-level" của [Hi Robot](02_hi_robot.md).
3. **Đối chứng failure-recovery giả vs thật**: dữ liệu recovery tổng hợp của
   LoHo-Manip so với dữ liệu can thiệp on-policy của
   [RaC](../recovery_data/01_rac.md), trên cùng task. Hai paper phát biểu ngược
   nhau về việc lỗi dàn dựng có dùng được không; thử nghiệm này phân xử được.
