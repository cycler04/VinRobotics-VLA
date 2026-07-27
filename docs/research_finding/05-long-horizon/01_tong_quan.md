# Long-horizon manipulation trong VLA — tổng hợp 15 paper

Tài liệu này là điểm vào của thư mục `05-long-horizon`. Nó nêu cách phân loại,
so sánh chéo, các mâu thuẫn giữa các paper, và đề xuất thử nghiệm. Chi tiết từng
paper nằm ở các báo cáo riêng được liên kết bên dưới.

**Hai danh sách nguồn**, cả hai đã đọc hết:

| Danh sách | Số paper | Trục | Trạng thái |
|---|---|---|---|
| [paper_link.txt](paper_link.txt) | 12 | Theo chủ đề long-horizon | Đã đọc đủ 12 |
| [sota_with_code.txt](sota_with_code.txt) | 6 (3 mới, 3 trùng) | **Có mã nguồn công khai** | Đã đọc đủ |

Toàn bộ PDF nằm ở `docs/papers/05-long-horizon/`. Bảy paper (PALM, LoHo-Manip,
Anticipation-VLA, MemoryVLA++, LingBot-VA, ACoT-VLA, SeedPolicy) được tải bổ sung
từ arXiv ngày 27/07/2026 và đã đối chiếu tiêu đề với danh sách trước khi đọc.

Paper trong danh sách thứ hai được **đánh dấu bằng banner `[SOTA-CODE]`** ngay
dưới tiêu đề báo cáo, không tách thư mục riêng — lý do và chỉ mục theo trục
khả-năng-tái-lập nằm ở [02_sota_co_code.md](02_sota_co_code.md).

## 1. Câu hỏi nghiên cứu của đợt đọc này

Long-horizon manipulation hỏng ở đâu, và mỗi paper vá vào đâu trong pipeline?

Câu hỏi phụ định hướng: nếu workspace chỉ được chọn **một** thay đổi để cải thiện
hiệu năng long-horizon, thì thay đổi nào có tỉ lệ lợi ích trên chi phí cao nhất?

## 2. Phân loại và chỉ mục

Năm nhóm dưới đây được chia theo **cơ chế can thiệp**, không theo thời gian xuất
bản hay theo benchmark. Lý do: cùng một cụm từ "long-horizon" đang chỉ ít nhất
bốn vấn đề khác nhau (xem mục 3), và chỉ có cách chia theo cơ chế mới làm lộ ra
rằng phần lớn các paper này **trực giao với nhau**, không cạnh tranh.

| Thư mục | Cơ chế | Paper |
|---|---|---|
| [hierarchical_agent/](hierarchical_agent/) | Suy luận subtask ở tầng cao, action ở tầng thấp | [π0.5](hierarchical_agent/01_pi0_5.md), [Hi Robot](hierarchical_agent/02_hi_robot.md), [LoHoVLA](hierarchical_agent/03_loho_vla.md), [LoHo-Manip](hierarchical_agent/04_loho_manip.md), [Anticipation-VLA](hierarchical_agent/05_anticipation_vla.md) |
| [memory_modules/](memory_modules/) | Bộ nhớ thời gian gắn thêm vào policy | [MemoryVLA](memory_modules/01_memoryvla.md), [MemoryVLA++](memory_modules/02_memoryvla_pp.md), [SeedPolicy](memory_modules/03_seedpolicy.md) |
| [future_prediction/](future_prediction/) | Dự đoán trạng thái tương lai để dẫn hướng hoặc sửa quyết định | [Seer](future_prediction/01_seer.md), [ReflectVLM](future_prediction/02_reflective_planning.md), [PALM](future_prediction/03_palm.md), [LingBot-VA](future_prediction/04_lingbot_va.md), [ACoT-VLA](future_prediction/05_acot_vla.md) |
| [skill_chaining/](skill_chaining/) | Xử lý điểm nối và lan truyền lỗi giữa các subtask | [Long-VLA](skill_chaining/01_long_vla.md) |
| [recovery_data/](recovery_data/) | Đổi thành phần dữ liệu, không đổi kiến trúc | [RaC](recovery_data/01_rac.md) |

**Taxonomy giữ nguyên sau hai đợt bổ sung, tổng 7 paper mới.** Cả bốn paper của
đợt một (PALM, LoHo-Manip, Anticipation-VLA, MemoryVLA++) lẫn ba paper của đợt hai
(LingBot-VA, ACoT-VLA, SeedPolicy) đều rơi vào nhóm có sẵn — **không lần nào cần
lập thư mục mới**. Đợt hai đến từ một danh sách được chọn theo tiêu chí hoàn toàn
khác (có mã nguồn, không theo chủ đề), nên việc nó vẫn khớp là bằng chứng độc lập
rằng cách chia theo cơ chế bắt đúng không gian thiết kế.

Một điều chỉnh cần ghi nhận: [ACoT-VLA](future_prediction/05_acot_vla.md) đề xuất
một trục thứ tư — hướng dẫn ở **không gian action**, bên cạnh ngôn ngữ và thị
giác. Nó được xếp vào `future_prediction/` vì **vị trí trong pipeline** giống hệt
Seer (dự báo trạng thái tại $t+n$ để điều kiện action), chỉ khác modality của thứ
được dự báo. Ma trận đầy đủ nằm ở mục 7 của báo cáo ACoT-VLA.

**Còn thiếu**: ba paper benchmark trong `paper_link.txt` (CALVIN, LIBERO,
LongBench) chưa được đọc riêng. Chúng xuất hiện xuyên suốt như môi trường đánh
giá; nếu cần hiểu kỹ giao thức chấm điểm thì phải đọc bổ sung.

## 3. "Long-horizon" đang chỉ bốn vấn đề khác nhau

Đây là kết luận quan trọng nhất của đợt đọc. Bốn chế độ hỏng độc lập:

1. **Không biết làm gì tiếp theo** — phân rã goal thành subtask.
   → hierarchical agent.
2. **Không biết mình đã làm tới đâu** — non-Markovian, hai quan sát giống nhau
   cần hành động khác nhau. Ví dụ kinh điển: *Push Buttons*, ảnh trước và sau khi
   nhấn gần như đồng nhất.
   → memory module. PALM đánh cùng vấn đề bằng một vô hướng progress.
3. **Hỏng tại điểm nối giữa subtask** — dynamic coupling, distribution shift ở
   biên, lan truyền lỗi.
   → skill chaining.
4. **Hỏng vì lỗi tích luỹ và không biết cách thoát khỏi trạng thái lỗi** — demo
   sạch không dạy phục hồi.
   → recovery data.

MemoryVLA++ đề xuất một chế độ hỏng **thứ năm** tách bạch với bốn cái trên:

5. **Không dự đoán được vật sẽ ở đâu** — task có động lực học ngoại sinh. Ví dụ:
   nắm vật trên băng chuyền đang chạy; nắm sớm quá hoặc muộn quá đều hỏng. Memory
   không giải được vì thông tin cần thiết nằm ở tương lai, không ở quá khứ.
   → imagination / world model.

Future prediction cắt ngang: Seer phục vụ (1)+(3) ở mức action, ReflectVLM phục
vụ (1) ở mức plan, PALM phục vụ (2)+(3), MemoryVLA++ phục vụ (5).

Hệ quả thực dụng: **so sánh trực tiếp các con số giữa hai nhóm khác nhau là vô
nghĩa.** MemoryVLA++ đạt 98.4% trên LIBERO còn π0.5 dọn bếp trong nhà lạ 10–15
phút — chúng không đo cùng một thứ.

## 4. Bảng so sánh chéo

| Paper | Vấn đề (mục 3) | Can thiệp ở đâu | Nhãn dữ liệu cần thêm | Backbone / quy mô | Đánh giá |
|---|---|---|---|---|---|
| π0.5 | 1 | Kiến trúc + dữ liệu | Subtask dạng câu, bounding box, lệnh nói | SigLIP 400M + Gemma 2.6B + expert 300M | Nhà thật, 10–15 phút |
| Hi Robot | 1 | Kiến trúc (2 model) + dữ liệu tổng hợp | Skill 1–3 s + prompt/utterance tổng hợp | 2× PaliGemma-3B | 3 robot thật, prompt mở |
| LoHoVLA | 1 | Kiến trúc hợp nhất + vòng điều khiển | Subtask theo bước | PaliGemma-3B + LoRA | Chỉ Ravens sim |
| LoHo-Manip | 1 | Manager tách rời + trace 2D | Vị trí pixel end-effector + phân đoạn primitive (**sinh tự động bằng VLM**) | Manager 4B + executor π0.5 | LIBERO, VLABench, Franka thật |
| Anticipation-VLA | 1 (+5) | Manager đệ quy theo value | Subgoal phân tầng $H$ mức + nhãn tiến độ 3 lớp | UMM Bagel + executor π0.5 | LIBERO, VLABench, Arx-X5 thật |
| MemoryVLA | 2 | Module gắn thêm | **Không cần gì thêm** | Prismatic 7B + DiT 300M | 6 benchmark + 12 task thật |
| MemoryVLA++ | 2 + 5 | Module gắn thêm + world model | Video thao tác (không cần nhãn action) | + SVD 1.5B đóng băng | 5 benchmark + 17 task thật, 3 robot |
| Seer | 1, 3 | Objective + attention mask | Không cần (dùng được play data không nhãn) | 65M/315M trainable | LIBERO, CALVIN, 6 task thật |
| ReflectVLM | 1 | Test-time compute + post-training | Không (dùng oracle relabel) | LLaVA-1.5-13B + InstructPix2Pix | Chỉ sim lắp ráp |
| PALM | 2, 3 | Affordance foresight + progress head | Affordance 4 loại (6 mô hình ngoài) + progress liên tục | GPT-2 style + DiT | CALVIN, LIBERO, xArm6 thật |
| Long-VLA | 3 | Attention mask + 1 chiều action | Nhãn pha nhị phân mỗi frame | MDT / HULC (nhỏ) | L-CALVIN + 2 task thật |
| RaC | 4 | **Chỉ thành phần dữ liệu** | Loại segment + cờ `is_human` | MM-DiT 368M | 3 task thật + 1 sim |
| LingBot-VA | 1, 2, 5 | Chuỗi AR hợp nhất video + action | **Không cần gì thêm** | Wan2.2-5B + action stream 350M = 5.3B | RoboTwin 2.0, LIBERO, 6 task thật |
| ACoT-VLA | 1, 3 | Hai reasoner trong không gian action, trên π0.5 | **Không cần gì thêm** (demo là nhãn) | π0.5 (SigLIP + Gemma 2B) + EAR/IAR | LIBERO, LIBERO-Plus, VLABench, AgiBot G1 |
| SeedPolicy | 2 | Trạng thái ẩn đệ quy + cổng từ logit attention | **Không cần gì thêm** | 33M (Tf) / 147M (CNN) — **một RTX 4090D** | RoboTwin 2.0 (50 task), 5 task thật |

## 5. Mâu thuẫn và điểm căng giữa các paper

### 5.1 Dữ liệu suy luận thắng bước suy luận lúc chạy — nhưng có điều kiện

Bốn paper độc lập chạm vào cùng một hiệu ứng:

- **π0.5**: biến thể *implicit HL* — có dữ liệu subtask trong training nhưng
  **không** suy luận subtask lúc chạy — xếp thứ hai, chỉ sau model đầy đủ. Bỏ hẳn
  dữ liệu HL (`no HL`) thì kém hẳn.
- **ReflectVLM**: *w/o reflect@test* đạt 82.2% so với 82.4% khi có reflection lúc
  chạy — chênh 0.2 điểm với chi phí gấp 25 lần (0.45 s vs 11.10 s mỗi bước).
- **PALM**: bốn head giải mã affordance **được gỡ bỏ lúc inference**; chỉ latent
  còn lại. Giám sát affordance định hình biểu diễn, nhưng không cần giải mã khi
  chạy.
- **Anticipation-VLA**: gắn một VLM lập kế hoạch **tĩnh** vào π0.5 cho 76.0 trên
  LIBERO — **thấp hơn** π0.5 trần (76.8). Thêm suy luận lúc chạy mà không thích
  ứng thì **có hại**.

Đọc ghép: **giá trị nằm ở tín hiệu giám sát định hình biểu diễn, không ở việc
chạy thêm một bước lý luận.** Ngoại lệ duy nhất là khi bước lý luận đó **thích
ứng theo tiến độ thực tế** — xem 5.2.

Mức tin cậy: **cao**. Bốn bằng chứng độc lập, bốn domain, cùng chiều. Đây là kết
luận có giá trị vận hành lớn nhất của cả tập.

### 5.2 Một model hay hai model? — đã có câu trả lời, kèm điều kiện

Trong đợt đọc đầu, đây là lỗ hổng thực nghiệm rõ nhất: không ai chạy so sánh
matched. Hai paper mới đã lấp:

- **LoHo-Manip** dùng chính **π0.5 làm executor**, fine-tune trên cùng dữ liệu,
  thêm một manager 4B ở trên. VLABench avg **0.39 vs 0.24**; real-world OOD vượt
  π0.5 fine-tune trên cùng 100 demo. Manager còn nâng được một executor thứ hai
  (StarVLA 0.18 → 0.24).
- **Anticipation-VLA** cũng dùng π0.5 làm executor: LIBERO avg **80.8 vs 76.8**,
  LIBERO-Long **63.2 vs 54.6**.

Nhưng Anticipation-VLA đồng thời **thu hẹp** kết luận: π0.5 + VLM tĩnh cho 76.0,
tức là **thua** π0.5 trần. Vậy điều kiện không phải "tách hai tầng", mà là:

> Tầng cao chỉ có giá trị khi nó **chạy lại và cập nhật theo tiến độ thực tế**.
> Một planner chạy một lần lúc đầu thì thêm nhiễu chứ không thêm thông tin.

Cả hai paper thắng đều thoả điều kiện này: LoHo-Manip dự đoán *phần còn lại* của
kế hoạch mỗi lần được gọi; Anticipation-VLA kích hoạt lại theo tín hiệu value.
Long-VLA cho bằng chứng bổ trợ ở hướng ngược lại — moving policy tách rời **cố
định** cải thiện bước 1–4 nhưng sụp ở bước 5 (34.2 vs 51.1).

Mức tin cậy: **trung bình-cao**. Hai so sánh matched executor, hai benchmark khác
nhau, cùng chiều. Vẫn chưa ai so hệ hai tầng với **π0.5 được train lại có dữ liệu
HL tương đương** — nên phần đóng góp của dữ liệu vs của kiến trúc chưa tách hẳn.

### 5.3 Nhớ nhiều hơn có tốt hơn không? — đã giải quyết

MemoryVLA (bản hội nghị) cho thấy $L = 64$ tụt về đúng mức của $L = 4$ (67.7 vs
71.9), không giải thích. MemoryVLA++ đo lại trên **ba** setting với thang $L$
khác nhau cho mỗi setting:

| Setting | Small | Default | Large |
|---|---|---|---|
| SimplerEnv (4/16/64) | 67.7 | **71.9** | 67.7 |
| Libero-Long-90 (8/16/32) | 94.2 | **95.6** | 95.6 |
| Real-Temporal (64/256/512) | 78 | **84** | 81 |

Kết luận đúng: **$L$ tối ưu phụ thuộc task**, không phải "memory dài thì tệ".
Real-Temporal cần $L = 256$, gấp 16 lần SimplerEnv. Nghịch lý ở bản hội nghị là
hệ quả của việc quét một thang $L$ duy nhất trên một benchmark ngắn.

Mức tin cậy: **trung bình**. Ba setting, nhưng vẫn chỉ ba điểm quét mỗi setting
và không có quy tắc chọn $L$ theo độ dài task.

### 5.4 Dữ liệu lỗi tổng hợp có dùng được không? — hai paper nói ngược nhau

- **RaC** lập luận rõ: hành vi phục hồi do người diễn từ trạng thái "dàn dựng"
  hoặc giả tạo **không phản ánh** lỗi mà policy thật sẽ gặp, vì lỗi gắn chặt với
  chính policy. Do đó phải thu on-policy, có người can thiệp.
- **LoHo-Manip** làm đúng cái RaC bác bỏ: tổng hợp dữ liệu failure-recovery bằng
  cách **thay vật đã nắm bằng một vật graspable khác trong cảnh**, và báo cáo là
  có tác dụng.
- **Hi Robot** cũng thắng nhờ dữ liệu tổng hợp, nhưng là tổng hợp *prompt người
  dùng*, không phải tổng hợp *trạng thái lỗi vật lý* — không cùng loại.

Không paper nào trích dẫn paper kia. Cách hoà giải hợp lý nhất: RaC tổng hợp lỗi
ở **tầng action** (policy phải học thoát ra bằng vận động), LoHo-Manip tổng hợp
lỗi ở **tầng ngữ nghĩa** (manager chỉ cần nhận ra "đang cầm nhầm vật"). Nhận ra
dễ hơn thoát ra rất nhiều — chính là lập luận verification-generation gap của
RaC, áp dụng ngược lại có lợi cho LoHo-Manip.

Mức tin cậy: **thấp**. Đây là diễn giải của người đọc, chưa có thí nghiệm nào
phân xử. Xem đề xuất 8.2.

## 6. Kết luận: đâu là can thiệp đáng giá nhất

Xếp theo tỉ lệ **lợi ích đo được / chi phí tích hợp**, cho bối cảnh của workspace
này (nghiên cứu, chưa có robot thật, chưa có training loop):

1. **RaC — thành phần dữ liệu.** Duy nhất trong tập không đụng kiến trúc và không
   đổi loss, mà cho khoảng cách lớn nhất: 78.3% với 5 giờ dữ liệu so với 75% với
   ~89 giờ của ALOHA Unleashed. Kèm theo một phép đo rẻ và tái lập được ngay:
   DROID chỉ có 3.68% episode chứa recovery. PALM củng cố cùng luận điểm từ hướng
   khác: bỏ 942 trajectory có nhãn tay làm mất nhiều CALVIN Avg Len (−0.90) hơn
   bỏ toàn bộ dữ liệu in-the-wild (−0.58).
2. **Progress head (PALM) — một số thực mỗi bước.** Ablation cho thấy riêng nó
   đóng góp +0.46 CALVIN Avg Len (4.02 → 4.48) và chỉ tốn một chiều output. Rẻ
   hơn hẳn phần affordance của cùng paper (vốn cần 6 mô hình ngoài để gán nhãn).
   Cùng chức năng "biết khi nào chuyển bước" với ngưỡng $K$ của LoHoVLA và value
   3 lớp của Anticipation-VLA — ba cách, chênh nhau một bậc chi phí.
3. **MemoryVLA — module memory.** Yêu cầu dữ liệu nhẹ nhất trong cả tập: một
   camera third-person + instruction. Lợi ích tỉ lệ đúng với mức phụ thuộc thời
   gian của task (+9 điểm trên task general, **+26** trên task temporal). Nay đã
   có số latency: **+4%** so với baseline, 0.8 GB.
4. **Long-VLA — attention mask.** Thay đổi nhỏ nhất về code, architecture-agnostic,
   đã chứng minh trên hai backbone. Rào cản duy nhất là heuristic điểm cắt pha
   còn thủ công.
5. **Seer — objective.** Baseline long-horizon rẻ nhất có thể tự train (65M tham
   số trainable), có code công khai, chạy được với play data **không nhãn**.
6. Manager tách rời (LoHo-Manip, Anticipation-VLA) nay đã có bằng chứng matched
   là thắng, nhưng đòi hỏi hai model và một executor π0.5 sẵn có. Đúng hướng, chưa
   phải bước đầu cho workspace này.
7. Hierarchical agent một-model (π0.5, Hi Robot, LoHoVLA) đòi hỏi nhãn subtask,
   dữ liệu nhiều nguồn và hạ tầng huấn luyện lớn.

**Mức tin cậy tổng thể của xếp hạng: trung bình.** Nó dựa trên chi phí tích hợp
(quan sát được từ paper) và lợi ích báo cáo (chưa được tái lập độc lập).

## 7. Khoảng trống mà tập paper này chỉ ra cho code hiện có

Đối chiếu với contract canonical v0.1 mô tả ở
[.agents/02_architecture.md](../../../.agents/02_architecture.md):

| Cần thêm | Paper đòi hỏi | Mức độ |
|---|---|---|
| Provenance **mức segment** + cờ `is_human` | RaC | Cao — rẻ, và mở ra phép đo ở mục 8.1 |
| Giá trị **progress liên tục** mỗi frame | PALM | Cao — rẻ, lợi ích đo được lớn |
| Nhãn pha nhị phân mỗi frame | Long-VLA | Trung bình — nhẹ |
| Vị trí pixel end-effector mỗi frame | LoHo-Manip | Trung bình — **sinh tự động bằng VLM**, không cần người |
| Nhãn tiến độ 3 lớp (progress/achieve/none) | Anticipation-VLA | Trung bình |
| Nhãn subtask dạng câu theo timestep | π0.5, Hi Robot, LoHoVLA, LoHo-Manip | Cao chi phí |
| Subgoal phân tầng $H$ mức | Anticipation-VLA | Rất cao — nặng nhất trong 12 paper |
| Nhãn affordance 4 loại | PALM | Rất cao — cần 6 mô hình ngoài |
| Video thao tác không nhãn action (adapt world model) | MemoryVLA++ | Thấp — dùng lại episode sẵn có |
| Không cần gì thêm | MemoryVLA, Seer | — |

Cảnh báo còn nguyên giá trị: π0.5 zero-pad action vector tới chiều lớn nhất trong
mọi dataset và normalize theo quantile 1%/99% **của từng dataset riêng**. Đây
đúng là kiểu giả định mà `.agents/02_architecture.md` cảnh báo — cùng shape không
có nghĩa là cùng semantics.

## 8. Thử nghiệm tiếp theo, xếp theo chi phí

### 8.1 Rẻ, không cần GPU, làm được ngay

**Đo tỉ lệ recovery trong dataset local.** Tái lập phép đo DROID của RaC: phân
loại từng segment thành recovery / correction / forward progress dựa trên khoảng
cách trạng thái tới tiền tố và hậu tố của demo, chạy qua `vla-data-tools`. Kết
quả trả lời trực tiếp "dữ liệu đang thiếu gì" trước khi đụng tới bất kỳ model nào.

Tiêu chí thành công: có được con số % episode chứa ≥1 recovery, đối chiếu được
với mốc 3.68% của DROID.
Ràng buộc: ước lượng dung lượng, RAM và thời gian trước khi chạy, theo
[.agents/03_conventions.md](../../../.agents/03_conventions.md). `LeRobotReader`
hiện đọc toàn bộ Parquet vào RAM.

### 8.2 Trung bình, cần GPU đơn

**Phân xử mâu thuẫn dữ liệu lỗi tổng hợp (mục 5.4).** So dữ liệu recovery tổng
hợp kiểu LoHo-Manip (thay vật đã nắm) với dữ liệu can thiệp on-policy kiểu RaC,
trên cùng task và cùng ngân sách frame. Giả thuyết cần kiểm: lỗi tổng hợp đủ dùng
ở **tầng nhận biết** nhưng không đủ ở **tầng vận động**.

**So sánh trực tiếp memory vs hierarchy trên cùng task temporal.** Vẫn chưa paper
nào lấp. Chạy MemoryVLA và một manager kiểu LoHo-Manip trên cùng bộ task đòi hỏi
nhớ trạng thái (Seq. Push Buttons, Clean Table & Count, Mikasa-Robo). Giả thuyết
cần bác bỏ: hierarchy đủ để thay memory.

**Chỉ progress head, không affordance.** Thêm một progress head vào VLA có sẵn,
không làm gì khác. Nếu thu được phần lớn khoảng cách của PALM thì toàn bộ hạ tầng
gán nhãn affordance là không cần thiết.

### 8.3 Đắt, cần nhiều GPU hoặc robot

**Tách đóng góp dữ liệu khỏi đóng góp kiến trúc trong so sánh một-model vs
hai-model** (mục 5.2): so hệ manager + π0.5 với **π0.5 được train lại trên đúng
lượng dữ liệu HL mà manager dùng**. Đây là điều kiện matched thật sự mà cả
LoHo-Manip lẫn Anticipation-VLA đều chưa làm.

**Kiểm tra tính cộng dồn của các cơ chế.** Không paper nào ghép hai cơ chế trở
lên và đo. Ứng viên rẻ nhất: Long-VLA phase mask + PALM progress head (cả hai đều
chỉ thêm một chiều vào action vector).

## 9. Điều tập paper này **không** trả lời

- **Không paper nào kết hợp hai cơ chế trở lên và đo tính cộng dồn.** Tất cả đều
  ablate cơ chế của chính mình so với không có gì. Không đổi sau cả hai đợt bổ
  sung. Ứng viên rẻ nhất để tự kiểm: ACoT-VLA (EAR: *đi đâu tiếp*) + PALM progress
  head (*đã đi được bao xa*) — cả hai đều không cần nhãn thêm.
- **Đối chứng chéo giữa các nhóm — nay đã có một ngoại lệ.**
  [SeedPolicy](memory_modules/03_seedpolicy.md) chạy **ba** cơ chế memory
  (ARMT-style, MemoryVLA-style, SEGA) trên cùng backbone Diffusion Policy và cùng
  giao thức, trên 10 task: SEGA > MemoryVLA-style > ARMT-style, khoảng cách lớn
  nhất ở task dài. Cảnh báo: đó là bản **tái hiện** MemoryVLA trong DP, không phải
  MemoryVLA đầy đủ với VLM 7B. Các đối chứng còn thiếu vẫn nguyên: dòng MemoryVLA
  không so với hệ hai tầng; PALM không so với MemoryVLA; Anticipation-VLA,
  LoHo-Manip và ACoT-VLA cùng xây trên π0.5, cùng đo trên VLABench, mà **không
  paper nào so với hai paper kia**.
- **Chi phí inference vẫn rời rạc.** Đã có: MemoryVLA +4%, MemoryVLA++ +29%
  (0.241 s trên RTX 4090); LoHo-Manip +19% mỗi episode; Hi Robot 73 ms/bước tầng
  thấp; ReflectVLM 11.10 s/bước; LingBot-VA "async nhanh gấp 2× sync" (không có
  số tuyệt đối). **Vẫn thiếu hoàn toàn**: PALM, ACoT-VLA, SeedPolicy — cả ba đều
  thêm module vào đường dẫn inference của policy real-time mà không đo. Với
  ACoT-VLA và SeedPolicy thì ta **tự đo được** vì có code.
- **Mã nguồn**: nay có 5 paper với repo công khai và 1 có checkpoint — xem
  [02_sota_co_code.md](02_sota_co_code.md). Trước đợt bổ sung thứ hai chỉ có Seer.
- **Chất lượng báo cáo giảm theo độ mới.** Trong 7 paper mới nhất, **4 paper không
  có mục Limitations đúng nghĩa**: MemoryVLA++ (34 trang journal, không có mục
  nào), LingBot-VA (chỉ "Future Work"), ACoT-VLA (không có), SeedPolicy (có tên
  mục nhưng không liệt kê giới hạn nào). Kèm theo đó là thói quen **không thảo
  luận các hồi quy của chính mình** — MemoryVLA++ tụt trên RememberColor5/9 và
  Libero-Plus Camera/Background; ACoT-VLA tụt trên Camera/Noise và
  Commonsense/Instruction. Mục 6 của từng báo cáo đã ghi lại những chỗ này.

## 10. Hai kết luận kỹ thuật được xác nhận độc lập

Hiếm khi hai nhóm khác nhau tìm ra cùng một điều bằng hai kiến trúc khác nhau. Có
hai trường hợp như vậy trong tập này, và cả hai đều đáng tin hơn phần còn lại:

1. **Action không cần biểu diễn thị giác tương lai đã khử nhiễu hoàn toàn.**
   [MemoryVLA++](memory_modules/02_memoryvla_pp.md) đo được **1 bước denoise là
   tốt nhất** (44.4; 3 bước 44.6; 5 bước 43.6).
   [LingBot-VA](future_prediction/04_lingbot_va.md) đạt cùng kết luận qua *Noisy
   History Augmentation*, chỉ khử nhiễu tới $s = 0.5$–$0.6$. Hệ quả: dùng world
   model làm **bộ trích đặc trưng động lực học** rẻ hơn nhiều so với dùng nó làm
   **bộ sinh ảnh** — và đối lập trực tiếp với
   [ReflectVLM](future_prediction/02_reflective_planning.md) (sinh pixel đầy đủ,
   11.10 s/bước).
2. **Giá trị nằm ở tín hiệu giám sát, không ở bước lý luận lúc chạy.** Bốn bằng
   chứng ở mục 5.1, cộng thêm ACoT-VLA: biến thể **đóng băng LLM backbone** đạt
   đúng cùng 98.5 trung bình trên LIBERO như bản tune đầy đủ.
