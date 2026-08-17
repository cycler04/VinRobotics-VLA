# Retry / Recovery trong VLA — tổng hợp 9 paper

## Ý tưởng chính

Chín paper cùng nhắm vào một triệu chứng: VLA được train trên demo sạch, thành
công, nên khi thực thi lệch đi thì không biết làm lại. Nhưng chúng can thiệp ở
nhiều vị trí khác nhau, từ post-training, sinh action, giám sát trước lỗi tới
chẩn đoán và phục hồi sau lỗi. Vì vậy không nên gọi tất cả là “retry system”.

**Verified:** phạm vi này gồm đúng chín PDF đang có trong
[`docs/papers/retry-handle/`](../../papers/retry-handle/). Chỉ mục nguồn
chuẩn là [paper_link.txt](../../papers/retry-handle/paper_link.txt); trạng
thái code nằm ở [sota_co_code.md](sota_co_code.md).

## 1. Câu hỏi nghiên cứu

Khi một rollout đi chệch, hệ thống thiếu gì: một policy được dạy tốt hơn từ đầu,
một cách sinh action tốt hơn ngay lúc chạy, hay khả năng nhận ra mình đã sai và
làm lại?

## 2. Chỉ mục theo giai đoạn can thiệp

Ba nhóm, chia theo **lúc nào cơ chế tác động**:

| Nhóm | Tác động lúc nào | Paper |
|---|---|---|
| [training/](training/) | **Trước deploy** — đổi trọng số policy, bằng dữ liệu hoặc objective | [NORA-1.5](training/nora_1_5.md), [SC-VLA](training/sc_vla.md), [RePO-VLA](training/repo_vla.md) |
| [action_generation/](action_generation/) | **Lúc sinh action** — cải thiện hoặc kiểm tra action trước khi thực thi | [RoboMonkey](action_generation/robomonkey.md), [ThinkAct](training/thinkact.md), [FPC-VLA](action_generation/fpc_vla.md) |
| [failure_adaptation/](failure_adaptation/) | **Sau khi đã lệch** — phát hiện lỗi đã xảy ra rồi sửa | [FailSafe](failure_adaptation/failsafe.md), [FLARE](failure_adaptation/flare.md), [ViFailback](failure_adaptation/vifailback.md) |

Ranh giới không tuyệt đối: FailSafe và FLARE đều *cũng* phải train lại policy trên
dữ liệu mới, ThinkAct *cũng* cần RL huấn luyện MLLM. Mỗi paper xếp theo **chỗ đóng
góp chính** — thứ mà bỏ đi thì paper không còn gì. Các giao thoa ghi trong phần
"Liên hệ" của từng báo cáo.

Bốn lưu ý phân loại:

- **RoboMonkey không phải recovery.** Nó lọc action *trước* khi thực thi, không
  phát hiện lỗi đã xảy ra, không quay lui. Xếp vào tập này vì vị trí can thiệp
  (test time, chống lỗi), không vì cơ chế.
- **SC-VLA cũng không phải recovery.** "Self-correcting" ở đây là *policy tự tinh
  chỉnh action của mình*, không phải phát hiện và sửa failure. Nó nâng độ chính
  xác để lỗi ít xảy ra, chứ không xử lý lỗi đã xảy ra.
- **NORA-1.5 cũng không phải recovery.** Toàn bộ can thiệp là post-training
  offline; lúc deploy không có gì ngoài chính policy. Xếp vào tập vì cùng trục
  *độ tin cậy*, và vì paper tự nêu rằng reward kiểu khoảng-cách-tới-demo gây phục
  hồi lỗi kém ở trạng thái off-distribution — đúng chẩn đoán của FLARE/FailSafe.
- **ThinkAct có cơ chế recovery nhưng chưa có phép đo.** Nó phát hiện lỗi và
  replan bằng chính reasoning MLLM khi đọc lại video $o_{t-N:t}$, không cần
  monitor riêng. Bằng chứng chỉ là ba hình định tính; không có tỉ lệ phát hiện
  hay tỉ lệ phục hồi.

**FailSafe**, **FLARE**, **RePO-VLA** và hệ thống **ViFailback + executor** vừa
xây dựng khả năng phục hồi sau lỗi vừa đo nó. ThinkAct có cơ chế nhưng chưa đo;
FPC-VLA chủ yếu ngăn lỗi tại keyframe trước khi action được thực thi.

## 3. So sánh tối thiểu

| Paper | Can thiệp ở đâu | Cần thêm gì | Bằng chứng chính | Robot thật |
|---|---|---|---|---|
| RoboMonkey | Test time, policy đóng băng | Verifier 7B + 20M cặp preference synthetic | OOD thật 35% → 60%; LIBERO-Long 49.8% → 56.5% | Có (WidowX-250S) |
| FailSafe | Dữ liệu + assistant chạy song song | 131k cặp failure–action sinh trong sim, VLM 7B | ManiSkill: π0-FAST +4.0, OpenVLA-OFT +8.0, OpenVLA +22.6 | **Không** |
| FLARE | Dữ liệu + skill bank + MLLM monitor | Augmentation, 10–20 demo/reset skill, Gemini-2.5-Pro | RoboMimic 9 task: π0.5 72.2% → 84.0% | Có (Piper arm) |
| SC-VLA | Kiến trúc (2 head phụ) + RL online | Nhãn tính từ demo có sẵn; 0.5–3M bước môi trường cho OAR | ManiSkill 0.72 → 0.82 (SPI) → 0.86 (+OAR) | Có, nhưng **chỉ SPI** |
| NORA-1.5 | Kiến trúc (action expert) + DPO offline | World model V-JEPA2-AC 1.3B; 960 giờ H100 cho giai đoạn expert | LIBERO 94.5 → 95.0 (DPO chỉ +0.6); Galaxea A1 thật +13.08 | Có (Galaxea A1) |
| ThinkAct | Kiến trúc dual-system + GRPO | MLLM 7B + DiT policy 432M, 16× A100, quỹ đạo 2D trích bằng detector | LIBERO 76.8 (DiT-Policy) → 84.4; SimplerEnv +11.4…+16.9 | **Không** |
| FPC-VLA | Fusion + VLM supervisor tại gripper keyframe | Qwen2.5-VL-7B, synthetic correction QA, 16× H100 | Ablation task average 74.4 (không supervisor) → 82.1; robot thật 86.0% task success | Có (Xiaomi, ALOHA) |
| RePO-VLA | RAI + progress value + value-conditioned policy | Failure/recovery rollout, V-JEPA value model, π0.5 | FRBench-Sim injected failure 15.0 → 37.0; real adversarial 20 → 75% chỉ ở 4× data/2 task | Có (bimanual) |
| ViFailback | External failure VLM + visual symbol + executor | 5,202 ALOHA trajectory, 58,128 VQA, Qwen3-VL-8B | Unseen real tasks: 52.4 → 73.0 (VSF), 50.8 → 74.6 (PMC) | Có (ALOHA) |

Không so xếp hạng trực tiếp các con số: chín paper dùng benchmark khác nhau
(SIMPLER/Bridge, ManiSkill, RoboMimic, ManiSkill3, LIBERO, SimplerEnv) và base
policy khác nhau (OpenVLA, OpenVLA-OFT/π0-FAST, π0.5, GR00T N1.5, NORA,
DiT-Policy).

## 4. Điều có thể kết luận từ chín paper

1. **Chẩn đoán chung: lỗi nằm ở chế độ dữ liệu, không ở kiến trúc.** FLARE nêu rõ
   nhất — demo là *trajectory-monotonic*, nên policy học tương quan giả giữa pose
   của chính nó và tiến độ task; gặp pose lạ nhưng môi trường hợp lệ thì nó tưởng
   là OOD và bỏ cuộc. FailSafe đồng ý theo hướng khác: demo sạch nên model chưa
   từng thấy trạng thái lỗi. Đây trùng khớp với kết luận của
   [RaC](../long-horizon/recovery_data/01_rac.md) ở corpus long-horizon.
2. **"Retry" và "reset" là hai bài toán khác nhau.** Taxonomy ID/OOD của FLARE là
   đóng góp khái niệm sạch nhất trong tập: ID error = môi trường còn hợp lệ, chỉ
   pose robot lạ → chỉ cần làm lại; OOD error = môi trường đã hỏng (ly đổ) →
   không action nào của task policy cứu được, cần skill dọn hiện trường riêng.
   FailSafe **chỉ** xử lý loại thứ nhất (tự nhận: motion-level, không xử lý được
   object-level).
3. **Phần "retry" rẻ và hiệu quả; phần "reset" đắt và chưa chạy tốt.** Ablation
   của FLARE: chỉ dùng augmentation (không có reset skill) đã vượt π0.5; thêm
   toàn bộ máy móc reset (demo người + adapter riêng + MLLM monitor) chỉ thêm
   **3.5%**. Và bản thân reset skill chỉ đạt 84–88% với object lớn nhưng **20–24%**
   với object nhỏ đã đổ. **Inferred:** nếu ngân sách hạn chế, làm phần retry
   trước là quyết định đúng.
4. **Verifier học preference bền hơn value function offline RL.** RoboMonkey cho
   thấy V-GPS bị reward hacking — sample quá 8 action thì performance *giảm* —
   trong khi verifier preference tiếp tục scale. Và học so sánh tương đối tổng
   quát hoá OOD tốt hơn hồi quy RMSE tuyệt đối (6% ở 64 sample).
5. **VLM thương mại không thay được finetune trên dữ liệu failure có action.**
   Bảng V của FailSafe: GPT-4o và Gemini-2.5-flash phát hiện *có lỗi* ở mức
   0.62–0.70, nhưng xác định đúng loại lỗi < 20% và cosine similarity của action
   sửa ≈ 0. Qwen2.5-VL luôn trả "không có lỗi". Nhưng FLARE cho thấy MLLM
   thương mại **đủ tốt cho vai trò khác**: Gemini-2.5-Pro phân loại ID/OOD đạt
   88–96%. Kết luận: MLLM dùng được để *phân loại và điều phối*, không dùng được
   để *sinh action sửa lỗi*.
6. **Số headline đều cần đọc lại.** "+22.6%" của FailSafe là trên OpenVLA có
   baseline 14.7%; trên hai baseline mạnh chỉ +4.0 và +8.0. "84.0% vs 57.8%" của
   FLARE phần lớn là do backbone π0.5; đóng góp riêng là +11.8. "16% fewer steps"
   của SC-VLA là delta nội bộ SPI→OAR, so với baseline mạnh nhất chỉ 8.7%.
   NORA-1.5 mang tên reward và DPO, nhưng trên LIBERO kiến trúc mang +6.6 còn DPO
   chỉ +0.6; "+45.56 điểm so với π0" trên Galaxea là do π0 sập ở nhóm có
   distractor, so với NORA thì là +12.2. ThinkAct hơn CoT-VLA đúng +0.5 trên
   LIBERO overall; con số đáng tin là +7.6 so với chính DiT-Policy của họ.
7. **Reward proxy offline đã đủ chín để thay reward thủ công.** NORA-1.5 dùng
   world model dự đoán embedding tương lai, ThinkAct dùng quỹ đạo 2D của gripper.
   Cả hai kiểm chứng được từ dữ liệu có sẵn, không cần simulator chính xác cho
   embodiment đích và không cần giờ robot. Đây là điểm khác biệt lớn nhất so với
   SC-VLA (0.5–3M bước môi trường online).
8. **Ngữ cảnh thời gian là điều kiện cần để phát hiện lỗi.** ThinkAct phải mở
   input từ một frame $o_t$ thành đoạn video $o_{t-N:t}$ mới self-correct được;
   FailSafe cần cửa sổ 10 frame liên tiếp. Hai paper độc lập cùng chỉ ra: **schema
   mức frame đơn là không đủ.**
9. **“Ngăn lỗi”, “phục hồi” và “học từ failure” là ba contract khác nhau.**
   FPC-VLA kiểm tra action trước execution; ViFailback chẩn đoán rồi giao guidance
   cho executor ngoài; RePO-VLA distill recovery vào policy và deploy không có
   monitor. Cả ba tăng success nhưng không thay thế nhau trực tiếp.
10. **Headline recovery dễ bị data/protocol confound.** RePO-VLA đạt `20 → 75%`
    ở cấu hình 4× recovery data và chỉ hai task; ViFailback thay cả supervisor
    lẫn executor; FPC-VLA dùng dữ liệu supervisor theo embodiment. Không headline
    nào tự cô lập causal gain của một module.

## 5. Giới hạn của tổng hợp

- **Không paper nào đo tỉ lệ ID error so với OOD error trong thực tế.** Cả kinh
  tế của phần Reset lẫn thứ tự ưu tiên đều phụ thuộc con số này.
- **Chi phí giám sát online vẫn chưa đầy đủ.** FPC-VLA là ngoại lệ hữu ích khi
  báo `0.176 s` ở non-keyframe và `1.766 s` ở keyframe, nhưng không báo deadline
  hay jitter; RoboMonkey chạy 1.5 Hz; FailSafe trộn overhead với replanning;
  FLARE và ViFailback không báo latency monitor end-to-end.
- **Bằng chứng recovery robot thật còn nhỏ và bị confound.** FLARE chỉ có 2 task
  × 40 trial; RePO-VLA 10 trial/task và kết quả mạnh nhất dùng 4× data trên hai
  task; ViFailback 21 trial/task và thay cả supervisor lẫn executor. FailSafe vẫn
  hoàn toàn trong simulator; SC-VLA không chạy OAR trên robot thật.
- Các thành công được báo cáo không phải kết quả đã tái lập trong workspace.
- Checkout hiện không chứa package `src/vla_data_tools/` dù overview và
  `pyproject.toml` vẫn mô tả entry point đó. Code đang thấy chủ yếu là pipeline
  ego-video inference/inspection; chưa có robot execution. Mọi đề xuất bên dưới
  là **Planned**, không phải capability hiện có.

## 6. Bước kiểm chứng tiếp theo

Xếp theo chi phí tăng dần:

1. **Planned — gán nhãn ID/OOD cho failure có sẵn.** Chạy một policy, gán nhãn
   tay vài chục failure theo taxonomy của FLARE. Rẻ nhất, và quyết định luôn có
   nên đầu tư vào phần Reset hay không.
2. **Planned — sinh nhãn SPI offline.** $p_t$ và $\Delta s_t$ của SC-VLA tính
   được thẳng từ trajectory có sẵn, không cần trường schema mới. Đây là can thiệp
   duy nhất trong corpus ban đầu mà dữ liệu hiện tại đã đủ.
3. **Planned — tái lập scaling law của RoboMonkey offline.** Sample $k$ action từ
   một VLA trên dataset local, đo RMSE oracle theo $k$, fit power law. Nếu độ dốc
   ≈ 0 thì tiền đề của RoboMonkey không áp dụng cho dữ liệu này.
4. **Planned — sinh dữ liệu perturbation & bridging.** Chỉ cần nội suy tuyến tính
   + SLERP giữa hai pose, không cần simulator vật lý. Điều kiện tiên quyết là
   trajectory phải có ranh giới subtask.
5. **Planned — replay verification kiểu FailSafe** trong một simulator có motion
   planning, đo tỉ lệ candidate recovery pass. Đắt nhất, và chỉ làm sau khi bước
   1 xác nhận rằng lỗi cần recovery đủ phổ biến.

## 7. Khoảng trống schema mà tập paper này chỉ ra

Các paper trong corpus cần annotation ở mức **frame hoặc segment**, không phải mức
episode:

| Cần gì | Paper nào cần | Schema v0.1 có chưa |
|---|---|---|
| `(observation, ground-truth action, instruction)` | RoboMonkey | **Có** |
| Pose end-effector + gripper theo timestep, chỉ số bước/độ dài episode | SC-VLA (SPI) | **Có** |
| Ranh giới subtask trong episode | FLARE | Chưa |
| Nhãn frame: `perturbation` / `bridging` / `task` | FLARE | Chưa |
| `failure_type`, `subtask`, vector $\Delta A$ 7 chiều gắn timestep | FailSafe | Chưa |
| Multi-view đồng bộ (3 view) + cửa sổ 10 frame liên tiếp | FailSafe | Cần xác minh |
| Truy cập frame $o_{t+N}$ trong cùng episode (subgoal image) | NORA-1.5 (WM subgoal) | **Có** |
| Cặp (action sinh ra, action chuẩn) để tính $L_1$ | NORA-1.5 (GTA) | **Có** |
| Quỹ đạo 2D gripper $K=8$ keypoint chuẩn hoá `[0,1]` trên ảnh | ThinkAct | Chưa — trích được từ video bằng detector |
| Cửa sổ video $o_{t-N:t}$ có độ dài xác định gắn timestep | ThinkAct | Cần xác minh |
| Gripper keyframe + correction direction/magnitude | FPC-VLA | Chưa; cần semantics action 7-D rõ ràng |
| `trajectory_type`, `failure_phase`, `recovery_start`, dense `value_label` | RePO-VLA | Chưa |
| Failure keyframe/subtask/type + visual-symbol geometry + VQA provenance | ViFailback | Chưa |

Đây cùng loại khoảng trống mà [RaC](../long-horizon/recovery_data/01_rac.md)
đã chỉ ra — **provenance ở mức segment, không phải mức episode**. Hai corpus độc
lập cùng chỉ vào một chỗ.

## Nguồn

- [Chỉ mục PDF và metadata](../../papers/retry-handle/paper_link.txt)
- Chín báo cáo theo thư mục ở mục 2; mỗi báo cáo liên kết trực tiếp tới PDF nguồn.
- [docs/papers/retry-handle/deep-research-report.md](../../papers/retry-handle/deep-research-report.md)
  là khảo sát rộng hơn có sẵn từ trước; nó bao gồm nhiều paper **không** có PDF
  trong repo và các citation count chưa xác minh lại. Không dùng làm nguồn chuẩn.
