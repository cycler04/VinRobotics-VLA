# Retry / Recovery trong VLA — tổng hợp 4 paper

## Ý tưởng chính

Bốn paper cùng nhắm vào một triệu chứng: VLA được train trên demo sạch, thành
công, nên khi thực thi lệch đi thì không biết làm lại. Nhưng chúng **can thiệp ở
bốn chỗ khác nhau trên đường từ dữ liệu tới action**, và chỉ ba trong bốn paper
thực sự nói về recovery theo nghĩa đen.

**Verified:** phạm vi này gồm đúng bốn PDF đang có trong
[`docs/papers/06-retry-handle/`](../../papers/06-retry-handle/). Chỉ mục nguồn
chuẩn là [paper_link.txt](../../papers/06-retry-handle/paper_link.txt); trạng
thái code nằm ở [02_sota_co_code.md](02_sota_co_code.md).

## 1. Câu hỏi nghiên cứu

Khi một rollout đi chệch, hệ thống thiếu gì: candidate action tốt hơn, khả năng
nhận ra mình đã sai, dữ liệu dạy cách quay lại, hay một vòng tinh chỉnh online?

## 2. Chỉ mục theo cơ chế

| Nhóm | Vấn đề chính | Paper |
|---|---|---|
| [test_time_verification/](test_time_verification/) | Chọn action tốt hơn **trước khi** thực thi, policy đóng băng | [RoboMonkey](test_time_verification/01_robomonkey.md) |
| [failure_recovery_data/](failure_recovery_data/) | Sinh dữ liệu failure + recovery mà demo sạch không có | [FailSafe](failure_recovery_data/01_failsafe.md), [FLARE](failure_recovery_data/02_flare.md) |
| [online_refinement/](online_refinement/) | Tinh chỉnh action online bằng RL với reward nội sinh | [SC-VLA](online_refinement/01_sc_vla.md) |

Hai lưu ý phân loại:

- **RoboMonkey không phải recovery.** Nó lọc action *trước* khi thực thi, không
  phát hiện lỗi đã xảy ra, không quay lui. Xếp vào tập này vì vị trí can thiệp
  (test time, chống lỗi), không vì cơ chế.
- **SC-VLA cũng không phải recovery.** "Self-correcting" ở đây là *policy tự tinh
  chỉnh action của mình*, không phải phát hiện và sửa failure. Nó nâng độ chính
  xác để lỗi ít xảy ra, chứ không xử lý lỗi đã xảy ra.

Chỉ **FailSafe** và **FLARE** thực sự xây dựng khả năng phục hồi sau lỗi.

## 3. So sánh tối thiểu

| Paper | Can thiệp ở đâu | Cần thêm gì | Bằng chứng chính | Robot thật |
|---|---|---|---|---|
| RoboMonkey | Test time, policy đóng băng | Verifier 7B + 20M cặp preference synthetic | OOD thật 35% → 60%; LIBERO-Long 49.8% → 56.5% | Có (WidowX-250S) |
| FailSafe | Dữ liệu + assistant chạy song song | 131k cặp failure–action sinh trong sim, VLM 7B | ManiSkill: π0-FAST +4.0, OpenVLA-OFT +8.0, OpenVLA +22.6 | **Không** |
| FLARE | Dữ liệu + skill bank + MLLM monitor | Augmentation, 10–20 demo/reset skill, Gemini-2.5-Pro | RoboMimic 9 task: π0.5 72.2% → 84.0% | Có (Piper arm) |
| SC-VLA | Kiến trúc (2 head phụ) + RL online | Nhãn tính từ demo có sẵn; 0.5–3M bước môi trường cho OAR | ManiSkill 0.72 → 0.82 (SPI) → 0.86 (+OAR) | Có, nhưng **chỉ SPI** |

Không so xếp hạng trực tiếp các con số: bốn paper dùng bốn benchmark khác nhau
(SIMPLER/Bridge, ManiSkill, RoboMimic, ManiSkill3) và bốn base policy khác nhau
(OpenVLA, OpenVLA-OFT/π0-FAST, π0.5, GR00T N1.5).

## 4. Điều có thể kết luận từ bốn paper

1. **Chẩn đoán chung: lỗi nằm ở chế độ dữ liệu, không ở kiến trúc.** FLARE nêu rõ
   nhất — demo là *trajectory-monotonic*, nên policy học tương quan giả giữa pose
   của chính nó và tiến độ task; gặp pose lạ nhưng môi trường hợp lệ thì nó tưởng
   là OOD và bỏ cuộc. FailSafe đồng ý theo hướng khác: demo sạch nên model chưa
   từng thấy trạng thái lỗi. Đây trùng khớp với kết luận của
   [RaC](../05-long-horizon/recovery_data/01_rac.md) ở corpus long-horizon.
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

## 5. Giới hạn của tổng hợp

- **Không paper nào đo tỉ lệ ID error so với OOD error trong thực tế.** Cả kinh
  tế của phần Reset lẫn thứ tự ưu tiên đều phụ thuộc con số này.
- **Không paper nào báo cáo chi phí giám sát online đầy đủ.** RoboMonkey chạy
  1.5 Hz và tự nhận không hợp control tần số cao; FailSafe đo overhead lẫn với
  replanning của simulator; FLARE không báo cáo latency của MLLM monitor.
- **Đánh giá recovery đều là sim.** FailSafe hoàn toàn không có robot thật.
  SC-VLA có robot thật nhưng không chạy nửa OAR. Chỉ FLARE kiểm chứng đúng cơ chế
  đề xuất trên phần cứng, và chỉ với 2 task × 40 trial.
- Các thành công được báo cáo không phải kết quả đã tái lập trong workspace.
- `vla-data-tools` hiện chỉ đọc/chuyển đổi dataset; chưa có training loop hay
  robot execution. Mọi đề xuất bên dưới là **Planned**, không phải capability
  hiện có.

## 6. Bước kiểm chứng tiếp theo

Xếp theo chi phí tăng dần:

1. **Planned — gán nhãn ID/OOD cho failure có sẵn.** Chạy một policy, gán nhãn
   tay vài chục failure theo taxonomy của FLARE. Rẻ nhất, và quyết định luôn có
   nên đầu tư vào phần Reset hay không.
2. **Planned — sinh nhãn SPI offline.** $p_t$ và $\Delta s_t$ của SC-VLA tính
   được thẳng từ trajectory có sẵn, không cần trường schema mới. Đây là can thiệp
   duy nhất trong tập 4 mà dữ liệu hiện tại đã đủ.
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

Cả bốn paper đều cần annotation ở mức **frame hoặc segment**, không phải mức
episode:

| Cần gì | Paper nào cần | Schema v0.1 có chưa |
|---|---|---|
| `(observation, ground-truth action, instruction)` | RoboMonkey | **Có** |
| Pose end-effector + gripper theo timestep, chỉ số bước/độ dài episode | SC-VLA (SPI) | **Có** |
| Ranh giới subtask trong episode | FLARE | Chưa |
| Nhãn frame: `perturbation` / `bridging` / `task` | FLARE | Chưa |
| `failure_type`, `subtask`, vector $\Delta A$ 7 chiều gắn timestep | FailSafe | Chưa |
| Multi-view đồng bộ (3 view) + cửa sổ 10 frame liên tiếp | FailSafe | Cần xác minh |

Đây cùng loại khoảng trống mà [RaC](../05-long-horizon/recovery_data/01_rac.md)
đã chỉ ra — **provenance ở mức segment, không phải mức episode**. Hai corpus độc
lập cùng chỉ vào một chỗ.

## Nguồn

- [Chỉ mục PDF và metadata](../../papers/06-retry-handle/paper_link.txt)
- Bốn báo cáo theo thư mục ở mục 2; mỗi báo cáo liên kết trực tiếp tới PDF nguồn.
- [docs/papers/06-retry-handle/deep-research-report.md](../../papers/06-retry-handle/deep-research-report.md)
  là khảo sát rộng hơn có sẵn từ trước; nó bao gồm nhiều paper **không** có PDF
  trong repo và các citation count chưa xác minh lại. Không dùng làm nguồn chuẩn.
