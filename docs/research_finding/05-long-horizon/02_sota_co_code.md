# Nhóm SOTA có mã nguồn — chỉ mục và đánh giá khả năng tái lập

Tài liệu này là chỉ mục cho tập paper thứ hai, nguồn:
[sota_with_code.txt](sota_with_code.txt). Trục phân biệt của tập này là **khả năng
tái lập** (có code / có checkpoint), không phải cơ chế.

Chỉ mục theo cơ chế cho toàn bộ 15 paper nằm ở
[01_tong_quan.md](01_tong_quan.md).

## 1. Cách đánh dấu, và vì sao không tách thư mục

Ba paper mới của tập này (LingBot-VA, ACoT-VLA, SeedPolicy) **đều rơi vào nhóm cơ
chế đã có**. Không paper nào cần thư mục mới. Vì vậy chúng được đặt đúng nhóm cơ
chế của mình và **đánh dấu bằng banner** `[SOTA-CODE]` ngay dưới tiêu đề, thay vì
tách ra một thư mục riêng.

Lý do: cơ chế và khả-năng-tái-lập là **hai trục trực giao**. Tách thư mục theo
nguồn danh sách sẽ làm vỡ trục cơ chế — MemoryVLA và MemoryVLA++ sẽ nằm hai nơi,
Seer sẽ tách khỏi PALM dù cùng họ future prediction. Đánh dấu giữ được cả hai trục
và tài liệu này làm nhiệm vụ chỉ mục cho trục thứ hai.

> Nếu cần tách hẳn thành thư mục riêng thì báo — di chuyển file và sửa link là
> việc cơ học.

## 2. Chỉ mục

| # | Paper | Nhóm cơ chế | Báo cáo | Code | Checkpoint |
|---|---|---|---|---|---|
| 1 | **LingBot-VA** (*Causal World Modeling for Robot Control*) | future prediction | [04_lingbot_va.md](future_prediction/04_lingbot_va.md) | [robbyant/lingbot-va](https://github.com/robbyant/lingbot-va) | [HuggingFace](https://huggingface.co/robbyant/lingbot-va) |
| 2 | **ACoT-VLA** | future prediction | [05_acot_vla.md](future_prediction/05_acot_vla.md) | [AgibotTech/ACoT-VLA](https://github.com/AgibotTech/ACoT-VLA) | — |
| 3 | **MemoryVLA** | memory modules | [01_memoryvla.md](memory_modules/01_memoryvla.md) | [shihao1895/MemoryVLA](https://github.com/shihao1895/MemoryVLA) | — |
| 4 | **Seer** | future prediction | [01_seer.md](future_prediction/01_seer.md) | [InternRobotics/Seer](https://github.com/InternRobotics/Seer) | — |
| 5 | **SeedPolicy** | memory modules | [03_seedpolicy.md](memory_modules/03_seedpolicy.md) | [Youqiang-Gui/SeedPolicy](https://github.com/Youqiang-Gui/SeedPolicy) | — |
| 6 | **Long-VLA** (*honorable mention*) | skill chaining | [01_long_vla.md](skill_chaining/01_long_vla.md) | **chưa phát hành** | — |

## 3. Sai lệch giữa danh sách và nguồn gốc — đã xác minh

Ba điểm cần ghi nhận, đã kiểm tra trực tiếp trên arXiv ngày 27/07/2026:

1. **LingBot-VA không phải tiêu đề paper.** `arXiv:2601.21998` có tiêu đề
   ***Causal World Modeling for Robot Control***. "LingBot-VA" là tên hệ thống bên
   trong (xuất hiện 33 lần trong PDF). Danh sách dùng tên hệ thống — nhất quán,
   nhưng sẽ không tìm thấy nếu tra theo tiêu đề.
2. **Venue của LingBot-VA chưa xác minh.** Danh sách ghi RSS 2026; bản PDF
   **không ghi venue nào**.
3. **Org của Seer khác nhau.** Danh sách ghi `InternRobotics/Seer`; bản PDF của
   paper ghi `OpenRobotLab/Seer`. Nhiều khả năng là đổi tên tổ chức, chưa xác minh.

Ngoài ba điểm trên, tiêu đề và ID của các mục còn lại khớp.

## 4. Xếp hạng khả năng tái lập trong bối cảnh workspace

Tiêu chí: chạy được với phần cứng và dữ liệu hiện có, không cần cluster.

| Hạng | Paper | Vì sao |
|---|---|---|
| 1 | **SeedPolicy** | Toàn bộ thí nghiệm chạy trên **một RTX 4090D**. Model 33M tham số. Không VLM backbone, không pretrain, **không cần nhãn nào ngoài demo**. Rẻ nhất trong cả 15 paper. |
| 2 | **Seer** | 65M tham số trainable (251M encoder đông lạnh). Chạy được với play data **không có nhãn ngôn ngữ** — đúng tình huống RLDS/OXE. Cần pretrain nhưng quy mô vừa. |
| 3 | **LingBot-VA** | 5.3B tham số, pretrain 1.4T token, 16K giờ dữ liệu — **không tái lập được** phần train. Nhưng là paper duy nhất **có checkpoint công khai**, nên inference và đo đạc thì làm được ngay. |
| 4 | **ACoT-VLA** | Xây trên π0.5, train cần 8× H100. Biến thể **đóng băng LLM** đạt cùng hiệu năng (98.5) nên chi phí có thể thấp hơn nhiều — đáng kiểm tra. |
| 5 | **MemoryVLA** | Backbone Prismatic 7B + DiT 300M, train 8× A100. Cơ chế PCMB thì độc lập backbone, tách ra dùng được. |
| — | **Long-VLA** | Chưa có code. |

## 5. Điều tập "có code" này thay đổi so với đợt đọc đầu

Ba cập nhật thực chất, không chỉ là thêm paper:

### 5.1 Lấp lỗ hổng "không ai đối chứng chéo giữa các nhóm"

Mục 9 của [01_tong_quan.md](01_tong_quan.md) ghi rằng đối chứng chéo gần như không
tồn tại. **SeedPolicy lấp một phần**: nó chạy ba cơ chế memory khác nhau
(ARMT-style, MemoryVLA-style, SEGA) trên **cùng backbone Diffusion Policy và cùng
giao thức huấn luyện**, trên 10 task. Kết quả: MemoryVLA-style > ARMT-style, SEGA >
cả hai; khoảng cách lớn nhất ở task dài.

Cảnh báo khi đọc: đây là bản **tái hiện** MemoryVLA bên trong DP, không phải
MemoryVLA đầy đủ với VLM 7B. Không suy ra được thứ hạng giữa hai hệ thống hoàn
chỉnh.

### 5.2 "Partial denoising là đủ" được xác nhận độc lập hai lần

- [MemoryVLA++](memory_modules/02_memoryvla_pp.md): **1 bước denoise** cho kết quả
  tốt nhất trên Mikasa-Robo (44.4); 3 bước 44.6, 5 bước 43.6.
- [LingBot-VA](future_prediction/04_lingbot_va.md): *Noisy History Augmentation*
  cho phép chỉ khử nhiễu tới $s = 0.5$–$0.6$ với 3 bước Euler, giảm một nửa chi
  phí sinh video.

Hai nhóm khác nhau, hai kiến trúc khác nhau, cùng kết luận: **action không cần
biểu diễn thị giác tương lai đã khử nhiễu hoàn toàn**. Đây là kết luận có giá trị
kỹ thuật cao nhất của tập "có code", và nó đối lập trực tiếp với
[ReflectVLM](future_prediction/02_reflective_planning.md) — vốn sinh ảnh pixel đầy
đủ với chi phí 11.10 s/bước.

### 5.3 Một trục taxonomy thứ tư xuất hiện

[ACoT-VLA](future_prediction/05_acot_vla.md) phân loại 25 phương pháp theo cột
"Guidance" thành `–` / `Visual` / `Linguistics` / `Action`. Ánh xạ sang cách chia
của chúng ta: Linguistics ≈ hierarchical agent, Visual ≈ future prediction, và
**Action là cột mà taxonomy hiện tại chưa tách riêng**.

Quyết định: giữ ACoT-VLA trong `future_prediction/` vì **vị trí trong pipeline**
giống hệt Seer — dự báo trạng thái tại $t+n$ để điều kiện action; chỉ khác modality
của thứ được dự báo. Ma trận đầy đủ nằm ở mục 7 của
[05_acot_vla.md](future_prediction/05_acot_vla.md).

## 6. Một quan sát về chất lượng báo cáo

Cả **ba** paper mới của tập này đều **không có mục Limitations đúng nghĩa**:

- LingBot-VA: chỉ có "Future Work".
- ACoT-VLA: không có mục nào.
- SeedPolicy: có mục tên "Limitations and Future Work" nhưng nội dung **không liệt
  kê giới hạn nào**.

Cộng với [MemoryVLA++](memory_modules/02_memoryvla_pp.md) (34 trang, journal
format, không có Limitations), đó là **4/7 paper mới nhất** trong toàn bộ đợt đọc.
Không paper nào trong ba paper này báo cáo **latency**, dù cả ba đều thêm module
vào đường dẫn inference của policy real-time.

Hệ quả cho workspace: với nhóm này, **giới hạn phải do người đọc tự dựng lại từ
bảng số**, và các hồi quy (chỗ phương pháp mới thua baseline) thường không được
tác giả bàn. Mỗi báo cáo trong thư mục này đã ghi rõ những chỗ đó ở mục 6 của nó.
