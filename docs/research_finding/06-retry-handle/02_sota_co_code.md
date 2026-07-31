# Paper có mã nguồn công khai — chỉ mục khả năng tái lập

Hai trong bốn paper có repository công khai. Đây là **Verified** ở mức đường dẫn
tồn tại (kiểm tra 31 Jul 2026); chưa clone, cài môi trường hay chạy checkpoint
trong workspace.

| Paper | Báo cáo | Code / trang dự án | Ràng buộc chính |
|---|---|---|---|
| RoboMonkey | [01_robomonkey.md](test_time_verification/01_robomonkey.md) | [Code](https://github.com/robomonkey-vla/RoboMonkey), [trang dự án](https://robomonkey-vla.github.io) | Verifier LLaVA-7B train trên 20M cặp preference, 8× H100. Deploy cần 28 GB VRAM và chạy ở 1.5 Hz. |
| SC-VLA | [01_sc_vla.md](online_refinement/01_sc_vla.md) | [Code](https://github.com/Kisaragi0/SC-VLA) | Stage I rẻ (50k iteration, 1× L40). Stage II cần 0.5–3 triệu bước môi trường online — chỉ khả thi trong simulator. |
| FailSafe | [01_failsafe.md](failure_recovery_data/01_failsafe.md) | [Trang dự án](https://jimntu.github.io/FailSafe/) — **chưa có code** | Paper ghi "plan to release"; trang dự án hiện không có link GitHub. Train cần 32× H100. |
| FLARE | [02_flare.md](failure_recovery_data/02_flare.md) | **Unknown** — paper không ghi repo hay trang dự án | Pipeline gồm MimicGen + LoRA trên π0.5 + Gemini-2.5-Pro + Any6D. Khó tái lập nhất trong tập. |

## Phần tái lập được mà không cần repo

Ba kỹ thuật trong tập này không phụ thuộc code của tác giả:

1. **Sparse World Imagination (SC-VLA)** — hai head MLP đọc hidden state trung
   gian, hai loss MSE. Nhãn $p_t$ và $\Delta s_t$ tính thẳng từ trajectory demo.
   Mang 10 trong 14 điểm cải thiện của SC-VLA.
2. **Perturbation & bridging (FLARE)** — nội suy tuyến tính cho vị trí, SLERP cho
   xoay. Không cần simulator vật lý cho phần bridging.
3. **Systematic verification bằng replay (FailSafe)** — phát lại trajectory qua
   pose sửa và giữ mẫu chỉ khi task chuyển từ fail sang success. Cần simulator có
   motion planning, không cần model nào.

## Cách dùng chỉ mục

- **Verified:** repo có đường dẫn công khai và trả HTTP 200 tại thời điểm kiểm
  tra.
- **Unknown:** version code, checkpoint, license, và việc có chạy được ở workspace
  này hay không.
- **Planned:** chỉ clone hoặc chạy khi một thử nghiệm có hypothesis, dataset nhỏ,
  ước lượng disk/RAM/VRAM và tiêu chí thành công — theo
  [.agents/03_conventions.md](../../../.agents/03_conventions.md).
