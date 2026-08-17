# Paper có mã nguồn công khai — chỉ mục khả năng tái lập

Ba trong chín paper có repository công khai. Đây là **Verified** ở mức đường dẫn
tồn tại (kiểm tra 31 Jul 2026; cập nhật corpus 05 Aug 2026); chưa
clone, cài môi trường hay chạy checkpoint trong workspace.

| Paper | Báo cáo | Code / trang dự án | Ràng buộc chính |
|---|---|---|---|
| NORA-1.5 | [nora_1_5.md](training/nora_1_5.md) | [Code](https://github.com/declare-lab/nora-1.5), [trang dự án](https://declare-lab.github.io/nora-1.5), [checkpoint HF](https://huggingface.co/declare-lab/nora-1.5) | **Đầy đủ nhất trong tập**: code + checkpoint + backbone chỉ 3.3B. Giai đoạn action-expert tốn ≈960 giờ H100 nếu train lại; chỉ chạy DPO thì rẻ hơn nhiều. |
| RoboMonkey | [robomonkey.md](action_generation/robomonkey.md) | [Code](https://github.com/robomonkey-vla/RoboMonkey), [trang dự án](https://robomonkey-vla.github.io) | Verifier LLaVA-7B train trên 20M cặp preference, 8× H100. Deploy cần 28 GB VRAM và chạy ở 1.5 Hz. |
| SC-VLA | [sc_vla.md](training/sc_vla.md) | [Code](https://github.com/Kisaragi0/SC-VLA) | Stage I rẻ (50k iteration, 1× L40). Stage II cần 0.5–3 triệu bước môi trường online — chỉ khả thi trong simulator. |
| FailSafe | [failsafe.md](failure_adaptation/failsafe.md) | [Trang dự án](https://jimntu.github.io/FailSafe/) — **chưa có code** | Paper ghi "plan to release"; trang dự án hiện không có link GitHub. Train cần 32× H100. |
| ThinkAct | [thinkact.md](training/thinkact.md) | [Trang dự án](https://jasper0314-huang.github.io/thinkact-vla/) — **chưa có code** | Không tìm thấy repo chính thức (`NVlabs/ThinkAct` trả 404). Ba giai đoạn train trên 16× A100: Qwen2.5-VL-7B + GRPO + DiT policy 432M + Q-Former. |
| FLARE | [flare.md](failure_adaptation/flare.md) | **Unknown** — paper không ghi repo hay trang dự án | Pipeline gồm MimicGen + LoRA trên π0.5 + Gemini-2.5-Pro + Any6D. Khó tái lập nhất trong tập. |
| FPC-VLA | [fpc_vla.md](action_generation/fpc_vla.md) | [Trang dự án](https://fpcvla.github.io/) — nút Code/Model hiện trỏ `#`, chưa có artifact tải được | Data engine cần RLDS có pose 7-D và binary gripper; supervisor Qwen2.5-VL-7B train trên 16 H100. |
| RePO-VLA | [repo_vla.md](training/repo_vla.md) | **Unknown** — PDF không ghi project page/repo/dataset URL | Cần failure/recovery rollout, adverse-state verification và π0.5; số recovery episode ở cấu hình 2×/4× không được công bố. |
| ViFailback | [vifailback.md](failure_adaptation/vifailback.md) | [Code](https://github.com/x1nyuzhou/ViFailback), [dataset](https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset), [model](https://huggingface.co/sii-rhos-ai/ViFailback-8B), [trang dự án](https://x1nyuzhou.github.io/vifailback.github.io/) | Artifact đã được liên kết công khai; chưa clone hoặc kiểm tra license/runtime trong workspace. |

## Phần tái lập được mà không cần repo

Năm kỹ thuật trong tập này không phụ thuộc code của tác giả:

1. **Sparse World Imagination (SC-VLA)** — hai head MLP đọc hidden state trung
   gian, hai loss MSE. Nhãn $p_t$ và $\Delta s_t$ tính thẳng từ trajectory demo.
   Mang 10 trong 14 điểm cải thiện của SC-VLA.
2. **Perturbation & bridging (FLARE)** — nội suy tuyến tính cho vị trí, SLERP cho
   xoay. Không cần simulator vật lý cho phần bridging.
3. **Systematic verification bằng replay (FailSafe)** — phát lại trajectory qua
   pose sửa và giữ mẫu chỉ khi task chuyển từ fail sang success. Cần simulator có
   motion planning, không cần model nào.
4. **GTA reward (NORA-1.5)** — $-\lVert a^* - a \rVert_1$ giữa action sinh ra và
   action chuẩn. Một dòng code, không cần world model. Đủ để dựng preference
   dataset và kiểm tra pipeline DPO.
5. **Trích quỹ đạo 2D gripper (ThinkAct)** — detector sẵn có → rút gọn bằng
   Ramer–Douglas–Peucker về $K$ keypoint chuẩn hoá `[0,1]`. Chạy được trên video
   có sẵn, kể cả video người không có action label.

## Cách dùng chỉ mục

- **Verified:** repo có đường dẫn công khai và trả HTTP 200 tại thời điểm kiểm
  tra.
- **Unknown:** version code, checkpoint, license, và việc có chạy được ở workspace
  này hay không.
- **Planned:** chỉ clone hoặc chạy khi một thử nghiệm có hypothesis, dataset nhỏ,
  ước lượng disk/RAM/VRAM và tiêu chí thành công — theo
  [.agents/03_conventions.md](../../../.agents/03_conventions.md).
