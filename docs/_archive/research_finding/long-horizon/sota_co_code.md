# Paper có mã nguồn công khai — chỉ mục khả năng tái lập

Năm trong bảy paper còn lưu có repository công khai. Đây là **Verified** ở mức
đường dẫn nguồn; chưa clone, cài môi trường hoặc chạy checkpoint trong workspace.

| Paper | Báo cáo | Code / checkpoint | Ràng buộc chính |
|---|---|---|---|
| LingBot-VA | [lingbot_va.md](future_prediction/lingbot_va.md) | [Code](https://github.com/robbyant/lingbot-va), [checkpoint](https://huggingface.co/robbyant/lingbot-va) | Pretraining 5.3B/1.4T token không thuộc phạm vi tái lập cục bộ. |
| ACoT-VLA | [acot_vla.md](future_prediction/acot_vla.md) | [Code](https://github.com/AgibotTech/ACoT-VLA) | Xây trên π0.5; report ghi train 8× H100. |
| MemoryVLA | [memoryvla.md](memory_modules/memoryvla.md) | [Code](https://github.com/shihao1895/MemoryVLA) | Report ghi backbone Prismatic 7B và DiT 300M. |
| Seer | [seer.md](future_prediction/seer.md) | [Code](https://github.com/OpenRobotLab/Seer/) | 65M tham số trainable, nhưng pretraining vẫn cần dữ liệu và compute phù hợp. |
| SeedPolicy | [seedpolicy.md](memory_modules/seedpolicy.md) | [Code](https://github.com/Youqiang-Gui/SeedPolicy) | Paper báo cáo trên một RTX 4090D; cần xác minh cấu hình trước khi tái lập. |

## Cách dùng chỉ mục

- **Verified:** repo có đường dẫn công khai trong [sota_with_code.txt](sota_with_code.txt).
- **Unknown:** version code, checkpoint, license và script có chạy được ở workspace
  này hay không.
- **Planned:** chỉ clone hoặc chạy khi một thử nghiệm có hypothesis, dataset nhỏ,
  ước lượng disk/RAM/VRAM và tiêu chí thành công.

π0.5 và Hi Robot vẫn có báo cáo/PDF trong corpus, nhưng không được gắn banner
`[SOTA-CODE]` vì chỉ mục hiện tại không ghi repository công khai cho chúng.
