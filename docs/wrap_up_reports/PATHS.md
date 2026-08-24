# Nơi lưu trữ các thành phần

**Workspace root: `/home/tho2/Dung_Workspace`** 

## Các repo (cả hai đều push lên https://github.com/VietnamRobotics/vla_core.git)

| Thành phần                                      | Đường dẫn                                                | Nhánh (Branch)           |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------- |
| Nhánh Memory (VLA + PCMB theo kiểu MemoryVLA)   | `/home/tho2/Dung_Workspace/vla_core`                       | `feature/memoryvla`     |
| Nhánh Baseline (điều khiển không có memory) | `/home/tho2/Dung_Workspace/vla_core-benchmark-libero-base` | `benchmark-libero/base` |

Bắt đầu tại đây trong từng repo:

- `/home/tho2/Dung_Workspace/vla_core/docs/progress_report_2026-08-24.md`
- `/home/tho2/Dung_Workspace/vla_core-benchmark-libero-base/docs/progress_report_2026-08-24.md`
- `/home/tho2/Dung_Workspace/vla_core-benchmark-libero-base/docs/benchmark_protocol.md` (hợp đồng đối sánh / parity contract)
- `README.md` + `CLAUDE.md` tại thư mục gốc của từng repo

## Tài liệu tham khảo MemoryVLA

| Thành phần                                                          | Đường dẫn                                                                      |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Mã nguồn công khai MemoryVLA (code tham chiếu của bài báo)     | `/home/tho2/Dung_Workspace/MemoryVLA`                                            |
| MemoryVLA+ (codebase phát triển tiếp, tham chiếu chế độ group) | `/home/tho2/Dung_Workspace/MemoryVLA-plus`                                       |
| File PDF bài báo                                                    | `/home/tho2/Dung_Workspace/vla_core/docs/memory_module/ref/memoryvla.pdf`        |
| Ghi chú về bài báo                                                | `/home/tho2/Dung_Workspace/vla_core/docs/memory_module/ref/memoryvla.md`         |
| Báo cáo thích ứng của chúng ta                                  | `/home/tho2/Dung_Workspace/vla_core/docs/memory_module/memoryvla_integration.md` |

## LIBERO

| Thành phần                                        | Đường dẫn                 |
| --------------------------------------------------- | ----------------------------- |
| Repo bộ mô phỏng / benchmark                     | `/mnt/SSD4/LIBERO`          |
| Bộ dữ liệu / Datasets (libero_10, libero_90, …) | `/mnt/SSD4/LIBERO/datasets` |

(Tài liệu cũ hơn có nhắc tới `/home/tho2/LIBERO` — đã được chuyển đi; các đường dẫn `/mnt/SSD4` mới là hiện hành.)

## Các lượt huấn luyện & kết quả (gitignored, chỉ lưu trên đĩa)

- Lượt chạy Memory: `/home/tho2/Dung_Workspace/vla_core/runs/libero_long_group_v1…v5`
- Lượt chạy Baseline: `/home/tho2/Dung_Workspace/vla_core-benchmark-libero-base/runs/libero_long_base*`
- Kết quả benchmark tham chiếu (16.4%): `/home/tho2/Dung_Workspace/vla_core-benchmark-libero-base/runs/libero_long_base_v3/eval_60000/results.json`
- Biểu đồ loss: `/home/tho2/Dung_Workspace/plot_loss.sh runs/<run>`

## Mục khác

- Môi trường Python: `/home/tho2/miniconda3/envs/vla_core/bin/python` (conda env `vla_core`)
- Các bản phát hành tiền huấn luyện corpus (Layer-1): `/mnt/SSD4/dataset/releases/{egodex_v06,egoverse_v06}` —
  bản thân repo `data_corpus` KHÔNG có trên máy này; chỉ có các bản phát hành (release) của nó.
- Ghi chú / EDA cấp workspace: `/home/tho2/Dung_Workspace/docs`
