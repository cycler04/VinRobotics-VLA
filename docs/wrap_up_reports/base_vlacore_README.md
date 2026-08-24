# vla_core-benchmark-libero-base — baseline LIBERO không sử dụng memory

Nhánh điều khiển cố định (frozen control arm) cho benchmark về memory. Sử dụng cùng model VLA với repo anh em `vla_core/` — **ngoại trừ module memory** — được huấn luyện trên cùng dữ liệu LIBERO với cùng cấu hình tối ưu hóa (optimization), nhằm đảm bảo sự khác biệt duy nhất giữa hai nhánh là (a) ngân hàng bộ nhớ (memory bank) và (b) cấu trúc tương quan batch (batch correlation structure) mà việc huấn luyện memory yêu cầu. `docs/benchmark_protocol.md` là hợp đồng đối sánh (parity contract).

## Model

```
Qwen3.5-0.8B (VLM đóng băng / frozen)
  └─ chia hidden-state theo từng lớp → luồng vision + luồng narrative
       └─ ActionHead: 24 khối cross-attention, flow matching (velocity MSE),
          vị trí RoPE, các action chunk 16 bước (~229M thông số)
```

Không có `model/memory.py`, không có memory gate — hoàn toàn giống từng byte với `vla_core` khi đặt `use_memory=False`. Cấu hình sẵn cho LIBERO: action_dim 7, chunk 16, tắt narrative, backbone đóng băng hoàn toàn (lora_r 0), trọng số chính fp32, autocast bf16.

Repo được fork từ `vla_core` tại commit `2823b4e` (giai đoạn pre-memory), sau đó được bổ sung các thành phần LIBERO tương tự (dataset, trainer, sim eval) nhưng không có module memory.

## So sánh

|                                  | Repo này (baseline)                                                                    | `vla_core` (memory)                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Sampling                         | `SourceTemperatureSampler` — trích xuất độc lập theo từng window               | `EpisodeGroupBatchSampler` — K group × G=16 window sắp xếp theo thời gian của một clip |
| Memory                           | Không có                                                                              | PCMB bank, đệ quy (recurrent) trên toàn group                                               |
| Tất cả các thành phần khác | Giống hệt (data, split, normalization, seed, optimizer, schedule, effective batch 64) | Giống hệt                                                                                     |

Phân bố lề của việc sampling theo từng window được kiểm định là khớp nhau (TV so với target: 0.0014 baseline / 0.0095 group — `outputs/sampling_audit/`).

Dữ liệu huấn luyện: kết hợp `libero_10` + `libero_90` (theo quy trình Long của bài báo), đã lọc các thao tác no-op, 4.900 demo huấn luyện / 100 demo val (giữ lại demo cuối cùng của mỗi file task làm val).
Metric benchmark: tỷ lệ thành công (success rate) trên bộ `libero_10`, 50 lượt thử/task, seed 7, chunk-execute 8 (`eval/libero_sim.py`).

## Chạy thử (Run)

Môi trường: conda env `vla_core` (`/home/tho2/miniconda3/envs/vla_core/bin/python`);
`transformers` phải tương thích với `Qwen3_5ForConditionalGeneration`.

```bash
bash scripts/train_libero_long.sh          # 40k step, batch 16 × accum 4 (eff. 64)
RESUME=runs/<run>/ckpt_last.pt STEPS=60000 bash scripts/train_libero_long.sh
python -m eval.libero_sim --ckpt runs/<run>/ckpt_best.pt --trials 50 --chunk-execute 8
pytest tests/
../plot_loss.sh runs/<run>                 # biểu đồ loss (script tại thư mục gốc workspace)
```

Mỗi lượt chạy sẽ ghi ra `runs/<name>/config.json`, `train.log`, `ckpt_last.pt` (điểm neo để resume), `ckpt_best.pt` (được chọn theo val trên libero_10).

## Trạng thái (Status)

Con số tham chiếu hiện tại đạt trung bình **16.4%** trên toàn bộ suite (`runs/libero_long_base_v3`, 60k step, ckpt_best, 50 lượt thử/task). Nhánh memory đang được huấn luyện lại do sự cố về learning-rate schedule; không thay đổi bất kỳ điều gì ở đây làm ảnh hưởng đến quá trình huấn luyện cho đến khi kết quả so sánh hoàn tất. Chi tiết: `docs/progress_report_2026-08-24.md`.
