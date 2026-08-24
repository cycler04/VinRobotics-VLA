# vla_core — Model VLA kết hợp với bộ nhớ nhận thức-tư duy (perceptual-cognitive memory)

Model Vision-Language-Action được xây dựng trên VLM **Qwen3.5-0.8B** đóng băng (frozen), đi kèm với chuyên gia hành động (action expert) dạng flow-matching và bộ ngân hàng bộ nhớ **PCMB theo kiểu MemoryVLA (tùy chọn)**.
Có hai hướng huấn luyện được triển khai ở đây:

1. **Run-1 corpus pretraining** — Không gian hành động con người-góc nhìn thứ nhất (human-ego) 153 chiều trên tập dữ liệu Layer-1 (repo anh em `data_corpus`; file `ACTION_SPEC.md` tại đó là hợp đồng nhãn / label contract).
2. **LIBERO-Long memory benchmark** — Hành động delta-EEF 7 chiều trên các demo `libero_10 + libero_90`, được đánh giá bằng tỷ lệ thành công của bộ mô phỏng trên bộ bài test `libero_10`. Nhánh điều khiển không dùng memory nằm ở repo anh em `vla_core-benchmark-libero-base/` (xem `docs/benchmark_protocol.md` tại đó).

## Model

```
Qwen3.5-0.8B (VLM đóng băng / frozen)
  └─ chia hidden-state theo từng lớp → luồng vision + luồng narrative
       └─ ActionHead: 24 khối cross-attention, flow matching (velocity MSE),
          vị trí RoPE, các action chunk 16 bước            (~229M thông số)
       └─ MemoryModule + MemoryBank (tùy chọn, --memory)   (~61M thông số)
       └─ ProprioEncoder (tùy chọn, proprio_dim)
Dual loss: flow-MSE (chuyên gia hành động) + weighted narrative LM (backbone/LoRA).
detach_action_input=True giúp giữ hai hàm loss này trên các tham số tách biệt (disjoint).
```

Tổng kiểm kê: 1.154B thông số, 555M thông số có thể huấn luyện (trainable) dưới cấu hình mặc định của corpus; cấu hình LIBERO (đóng băng hoàn toàn backbone, tắt narrative) sử dụng ~290M (head + memory).
Chi tiết: `docs/memory_module/model_parameters.md`.

### Ý tưởng về bộ nhớ (The memory idea)

Model cơ sở là loại đơn khung hình (single-frame) — nó không thể phân biệt các trạng thái giống hệt nhau về mặt thị giác trước và sau một hành động. Chúng tôi đưa Ngân hàng Bộ nhớ Nhận thức-Tư duy (Perceptual-Cognitive Memory Bank - PCMB) của MemoryVLA (ICLR 2026) vào giai đoạn *pretraining*, chứ không phải một giải pháp bổ sung sau khi finetune:

- Một `MemoryBank` cho mỗi vị trí batch (batch slot); các đặc trưng hợp nhất (sau khi retrieval + gate) được push và mặc định là detach; hợp nhất gộp lân cận (adjacent-merge consolidation) khi đạt sức chứa tối đa (L=16).
- Ngân hàng bộ nhớ duy nhất từ lớp ẩn cuối cùng (last hidden layer), được broadcast đến tất cả 24 khối head thông qua một luồng bộ nhớ có cổng (gated memory stream) chuyên dụng (các gate bắt đầu từ 0 và không bao giờ bị weight-decay).
- **Group mode** (`EpisodeGroupBatchSampler`): batch = K group × G=16 window sắp xếp theo thời gian của một clip; bank được xây dựng đệ quy (recurrent) trong phạm vi group (bám sát bài báo gốc).
- **Continuous mode** (`--continuous`): các bank duy trì liên tục qua các batch theo từng stream slot — đây là điểm so sánh mới; MemoryVLA+ trước đây chỉ hỗ trợ group mode.
- Các bước thời gian (timestep) là chỉ số tuyệt đối trong clip (`t0`), áp dụng tương tự cho cả train và eval.

Tài liệu thiết kế: `docs/memory_module/memoryvla_integration.md`, `docs/memory_module/data_sampling/README.md`.

## Cấu trúc thư mục (Layout)

```
model/    vla_model.py · action_head.py · memory.py · proprio_encoder.py · config.py
data/     processing.py (đóng gói Qwen chat/vision) · corpus_dataset.py ·
          libero_dataset.py · collate.py · norm.py · hier.py
train/    pretrain.py (GPU đơn) · pretrain_ddp.py (torchrun) · common.py
eval/     libero_sim.py (triển khai đánh giá benchmark) · libero_bridge.py · offline.py
scripts/  train_libero_long.sh · train_libero_long_supervised.sh · smoke_*.sh
configs/  releases.json · action_norm_libero.json · action_norm_human_ego.json
docs/     memory_module/ (integration, sampling, runs, budget) · walkthrough/
tests/    bộ test pytest (action head, memory, samplers, datasets, norm, guards)
```

## Chạy thử (Run)

Môi trường: conda env `vla_core` (`/home/tho2/miniconda3/envs/vla_core/bin/python`).
Cố định phiên bản `transformers` có hỗ trợ `Qwen3_5ForConditionalGeneration` (5.x).

**Huấn luyện trước trên Corpus** (cần repo `data_corpus` trong `PYTHONPATH`):

```bash
export PYTHONPATH=/path/to/data_corpus/src:$PWD
python -m train.pretrain --releases-json configs/releases.json \
    --steps 100000 --batch 8 --accum 4              # lượt chạy thật
python -m train.pretrain --releases-json configs/releases.json \
    --steps 50 --overfit 8 --batch 2 --log-every 1  # kiểm tra độ ổn định: loss phải hội tụ/sụt giảm mạnh
```

**Huấn luyện memory LIBERO-Long** (40k step, effective batch 64):

```bash
bash scripts/train_libero_long.sh                  # group mode, GPU đơn (~46 giờ)
DDP=1 bash scripts/train_libero_long.sh            # torchrun 2-GPU (~24 giờ)
CONTINUOUS=1 bash scripts/train_libero_long.sh     # stream mode, duy trì các bank
RESUME=runs/<run>/ckpt_last.pt bash scripts/train_libero_long.sh
# wrapper tự khởi động lại trên máy dùng chung (sống sót qua SIGKILL từ tiến trình khác):
nohup bash scripts/train_libero_long_supervised.sh & disown
```

**Đánh giá Benchmark** (bộ LIBERO-Long, quy trình bài báo với seed 7):

```bash
python -m eval.libero_sim --ckpt runs/<run>/ckpt_best.pt --trials 50 --chunk-execute 8
```

**Test / Chẩn đoán**:

```bash
pytest tests/                       # bộ test unit + regression
bash scripts/smoke_single.sh        # smoke test trên GPU
../plot_loss.sh runs/<run>          # biểu đồ loss (script tại thư mục gốc workspace)
```

## Trạng thái (Status)

Nhánh baseline (không dùng memory) đạt **16.4%** trên LIBERO-Long; model dùng memory hiện ở mức **3.4%** — bị nghẽn do lỗi cấu hình learning-rate schedule, cần huấn luyện lại trước khi kết quả so sánh có hiệu lực. Báo cáo đầy đủ: `docs/progress_report_2026-08-24.md`.

## Ghi chú (Notes)

- Các video egodex dùng định dạng AV1 → giải mã khung hình thông qua tiến trình con ffmpeg (nâng cấp lên PyAV là một tùy chỉnh để tăng hiệu năng).
- Tập dữ liệu Corpus = 4.48M window (tập train, phân tách val không trùng group qua `part="val"`).
- Giấy phép xp10m là KHÔNG THƯƠNG MẠI (NON-COMMERCIAL) — áp dụng cho các trọng số được huấn luyện trên đó.
- Các bản sửa lỗi trích xuất lịch sử (lỗi loss h_a, rò rỉ hidden-state padding, biến đổi config) đều được bảo vệ bởi các test regression trong `tests/`.
