# Cách chạy, trạng thái kiểm chứng và khoảng trống

## Lệnh được code mô tả

README yêu cầu đưa cả source của `data_corpus` và thư mục `vla_core` vào Python path:

```bash
cd third_party/02_vla_core
export PYTHONPATH=/path/to/data_corpus/src:$PWD
```

Sanity run được đề xuất:

```bash
python -m train.pretrain \
  --releases-json configs/releases.json \
  --steps 50 \
  --overfit 8 \
  --batch 2 \
  --log-every 1
```

Run dài được README mô tả:

```bash
python -m train.pretrain \
  --releases-json configs/releases.json \
  --steps 100000 \
  --batch 8 \
  --accum 4
```

Các lệnh trên **chưa chạy thành công trong workspace**. Snapshot không có dependency manifest,
package `data_corpus`, dataset release hoặc model weights.

## Kết quả kiểm tra ngày 2026-07-24

Chạy từ root workspace:

```bash
python3 -m compileall -q third_party/02_vla_core
```

Kết quả: pass, không có syntax error.

```bash
python3 -m pytest -q third_party/02_vla_core/tests
```

Kết quả: dừng trong test collection, cả hai module lỗi `ModuleNotFoundError: No module named
'torch'`. Chạy lại bằng `.venv/bin/python` cho cùng kết quả.

Probe import:

| Dependency/runtime | Python hệ thống | Workspace `.venv` |
| --- | --- | --- |
| `torch` | thiếu | thiếu |
| `transformers` | thiếu | thiếu |
| `cv2` | thiếu | thiếu |
| `corpus` | thiếu | thiếu |
| Pillow | `12.1.1` | thiếu |
| FFmpeg binary | `8.0.1` | dùng chung hệ thống |

Đây là trạng thái môi trường hiện tại, không phải bằng chứng code sẽ fail sau khi cài đúng
dependency.

## Test coverage có trong snapshot

`tests/test_action_head.py` kiểm:

- forward có proprio và padding mask;
- forward không có proprio;
- narrative padding không làm rò garbage vào output;
- gradient tồn tại trong action head.

`tests/test_utils_extract.py` kiểm:

- shape vision/narrative stream với batch size 2;
- batch padding bị loại khỏi narrative stream;
- backward-compatible path khi không có attention mask.

Không có test cho:

- `pack_actions()`, rotation conversion và element mask;
- decode frame hoặc adapter `Layer1PretrainSampler`;
- tokenizer/chat-template label boundary;
- collator với image/text length khác nhau;
- `VLAModel.forward()` và flow loss end-to-end;
- `predict_action()` với batch padding/EOS;
- checkpoint save/load;
- train CLI, overfit collapse, DDP;
- validation/evaluation.

## Những claim chưa đủ bằng chứng

### Multi-GPU DDP

`pretrain.py` tự mô tả “single/multi-GPU via torchrun DDP”, nhưng không có bất kỳ thành phần
DDP nào:

- không `init_process_group`;
- không `LOCAL_RANK`;
- không `DistributedDataParallel`;
- không `DistributedSampler`;
- mặc định `--device cuda:0`.

**Verified:** implementation hiện chỉ là single-process training loop. Chạy nhiều process
bằng `torchrun` có nguy cơ mỗi process dùng cùng GPU và ghi cùng checkpoint.

### Dataset scale và split

README nói train dataset có `4.48M windows` và validation group-disjoint qua `part="val"`.
Snapshot chỉ truyền `part` vào sampler ngoài; không có metadata hoặc command output để tái
kiểm hai claim này.

### License

README nói `xp10m` có license non-commercial và constraint lan sang weight được train.
Snapshot không kèm license file, dataset card hay source URL. Cần xác nhận trực tiếp từ
license của release trước khi dùng hoặc phân phối model.

### Hai lỗi batch inference tiềm ẩn

`predict_action()` thay attention mask sau generation bằng tensor toàn `1`. Với batch có độ
dài prompt hoặc generation khác nhau, token pad có thể bị xem là narrative token hợp lệ khi
re-encode. Comment trong code giả định generated sequence không có padding, nhưng điều này
chưa có test với `B > 1`.

Ngoài ra, `pad_token_id = config.pad_token_id or config.eos_token_id` coi ID `0` là false.
Nếu tokenizer hợp lệ dùng pad ID `0`, code sẽ thay nó bằng EOS ID. Nên kiểm `is None` thay vì
dùng truthiness.

## Bảng trạng thái năng lực

| Năng lực | Trạng thái | Bằng chứng/giới hạn |
| --- | --- | --- |
| Action packing `16 × 153` | Implemented, static-verified | Code rõ; thiếu upstream semantics/normalization |
| Qwen processor | Implemented, chưa chạy | Thiếu `transformers` và model artifact |
| Hidden-state split | Implemented, có unit test source | Test chưa chạy vì thiếu `torch` |
| Flow action head | Implemented, có unit test source | Chưa có full-model smoke test |
| Proprio conditioning | Implemented trong model | Tắt trong run-1, không có data path |
| Single-process pretraining | Implemented ở mức code | Chưa chạy end-to-end |
| Multi-GPU DDP | Chưa implement | Docstring vượt quá code |
| Inference sampling | Implemented ở mức tensor | Không có CLI/checkpoint loader/denormalization |
| Evaluation | Chưa implement | `eval/` rỗng |
| Robot execution | Ngoài snapshot | Không có runtime/controller/safety layer |

## Thứ tự kiểm chứng được đề xuất

1. Pin Python, PyTorch, Transformers, OpenCV, FFmpeg và revision Qwen trong một
   `pyproject.toml`/lockfile.
2. Gắn đúng revision `data_corpus`; đưa `ACTION_SPEC.md`, release schema và license vào
   provenance có thể truy lại.
3. Chạy unit test hiện có trên CPU.
4. Thêm test `pack_actions()` và processor/collator với batch có độ dài khác nhau.
5. Chạy `--overfit 8` và lưu loss curve, peak VRAM/RAM, sample/s cùng exact config.
6. Kiểm gradient theo component để xác nhận narrative loss không train backbone khi frozen.
7. Thêm validation split, checkpoint resume và một inference harness load checkpoint.
8. Chỉ sau các bước trên mới triển khai DDP hoặc run toàn corpus.

## Rủi ro vận hành

- `configs/releases.json` chứa absolute path riêng của máy nguồn; phải tạo config mới thay vì
  giả định các path này tồn tại.
- Mỗi sample spawn một FFmpeg process và decode frame; nên benchmark loader trước khi tăng
  worker/GPU count.
- Checkpoint chứa toàn bộ `model.state_dict()`, gồm cả frozen Qwen; file có thể lớn dù phần
  trainable chỉ nằm ở action head.
- Checkpoint không có optimizer/RNG state, nên không resume chính xác.
- `--overfit N` cắt `ds.index` sau khi dataset đã enumerate toàn bộ window; nó không giảm chi
  phí scan metadata ban đầu của sampler ngoài.
