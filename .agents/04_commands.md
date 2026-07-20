# Lệnh đã kiểm chứng

Chạy từ root `/home/dung/Workspace/VinRobotics`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[rlds,dev]'
source scripts/activate_vla_env.sh
```

`activate_vla_env.sh` phải được `source`, không chạy như executable. Extra `rlds` nặng vì có
TensorFlow/CUDA; chỉ cài khi cần RLDS. `requirements.txt` không phải lựa chọn nhẹ vì kéo cả
RLDS và test dependencies.

## Test và kiểm tra nhẹ

```bash
.venv/bin/pytest -q
.venv/bin/python -m compileall -q src scripts tests
bash -n scripts/activate_vla_env.sh scripts/download_vla_sample.sh
.venv/bin/python -m pip check
```

Baseline ngày 2026-07-16: `4 passed`. Repo chưa cấu hình lint/type check chính thức.

## CLI

```bash
.venv/bin/python -m vla_data_tools --help

.venv/bin/python -m vla_data_tools inspect \
  --format lerobot \
  --path dataset/lerobot_pusht \
  --max-episodes 2

.venv/bin/python -m vla_data_tools inspect \
  --format rlds \
  --path dataset/droid_200 \
  --max-episodes 1 \
  --decode-images false

.venv/bin/python -m vla_data_tools convert \
  --input-format lerobot \
  --input dataset/lerobot_pusht \
  --output-format hdf5 \
  --max-episodes 2
```

Output mặc định: `output/<input-name>/<input-name>.hdf5|parquet`.

`--max-episodes` giới hạn episode sinh ra, nhưng với LeRobot hiện chưa đảm bảo giảm toàn bộ
I/O/RAM metadata ban đầu. `--decode-images true` làm output và RAM tăng mạnh; mặc định false.

## Tải sample

```bash
bash scripts/download_vla_sample.sh --help
bash scripts/download_vla_sample.sh --only lerobot
bash scripts/download_vla_sample.sh --only droid200 --target-episodes 200
bash scripts/download_vla_sample.sh --only oxe200
```

Script tải vào `dataset/`, resume file `.part` theo size và ghi `sample_manifest.tsv`. Kiểm tra
disk bằng `du -sh dataset output` trước khi tải hoặc convert lớn.

## Các lệnh/path không dùng

- Không dùng `python -m tools.inspect` hoặc `--repo-id` trong roadmap cũ.
- `local_video_server.py` hiện ở `src/`, không phải `scripts/`; nó thuộc Caption QA helper,
  không phải VLA ingestion.
- Git commands hiện fail vì `.git/` rỗng. Không tự chạy `git init` để “sửa” nếu người dùng
  chưa yêu cầu.

## Kiểm tra hạ tầng Codex

```bash
codex --version
python3 .agents/scripts/01_validate_workspace.py
python3 .agents/scripts/01_validate_workspace.py --full
```

Bản mặc định chỉ kiểm cấu trúc, link, memory, skill và TOML. `--full` chạy thêm pytest,
compileall, shell syntax và CLI help. Project config ở `.codex/config.toml` chỉ được Codex
nạp khi workspace được trust; restart/new session sau khi đổi config hoặc custom agent.
