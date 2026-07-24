# Lệnh

Chạy từ repository root trừ khi có ghi chú khác.

## Khảo sát an toàn

```bash
git status --short
git log -5 --oneline --decorate
git submodule status
python3 --version
```

Worktree có thể đang dở; `status` không sạch không phải lý do để reset.

## Môi trường tối thiểu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Dùng extra nặng chỉ khi cần RLDS:

```bash
python -m pip install -e '.[rlds,dev]'
source scripts/activate_vla_env.sh
```

`rlds` có TensorFlow/CUDA và có thể tải/cài lâu. Script activate chỉ hữu ích sau
khi môi trường đã tồn tại.

## Test và smoke check

```bash
python -m pytest -q
python -m vla_data_tools --help
```

Không install package thì dùng:

```bash
PYTHONPATH=src python -m vla_data_tools --help
```

Tại lần khảo sát 2026-07-24, máy chưa có `.venv`, `pytest`, `numpy`, `pyarrow`
hoặc `h5py`; vì vậy pytest và CLI import chưa chạy được. AST của 10 file Python
và cú pháp hai shell script đã qua kiểm tra.

Project gốc chưa cấu hình lint/type-check. Không hứa có `ruff`, `mypy` hay
pre-commit.

## CLI dữ liệu

```bash
python -m vla_data_tools inspect --help
python -m vla_data_tools convert --help
```

Luôn inspect/smoke test ít episode trước. Với LeRobot, `--max-episodes` không
ngăn bước đọc toàn bộ Parquet metadata/frame vào RAM.

Downloader:

```bash
bash scripts/download_vla_sample.sh --help
```

Script hard-code `.venv/bin/python`; tạo `.venv` trước. Download có thể lớn, nên
chọn một source nhỏ và đo disk trước.

## Submodule Kinetix

Đọc README trong submodule trước. Môi trường dùng `uv sync` và các lệnh dạng
`uv run src/train_expert.py`. Workload mặc định lớn; cấu hình distributed yêu
cầu số GPU phù hợp với 12 level. Không chạy mặc định như smoke test.

## Hạ tầng agent

```bash
python3 .agents/scripts/01_validate_workspace.py --full
```

Lệnh này kiểm tra numbering, link/path nội bộ, TOML/YAML, skill và gitignore.

