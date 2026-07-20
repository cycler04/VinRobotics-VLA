# Download, inspect, and convert the three VLA samples

This Sprint 1 implementation covers three representative sources:

```text
LeRobot pusht ───────────────┐
DROID partial RLDS ─────────┼─> canonical episode v0.1 ─> output/<dataset>/*
OXE UTokyo PR2 (RLDS) ──────┘
```

The converter preserves episode boundaries, language, raw state/action vectors,
terminal flags, and source metadata. Action/state semantics remain source-defined or
unknown for several datasets, and RGB may be stored only as a source reference. See
[the converter report](../dataset_converter_report.md) for the verified contract,
conversion losses, runtime evidence, and current training-readiness limits.

## Setup

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e '.[rlds,dev]'
source scripts/activate_vla_env.sh
```

The editable install is required because the Python package lives under
`src/vla_data_tools/`. It makes both `python -m vla_data_tools` and the
`vla-data-tools` console command available from any working directory.
On Linux, the `rlds` extra installs TensorFlow with pip-managed NVIDIA CUDA
libraries. TensorFlow still falls back to CPU automatically when no compatible
GPU is visible. The activation helper also adds those pip-managed CUDA library
directories to `LD_LIBRARY_PATH`, which is required by some TensorFlow wheels.

Check GPU discovery after installation:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

A detected device appears as `PhysicalDevice(... device_type='GPU')`. The RLDS
reader enables memory growth so TensorFlow does not reserve all GPU memory. File
reading, image decoding, and HDF5/Parquet writing remain primarily CPU and I/O
workloads, so GPU support does not imply a large conversion speedup.

For Hugging Face access, `.env` may contain:

```dotenv
HF_TOKEN=hf_your_token
```

The downloader reads this value without printing it. The token applies to the
LeRobot Hub download. DROID and OXE are downloaded anonymously from their
official public Google Cloud buckets.

## Download all three datasets

```bash
bash scripts/download_vla_sample.sh
```

The command is resumable and writes one manifest per dataset:

```text
dataset/
  lerobot_pusht/                                  # 206 episodes
  droid_200/                                      # 5 shards, 228 available; use 200
  oxe/utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds/ # 240 episodes
```

Download only one source when needed:

```bash
bash scripts/download_vla_sample.sh --only lerobot
bash scripts/download_vla_sample.sh --only droid200 --target-episodes 200
bash scripts/download_vla_sample.sh --only oxe200
```

## Inspect and validate

LeRobot:

```bash
python -m vla_data_tools inspect \
  --format lerobot \
  --path dataset/lerobot_pusht \
  --max-episodes 2
```

DROID RLDS:

```bash
python -m vla_data_tools inspect \
  --format rlds \
  --path dataset/droid_200 \
  --max-episodes 200 \
  --decode-images false
```

OXE RLDS:

```bash
python -m vla_data_tools inspect \
  --format rlds \
  --path dataset/oxe/utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds \
  --max-episodes 200 \
  --decode-images false
```

The JSON output reports episode and step counts, control rate, cameras, image
shape, state/action shape and dtype, instruction coverage, action convention,
and validation errors. Remove `--max-episodes` to inspect the complete sample.

These RLDS releases do not carry per-step timestamps. The adapter records that
timestamps were synthesized. Dataset profiles use DROID at 15 Hz and the legacy
ASU sample at 125 Hz. UTokyo PR2 does not declare a rate in its TFDS metadata,
so its conversion records a 10 Hz default assumption. Override it when a more
authoritative source rate is available:

```bash
python -m vla_data_tools inspect \
  --format rlds --path dataset/my_rlds \
  --control-rate-hz 10 --max-episodes 2
```

## Convert to HDF5

Converted files go to their own `output/<dataset>/` directory by default. For a
200-episode smoke conversion, keep RGB in the local TFRecords and write
references by using `--decode-images false`:

```bash
python -m vla_data_tools convert \
  --input-format rlds \
  --input dataset/droid_200 \
  --output-format hdf5 \
  --max-episodes 200 \
  --decode-images false
```

Convert 200 OXE UTokyo PR2 episodes:

```bash
python -m vla_data_tools convert \
  --input-format rlds \
  --input dataset/oxe/utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds \
  --output-format hdf5 \
  --max-episodes 200 \
  --decode-images false
```

Convert 200 LeRobot episodes:

```bash
python -m vla_data_tools convert \
  --input-format lerobot \
  --input dataset/lerobot_pusht \
  --output-format hdf5 \
  --max-episodes 200
```

The default outputs are:

```text
output/
  droid_200/droid_200.hdf5
  utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds/
    utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds.hdf5
  lerobot_pusht/lerobot_pusht.hdf5
```

The dataset directory name is used for both the output folder and filename.
Use `--output path/to/name.hdf5` only when a custom path is needed.

## Convert to Parquet and optional image assets

Decoded images are kept outside Parquet as per-episode NumPy assets, while
Parquet stores frame data and references to those files:

```bash
python -m vla_data_tools convert \
  --input-format rlds \
  --input dataset/droid_200 \
  --output-format parquet \
  --max-episodes 200 \
  --decode-images false
```

The result is:

```text
output/droid_200/droid_200.parquet
```

With `--decode-images false`, Parquet keeps TFRecord/video references. With
`--decode-images true`, RGB is decoded beside the Parquet file under
`output/<dataset>/<name>_assets/episode_*/*.npy`. LeRobot video remains a source
reference rather than being copied.

## Test

```bash
python -m pytest -q
```

The tests cover LeRobot v2/v3 reading, canonical validation, sample limits,
HDF5/Parquet writing, decoded image storage, task text, episode flags, and video
references. They do not currently exercise `RLDSReader` or round-trip loading. Real
conversion outputs are available under `output/`.
