# Download sample VLA datasets for ingestion and conversion

[100.89.98.89:7861/api/videos/download-zip?batch=0&amp;size=500](http://100.89.98.89:7861/api/videos/download-zip?batch=0&size=500)

## Purpose

This guide prepares **small, practical samples** from five selected VLA data ecosystems:

1. LeRobot Hub
2. DROID
3. Open X-Embodiment (OXE)
4. AgiBot World
5. RoboMIND

The goal is to inspect and convert data, not to mirror multi-terabyte corpora. Keep every downloaded source immutable and write converted outputs to a separate directory.

```text
~/Dataset/VLA/
  raw/
    lerobot_pusht/
    droid_100/
    oxe/<dataset_name>/
    agibot/<task_id>/
    robomind/<benchmark>/<task_name>/
  converted/
    internal_hdf5/
    internal_parquet/
    lerobot_v2/
```

## Before starting

### Disk target

For this sample-only guide, reserve at least **30 GB free**. DROID 100 episodes is about 2 GB; extraction and conversion create additional temporary/output copies.

Do **not** download full OXE, full DROID, AgiBot Alpha/Beta, or full RoboMIND during parser development.

### Common tools

```bash
sudo apt update
sudo apt install -y git git-lfs ffmpeg tar

python3 -m venv ~/venvs/vla-data
source ~/venvs/vla-data/bin/activate
python -m pip install -U pip huggingface_hub
```

Use a separate terminal session or `tmux` for large downloads.

## Recommended order

| Order | Dataset      | Sample target                               | Why                                                           |
| ----: | ------------ | ------------------------------------------- | ------------------------------------------------------------- |
|     1 | LeRobot      | `lerobot/pusht`, about 8 MB               | Verify environment and inspect a LeRobot dataset immediately. |
|     2 | DROID        | `droid_100`, 100 episodes / about 2 GB    | First real RLDS sample; good for RLDS reader and validator.   |
|     3 | OXE          | 5–10 episodes from one constituent dataset | Learn variation in OXE RLDS schema without TB-scale download. |
|     4 | AgiBot World | One task directory                          | Practice task-scoped download and a non-RLDS source.          |
|     5 | RoboMIND     | One task archive in one embodiment          | Practice HDF5/tar-part ingestion and extraction.              |

## 1. LeRobot Hub sample

LeRobot is an ecosystem and format, not one dataset. Start with `lerobot/pusht`; it is a small simulation dataset, useful for testing the LeRobot reader before working with real robot data.

```bash
mkdir -p ~/Dataset/VLA/raw/lerobot_pusht

huggingface-cli download \
  lerobot/pusht \
  --repo-type dataset \
  --local-dir ~/Dataset/VLA/raw/lerobot_pusht
```

Check downloaded size:

```bash
du -sh ~/Dataset/VLA/raw/lerobot_pusht
```

Optional Python check after installing LeRobot:

```bash
pip install lerobot
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/pusht")
print(dataset.meta)
print(dataset[0].keys())
```

Expected learning outcome: identify Parquet/video-or-image assets, episode metadata, frame index, action tensor, observation state, and task fields.

## 2. DROID RLDS sample

DROID provides an official `droid_100` debugging subset: 100 RLDS episodes at about 2 GB. Use this rather than the 1.7 TB full RLDS release.

Install Google Cloud CLI/`gsutil` following the official Google Cloud installation instructions, then confirm:

```bash
gsutil version
```

Download:

```bash
mkdir -p ~/Dataset/VLA/raw

gsutil -m cp -r \
  gs://gresearch/robotics/droid_100 \
  ~/Dataset/VLA/raw/
```

Verify:

```bash
du -sh ~/Dataset/VLA/raw/droid_100
find ~/Dataset/VLA/raw/droid_100 -type f | head
```

Full-release reference only; do not run these in the sample phase:

```text
gs://gresearch/robotics/droid      # RLDS full: 1.7 TB
gs://gresearch/robotics/droid_raw  # raw stereo/HD: 8.7 TB
```

Expected learning outcome: parse RLDS `episode -> steps`, inspect multi-camera observations, state/action arrays, language annotation, timestamps, and terminal flags.

## 3. Open X-Embodiment sample

OXE is a collection of constituent datasets stored in RLDS/TFDS form. It is not one single archive. Download only one constituent dataset and then restrict parsing to 5–10 episodes.

Install read dependencies:

```bash
pip install tensorflow tensorflow-datasets
```

List available public dataset prefixes:

```bash
gsutil ls gs://gdm-robotics-open-x-embodiment/
```

Choose a dataset name from that listing. Example with `fractal20220817_data`:

```bash
mkdir -p ~/Dataset/VLA/raw/oxe

gsutil -m cp -r \
  gs://gdm-robotics-open-x-embodiment/fractal20220817_data \
  ~/Dataset/VLA/raw/oxe/
```

Verify:

```bash
du -sh ~/Dataset/VLA/raw/oxe/fractal20220817_data
find ~/Dataset/VLA/raw/oxe/fractal20220817_data -type f | head
```

Load with TFDS by setting the directory to the parent that contains the downloaded builder data:

```python
import os
import tensorflow_datasets as tfds

ds = tfds.load(
    "fractal20220817_data",
    data_dir=os.path.expanduser("~/Dataset/VLA/raw/oxe"),
    split="train",
)

for episode in ds.take(1):
    print(episode.keys())
```

If this particular prefix is unavailable, choose another visible prefix from `gsutil ls`. Do not assume every historical OXE constituent remains accessible in the same bucket state.

Expected learning outcome: compare this RLDS schema against DROID; preserve raw action semantics and `action_spec` rather than assuming identical action vectors mean identical controls.

## 4. AgiBot World: one-task sample

AgiBot World Alpha is about 8.5 TB and Beta about 43.8 TB. Download only one task. Some files may require Hugging Face authentication.

```bash
huggingface-cli login
```

Replace `327` with a valid task id from the AgiBot task catalog. The following is a task-scoped download template:

```bash
TASK_ID=327
DEST=~/Dataset/VLA/raw/agibot/task_${TASK_ID}

huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  agibot-world/AgiBotWorld-Alpha \
  --local-dir "$DEST" \
  --include "observations/${TASK_ID}/**" \
  --include "task_info/task_${TASK_ID}.json" \
  --include "parameters/**" \
  --include "proprio_stats/**"
```

Verify the task metadata before attempting conversion:

```bash
du -sh "$DEST"
find "$DEST" -maxdepth 3 -type f | head -30
```

Keep the task JSON, robot parameters and proprioception statistics together with its observation files; they are needed to correctly interpret action/state conventions.

Expected learning outcome: build an adapter from a task-organized raw source into the canonical episode schema, then optionally write LeRobot v2 output.

## 5. RoboMIND: one HDF5 task archive

RoboMIND is about 12.3 TB overall and is organized into benchmark/embodiment directories. It can be gated on Hugging Face: accept access conditions on its dataset page and use a Hugging Face token before downloading.

First inspect the repository tree in the browser and choose exactly one task under one embodiment, for example a directory below:

```text
benchmark1_0_compressed/h5_ur_1rgb/<task-name>/
```

Download only that task. Replace `<task-name>` with the actual directory name copied from the dataset tree:

```bash
BENCHMARK=benchmark1_0_compressed
EMBODIMENT=h5_ur_1rgb
TASK_NAME=<task-name>
DEST=~/Dataset/VLA/raw/robomind/${BENCHMARK}/${EMBODIMENT}/${TASK_NAME}

huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  x-humanoid-robomind/RoboMIND \
  --local-dir "$DEST" \
  --include "${BENCHMARK}/${EMBODIMENT}/${TASK_NAME}/**"
```

RoboMIND task data may be split into files such as `task.tar.gz.part-aa`, `part-ab`, and so on. Only when **all** parts for one archive are present:

```bash
cd "$DEST"
cat task.tar.gz.part-* > task.tar.gz
tar -xzf task.tar.gz
```

Replace `task` above with the actual common archive basename. Verify extracted content:

```bash
find "$DEST" -type f \( -name '*.h5' -o -name '*.hdf5' \) | head
du -sh "$DEST"
```

Expected learning outcome: implement HDF5 inspection, identify image/state/action/language datasets and map them into the canonical episode schema. Keep the original HDF5 unchanged.

## After every download: record a manifest

Create one manifest per sample to make conversion reproducible:

```bash
cd ~/Dataset/VLA/raw
find lerobot_pusht droid_100 oxe agibot robomind -type f -printf '%p\t%s\n' \
  > sample_manifest.tsv
```

For each source, record in your project notes:

- dataset/repository name and revision/date;
- exact downloaded path;
- task id or OXE constituent name;
- number of files and total bytes;
- episode count parsed;
- state/action dimensions and dtypes;
- camera names/resolutions;
- timestamp/frequency behavior;
- action coordinate frame, units, absolute-versus-delta convention;
- license/access condition.

## Next action after downloads

Run the same inspector over every sample and output:

```text
episodes, steps per episode, camera keys, image shape,
state/action shape + dtype, instruction coverage,
timestamp monotonicity, terminal flags, NaN/Inf count
```

Convert through the canonical internal schema only:

```text
RLDS / LeRobot / HDF5 source
  -> canonical episode
  -> internal HDF5 or Parquet + video
  -> LeRobot v2
```

Do not implement direct pairwise `RLDS <-> LeRobot` converters. During every conversion, preserve episode boundaries, raw actions, action/state semantics, timestamps, camera mapping, calibration and source metadata.

## Official references

- [DROID dataset download documentation](https://droid-dataset.github.io/droid/the-droid-dataset)
- [Open X-Embodiment repository and manual TFDS download](https://github.com/google-deepmind/open_x_embodiment)
- [LeRobot dataset documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [AgiBot World repository](https://github.com/OpenDriveLab/AgiBot-World)
- [RoboMIND on Hugging Face](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)
