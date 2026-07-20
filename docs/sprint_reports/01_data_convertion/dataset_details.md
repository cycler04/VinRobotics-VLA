# Local VLA Dataset Details

These are robot-demonstration datasets, not ordinary video collections. Each
episode aligns camera observation(s), robot state, optional language, and the
action executed at every timestep. They are suitable for robot policy and VLA
training: `image + instruction + state -> action`.

## Source datasets

| Dataset | What it represents | Native format | Local sample observed |
| --- | --- | --- | --- |
| LeRobot PushT | Simulated task: push a T-shaped block to a target. Useful as a small, predictable format test. | LeRobot v3: frame Parquet, MP4 camera video, episode/task metadata. | One 96x96 RGB camera, 2D state, 2D action, 10 Hz, task text. |
| DROID | Real-robot manipulation demonstrations with synchronized external and wrist cameras. | RLDS/TFDS: `episode -> steps` in TFRecord shards. | Three 180x320 RGB cameras, 8D state, 7D action, 15 Hz. The inspected local subset has no language text. |
| OXE UTokyo PR2 | Manipulation demonstrations collected with a PR2 robot; one constituent dataset within OXE. | RLDS/TFDS in TFRecord shards. | One 128x128 RGB camera, 7D state, 8D action, task text. The converter records a 10 Hz assumption because this release does not declare a rate. |
| OXE ASU tabletop | A separate tabletop-manipulation constituent dataset within OXE. | RLDS/TFDS in TFRecord shards. | One 224x224 RGB camera, 7D state, 7D action, task text, 125 Hz. |

OXE (Open X-Embodiment) is a collection of datasets, not one consistent robot
format. Its state and action vectors must therefore remain dataset-specific.

## Converted representation

The converter writes one file per source dataset under `output/<dataset>/`.
HDF5 stores episode groups and numeric tensors; Parquet stores frame rows and
references to image assets when images are decoded. Both preserve episode
boundaries, timestamps, terminal flags, raw state/action values, camera
metadata, task text, and source/action metadata.

With `--decode-images false`, RGB remains referenced in the source MP4/TFRecord
data. With `--decode-images true`, Parquet image arrays are saved beside the
file as NumPy assets.

## Typical usage

- Inspect and validate a new robot-data source before training.
- Train behavior-cloning or VLA policies from observation, language, state, and
  action sequences.
- Compare or preprocess datasets through one canonical episode interface.

They are not equivalent to egocentric-video datasets such as EgoDex: video is
only one input modality here, while aligned robot actions are the learning
target.
