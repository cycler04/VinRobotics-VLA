# scripts/

Runnable entry points for VLA experiments. Everything here is a plain
`python scripts/<name>.py` CLI — no package install, no framework, run from the
repository root.

> **Rule for agents and humans: adding a script means adding its section to this
> file in the same change.** A script with no section here does not exist as far
> as the next person is concerned. Keep the table below and the per-script
> section in sync, and delete the section when you delete the script.

## Scripts

| Script                                                          | Reads                                    | Writes                       | Needs GPU |
| --------------------------------------------------------------- | ---------------------------------------- | ---------------------------- | --------- |
| [`prepare_sample_dataset.py`](prepare_sample_dataset.py)       | `/mnt/SSD4` v06 releases + data_corpus | `dataset/<name>/`          | no        |
| [`run_vla_inference.py`](run_vla_inference.py)                 | `dataset/<name>/`                      | `output/<tag>/`            | yes       |
| [`run_egoverse_success_eval.py`](run_egoverse_success_eval.py) | `dataset/<name>/`                      | `output/<tag>/`            | yes       |
| [`pull_from_tho2.sh`](pull_from_tho2.sh)                       | remote`src/`, `scripts/`             | local`src/`, `scripts/`  | no        |
| [`push_to_tho2.sh`](push_to_tho2.sh)                           | local`src/`, `scripts/`              | remote`src/`, `scripts/` | no        |
| [`sync_docs_with_tho2.sh`](sync_docs_with_tho2.sh)             | local or remote`docs/`                 | remote or local`docs/`     | no        |

`dataset/` and `output/` are gitignored (`.gitignore` lines 38-39). Nothing here
writes inside `third_party/`.

## Workspace sync

Run from the repository root. Every command previews changes by default; add
`--apply` only after checking the preview. These scripts do not delete files.

```bash
# Code: server -> local
bash scripts/pull_from_tho2.sh
bash scripts/pull_from_tho2.sh --apply

# Code: local -> server
bash scripts/push_to_tho2.sh
bash scripts/push_to_tho2.sh --apply

# Docs: server -> local
bash scripts/sync_docs_with_tho2.sh pull
bash scripts/sync_docs_with_tho2.sh pull --apply

# Docs: local -> server
bash scripts/sync_docs_with_tho2.sh push
bash scripts/sync_docs_with_tho2.sh push --apply
```

All three scripts connect through the `vinrobotics` SSH alias, which resolves
to user `tho2`, and use `/home/tho2/Dung_Workspace/VinRobotics` as the remote
workspace. Code sync is limited to `src/` and `scripts/`; docs sync is limited
to `docs/`.

## Prerequisites

Conda `base` (Python 3.13) already has what these need:

```
torch 2.13.0+cu130 · torchvision 0.28.0+cu130 · transformers 5.14.1
opencv-python-headless · ffmpeg (system) · 2x RTX 5090
```

Backbone weights, HF safetensors format — the `models/*.gguf` files are
Q4-quantized text-only llama.cpp builds with no vision tower and **cannot** be
used:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-0.8B', local_dir='models/qwen3.5-0.8b-hf')"
```

`prepare_sample_dataset.py` additionally needs the corpus loader and the raw
releases; both paths are constants at the top of that script:

```
/mnt/SSD3/code/VinRobotics/data_corpus/src        Layer1PretrainSampler
/mnt/SSD4/dataset/releases/{egodex,egoverse,xp10m}_v06   (per configs/releases.json)
```

The inference scripts need neither — that is the point of the split.

---

## `prepare_sample_dataset.py`

Pulls a small self-contained sample out of a Layer-1 v06 release into
`dataset/`. Decodes one head-camera frame per window (ffmpeg, AV1/HEVC-safe) and
packs the 153-dim action chunk per `ACTION_SPEC`, then writes plain PNG + NPZ so
the inference scripts never touch the 245 GB release or the corpus loader again.

```bash
# egodex, held-out split
python scripts/prepare_sample_dataset.py --n 18 --part val

# egoverse task windows (see split caveat below)
python scripts/prepare_sample_dataset.py --sources egoverse --part train --n 24 \
    --out dataset/egoverse_sample
```

Output layout:

```
dataset/<name>/index.json          metadata + prompt + task_name + narrative segments
                                   + timebase, per-window fps/stride/anchor/step times
dataset/<name>/frames/<key>.png    head camera frame
dataset/<name>/actions/<key>.npz   actions (16,153) + mask (16,153), raw metric units
                                   step_frames (16,)  absolute source-video frame per step
                                   step_seconds (16,) that frame's time in the clip, s
                                   step_offsets (16,) seconds after the anchor (k/10 Hz)
```

**Action timestamps.** Every predicted step is tied to a real instant of the clip
the observation came from: `step_frames[k]` is the absolute frame the step
targets (in v06 the `frame` column *is* the decode index, so it indexes
`index.json`'s `video` directly), and `step_seconds[k] = step_frames[k] / fps`.
The anchor frame — the one the input image was decoded from — is `anchor_frame` /
`anchor_seconds`. A 16-step chunk at 10 Hz spans 1.6 s, so `step_offsets` runs
0.1 … 1.6 regardless of the clip's native fps (`stride = fps / action_hz`).

Key flags: `--sources` (comma-separated release keys), `--part train|val|all`,
`--n`, `--exclude-tasks` (default `debug`), `--n-steps`, `--action-hz`, `--seed`,
`--out`.

Windows are drawn round-robin over `task_name` so one task cannot dominate a
small sample. Clips whose video fails to decode are skipped and logged.

**Split caveats, both real:**

- egodex `val` contains exactly 3 task groups (`tie_untie_rubberband`,
  `push_pop_toy`, `braid_unbraid`) — the split is group-disjoint by task.
- egoverse `val` is **100% `debug` clips** (617/617). All 71 real tasks
  (7,237 clips) live in `train`, so egoverse currently gives no task-level
  validation signal at all. Use `--part train` and treat the result accordingly.

## `run_vla_inference.py`

Runs vla_core inference over a prepared sample: generates the narrative, then
predicts the action chunk via the flow-matching head. Writes predictions and
timings. No metric interpretation — use the eval script for that.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_vla_inference.py --tag smoke
CUDA_VISIBLE_DEVICES=0 python scripts/run_vla_inference.py \
    --dataset dataset/egoverse_sample --limit 4 --tag quick
```

Writes `output/<tag>/predictions.npz` (predictions, actions_gt, action_mask,
`step_frames` and `step_seconds` both `(N,16)`, keys) and
`output/<tag>/summary.json` (per-sample narrative, masked MAE, seconds, plus the
timestamps below).

Key flags: `--dataset`, `--model`, `--ckpt` (ActionHead state_dict; random init
without it), `--out`, `--tag`, `--limit`, `--device`, `--dtype`.

## `run_egoverse_success_eval.py`

Same inference pass, plus per-task success scoring.

**egoverse ships no success or reward label** — it is offline human ego video
with no environment to roll out in. Success here is an explicitly defined
open-loop trajectory-match proxy per 16-step window, computed only on
`valid`-masked components:

```
success = head_pos_err ≤ 5cm  ∧  head_rot_err ≤ 10°
        ∧ hand_pos_err ≤ 5cm  ∧  hand_rot_err ≤ 15°
        ∧ kp_err       ≤ 3cm
```

Rotations are compared as geodesic angle after 6D → R Gram-Schmidt. A window
with no valid hand yields `nan` and is counted as a failure, never a silent
pass. Narrative quality is scored separately as token-F1 against the window's
non-joystick ground-truth segments. This is a proxy, not task completion — real
success needs a simulator, a robot, or a trained success classifier.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_egoverse_success_eval.py --tag egoverse_eval
python scripts/run_egoverse_success_eval.py --selftest     # metric math, no GPU
```

Writes `output/<tag>/success.json` (per-window records, per-task aggregation,
tolerances, overall rates, plus the timestamps below) and
`output/<tag>/predictions.npz` (same arrays as above, including `step_frames`
and `step_seconds`).

Key flags: `--ckpt`, `--action-norm`, `--head-pos-tol`, `--head-rot-tol`,
`--hand-pos-tol`, `--hand-rot-tol`, `--kp-tol`, `--narrative-f1`, plus the
common `--dataset/--model/--out/--tag/--limit/--device/--dtype`.

`--selftest` asserts the metric math end to end: identity match, 10 cm offset,
90° rotation, one hand missing, all hands missing, F1 bounds.

**Numbers are not comparable until two things exist.** `success.json` records
this as `metrics_comparable: false`:

1. A trained ActionHead checkpoint (`--ckpt`). Without one the head is
   random-init, and mean head rotation error lands at ~126° — the expected
   geodesic distance between two uniformly random rotations.
2. `configs/action_norm_human_ego.json` (`python -m data.compute_action_stats`
   in vla_core). Per ACTION_SPEC the head works in q01/q99-normalized space
   while the corpus stores raw metres; `--action-norm` de-normalizes predictions
   back to metres so the tolerances above mean what they say.

---

## Timestamps in output

Both runners carry the dataset timebase into their results, so a prediction is
never a bare array — it says which clip, which frames, and when it was produced.

Per window, in `summary.json` / `success.json` `results[]`:

| field                                 | meaning                                                    |
| ------------------------------------- | ---------------------------------------------------------- |
| `video`                             | absolute path of the source clip                           |
| `fps`                               | that clip's native rate (ffprobe-verified per clip in v06) |
| `anchor_frame` / `anchor_seconds` | the frame the input image was decoded from                 |
| `step_frames` (16)                  | absolute source-video frame each predicted step targets    |
| `step_seconds` (16)                 | that frame's time in the clip                              |
| `predicted_at`                      | wall-clock ISO-8601 of that window's forward pass          |

Per run, at the top of the same file: `started_at`, `finished_at`, `command`,
`dataset_generated_at` (when `prepare_sample_dataset.py` built the input) and
`timebase` (how to read the frame numbers).

```
anchor_frame 13219 @ 440.633s  ->  step_frames [13222, 13225, ... 13267]
                                   step_seconds [440.733, 440.833, ... 442.233]
```

Overlaying a predicted chunk on the source video is then
`ffmpeg -i <video> -vf select=eq(n\,<step_frames[k]>)`.

Datasets built before this existed have no `step_frames` in their NPZ; both
runners stop with a message telling you to re-run `prepare_sample_dataset.py`
rather than silently emitting untimed predictions.

---

## Full walkthrough

```bash
# 1. weights (once)
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-0.8B', local_dir='models/qwen3.5-0.8b-hf')"

# 2. local samples (once per split you care about)
python scripts/prepare_sample_dataset.py --n 18 --part val
python scripts/prepare_sample_dataset.py --sources egoverse --part train --n 24 \
    --out dataset/egoverse_sample

# 3. run
CUDA_VISIBLE_DEVICES=0 python scripts/run_vla_inference.py --tag smoke
CUDA_VISIBLE_DEVICES=0 python scripts/run_egoverse_success_eval.py --tag egoverse_eval
```

## Known gaps

- No ActionHead checkpoint exists anywhere on this host. Every action number
  produced today is noise; only shapes, plumbing and narratives are meaningful.
- `configs/action_norm_human_ego.json` has not been generated.
- The prompt built from corpus text (`"Task context: ...\nLocomotion: ..."`)
  reads to Qwen as a question — it answers `"Yes, the robot is stationary."`
  `VLAProcessor._build_messages` needs reshaping before training, or the
  narrative loss fights the chat prior.
- vla_core's training path (`data/collate.py`, `train/pretrain.py`) does not
  emit or pass `mm_token_type_ids` and will hit the same M-RoPE `ValueError` the
  inference path did.
- Windows anchored at `t0` sometimes have no overlapping annotation segment
  (`"no narration"`), which floors narrative F1 through no fault of the model.
