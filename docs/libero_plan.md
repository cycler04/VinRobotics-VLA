
# LIBERO integration plan

Goal: train the memory model on LIBERO demonstrations (`/home/tho2/LIBERO/datasets`, robomimic-style HDF5: `libero_spatial / object / goal / 10 / 90`) and evaluate success rate in the LIBERO simulator — the paper's protocol for its group-mode results (16 random frames per episode, `mem_length=16`, suites trained separately; Long-10/90 jointly, 40k steps).

Verified facts about the data (2026-08-19):

- Per task one file, 50 demos: `data/demo_k/actions (T,7)` float64 (delta-EEF + gripper), `obs/agentview_rgb`, `obs/eye_in_hand_rgb` (T,128,128,3) uint8, `obs/joint_states (7)`, `obs/gripper_states (2)`, `ee_pos/ee_ori/ee_states`.
- Language lives in `data.attrs["problem_info"].language_instruction`.
- Simulator package importable from `/home/tho2/LIBERO/libero/libero` (`benchmark/`, `envs/`).

What stays untouched (the point of the adapter approach): `EpisodeGroupBatchSampler` / `ContinuousEpisodeBatchSampler`, `VLACollator`, `MemoryModule`/`MemoryBank` incl. the new persistence path, both trainers, `predict_action`. They only require a dataset exposing `index` of `(clip_idx, t0)` rows, `sampler.clips[ci]["source"]`, and the sample-dict keys (`clip_id`, `t0` included since the bank-persistence change — see `docs/memory_module/bank_persistence.md`).

---

## Step 1 — `data/libero_dataset.py`: dataset adapter

**What:** `LiberoPretrainDataset` mirroring `CorpusPretrainDataset`'s public surface:

- "clips" = (task file, demo) pairs; `clip_id = "libero_object/pick_up_the_butter…/demo_0"`, `source = "libero_object"` (per suite).
- `index` = numpy `(N,2)` of `(clip_idx, t0)`, one window per frame (`window_stride_s=0` equivalent), built at init exactly like `corpus_dataset.py:192-199`.
- `__getitem__` returns the same dict keys the collator consumes: `image` (agentview frame, PIL/np), `task` (language_instruction), `history`/`recent`/`narrative` empty (see step 3), `actions` = next 16 raw actions from `t0`, `action_mask` zero-padded past the demo end, `source`, `clip_id`, `t0`, `proprio` optional (step 4).
- HDF5 opened lazily per worker (h5py handles don't survive fork), file-handle cache per worker.
- **No-op filtering**: drop frames whose action is a no-op, replicating OpenVLA/MemoryVLA's `libero_*_no_noops` RLDS mixes — the paper's LIBERO numbers are trained on filtered data, so an unfiltered adapter would not reproduce the baseline. (MemoryVLA trains LIBERO from those RLDS mixes via a TF pipeline — not worth importing; the filter is the only part of it that affects results.)

**Why needed:** the entire existing pipeline consumes one dataset interface; a second loader implementing it is the only new data code required. Rewriting the corpus loader to also parse HDF5 would couple two unrelated formats. LIBERO images come as arrays, not mp4 — decode path can't be shared anyway.

## Step 2 — action space: `action_dim=7` + LIBERO norm stats

**What:** config `action_dim` becomes dataset-dependent (7 for LIBERO vs `ACTION_DIM=153` human-ego). New stats tool (or a mode in `data/compute_action_stats.py`) computing per-dim center/scale over LIBERO actions → `configs/action_norm_libero.json`; `LiberoPretrainDataset` normalizes with it, dead-zone list empty.

**Why needed:** the action head's output layer is sizessh tho2@100.89.98.89d by `action_dim` — 153-dim human keypoint targets and 7-DoF robot deltas are different spaces; without new stats, normalization built for `r_kp21`/`r_rot6d` would scale robot actions nonsensically and the flow-matching loss would be dominated by whichever dims happen to be large.

## Step 3 — narrative-free path

**What:** LIBERO samples carry `task` but no hierarchy. Emit empty `history`/`recent` and `narrative=None`; verify `VLACollator` then yields `labels` that contribute zero narrative CE (`has_narrative` already exists — confirm the masking path, add a test). Train with `--narrative-loss-weight 0` (flag exists via config `narrative_loss_weight`) and skip `unfreeze_embeddings`/control-token registration for LIBERO runs.

**Why needed:** the LM loss and `<|done|>`/`<|endsub|>` targets are built from the corpus hierarchy cache, which does not exist for LIBERO. Fabricating targets from the task string would train the narrative head on noise; masking it out keeps the action path identical to the paper's LIBERO setup (paper trains vision→action only, single third-person RGB, no narrative). Leaving embeddings frozen also removes the catastrophic-forgetting machinery (`restrict_embedding_grad`) from the run — not needed when no new tokens are emitted.

## Step 4 — proprio (optional, off by default)

**What:** wire `joint_states (7) + gripper_states (2)` → `proprio_dim=9` behind the existing `ProprioEncoder` knob; default off for the first runs.

**Why needed (as an option):** paper's LIBERO input is RGB + instruction only — matching it first keeps the comparison clean. The encoder already exists and LIBERO provides proprio, so the knob is nearly free; useful ablation later, but off by default so a regression can't hide in an extra input.

## Step 5 — training wiring

**What:** `--dataset libero --libero-root /home/tho2/LIBERO/datasets --libero-suites libero_object …` in `train/common.py` selecting `LiberoPretrainDataset`; everything downstream unchanged. Group mode default `G=16` (paper: 16 random frames, matches `memory_length=16`); `--continuous` available since the persistence change makes it meaningful. Suites trained separately per paper protocol (Long-10+90 jointly).

**Why needed:** trainers/build functions currently hardcode the corpus dataset; one selection branch is the entire change. G=16 default because that is the paper's LIBERO configuration and our EDA (`tools/eda_memory_length.py`) chose L against episode lengths in paper units.

## Step 6 — LIBERO simulator eval: `eval/libero_sim.py`

**What:** rollout script using `/home/tho2/LIBERO` benchmark API: for each task in a suite, N rollouts (paper protocol; LIBERO standard = 20/task, 500 steps max), per-episode fresh `MemoryBank`, `predict_action(..., memory=bank, memory_step=<env step>)` executing 16-step chunks (re-plan every chunk), success = env termination flag. Reports per-task and suite-mean success. **Port the skeleton from the reference**: `~/Dung_Workspace/MemoryVLA/evaluation/libero/` (`eval_libero.py` suite/env/initial-state/success loop, `libero_utils.py`/`robot_utils.py` helpers) — its `episode_first_frame='True'` bank-reset pattern maps to our fresh-bank-per-episode; only the policy call changes.

**Why needed:** the benchmark metric is simulator success rate; `eval/offline.py` measures action regression on held-out windows, which cannot show whether memory helps task completion — the entire motivation of the memory module. `memory_step` must be the env step counter so TE units match training-time absolute t0 (the bank-persistence change made TE steps absolute; a 0,1,2 push ordinal would be a train/test mismatch again).

## Step 7 — tests + smoke

**What:**

- Unit: `tests/test_libero_dataset.py` — index shape, chunk padding + `action_mask` at demo tail, `clip_id`/`t0` present, empty-narrative labels produce zero narrative loss.
- Smoke: overfit one task file (`--overfit 8`), 24 steps, loss falls; then a 2-rollout sim eval on that task to prove the env loop runs.

**Why needed:** the demo-tail padding and the narrative-masking path are the two places silent correctness bugs can enter (wrong mask → training on garbage past episode end; wrong labels → narrative CE on empty strings). Each gets the one small test that fails if broken, per repo test style.

## Step 8 — run matrix (after everything above is green)

Paper-shaped baseline: per-suite group-mode runs (Spatial/Object/Goal 20k steps, Long 40k), G=16, detach-push default. Then the experiment this repo exists for: same suites with `--continuous` + persistent banks vs group mode — LIBERO Long is where cross-batch memory should show, and MemoryVLA+ shipped only group mode (`parallel_stream` unimplemented), so this comparison is novel.

**Why this order:** baseline first isolates the adapter (comparable to published numbers ⇒ adapter correct); the continuous-vs-group comparison is only interpretable once the baseline reproduces.

---

## Known risks

- **Env dependencies**: LIBERO sim needs `robosuite`/MuJoCo in the env used for eval; verify import before writing the rollout loop (`python -c "from libero.libero import benchmark"` with `/home/tho2/LIBERO` on `PYTHONPATH`).
- **128×128 inputs**: far below the corpus resolution; Qwen processor upscales, but check token counts/VRAM once — smaller images may actually free VRAM vs corpus runs.
- **Action chunk vs control rate**: LIBERO demos are ~20 Hz; chunk of 16 raw steps ≈ 0.8 s vs corpus 1.6 s. Fine for training; just don't reuse `action_hz`-derived constants from the corpus config (e.g. `action_horizon_s`) — the dataset adapter owns its own timing.
- **Wrist camera unused** initially (paper uses third-person only); revisit with proprio ablation.
