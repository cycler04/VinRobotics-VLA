#!/usr/bin/env python
"""Pull a small self-contained sample out of the Layer-1 corpus into ./dataset/.

Reads the big v06 releases on /mnt/SSD4 (needs the data_corpus repo + ffmpeg),
decodes one RGB frame per window and packs the 153-dim action chunk, then writes
everything as plain PNG + NPZ so the inference runner never has to touch the
245 GB release or the corpus loader again.

    python scripts/prepare_sample_dataset.py --n 32

Layout written:
    dataset/egodex_sample/index.json          per-window prompt/metadata + timebase
    dataset/egodex_sample/frames/<key>.png    head-camera frame
    dataset/egodex_sample/actions/<key>.npz   actions/mask (16,153) + step_frames (16,)
                                              + step_seconds (16,) + step_offsets (16,)

Every action step carries the absolute source-video frame it targets and that
frame's time in the clip, so a predicted chunk can be laid back over the video.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VLA_CORE = ROOT / "third_party" / "02_vla_core"
DATA_CORPUS = Path("/mnt/SSD3/code/VinRobotics/data_corpus/src")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--releases-json", default=str(VLA_CORE / "configs" / "releases.json"))
    ap.add_argument("--sources", default="egodex", help="comma-separated release keys")
    ap.add_argument("--part", default="val", choices=["train", "val", "all"])
    ap.add_argument("--n", type=int, default=32, help="number of windows to extract")
    ap.add_argument("--exclude-tasks", default="debug",
                    help="comma-separated task_name values to skip (egoverse's 79%% "
                         "'debug' clips carry joystick-only narratives)")
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--action-hz", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "dataset" / "egodex_sample"))
    args = ap.parse_args()

    sys.path[:0] = [str(VLA_CORE), str(DATA_CORPUS)]
    from corpus.labels.pretrain_loader import Layer1PretrainSampler
    from data.corpus_dataset import decode_frame, pack_actions

    releases = json.loads(Path(args.releases_json).read_text())
    keys = args.sources.split(",")
    sampler = Layer1PretrainSampler({k: releases[k] for k in keys},
                                    n_steps=args.n_steps, action_hz=args.action_hz,
                                    part=args.part)
    print(f"{len(sampler.clips)} clips in part={args.part}")

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "actions").mkdir(parents=True, exist_ok=True)

    # One window per clip, round-robin over split groups so tasks stay varied
    # (egodex val is task-grouped: 3 tasks only, and a flat random draw misses one).
    rng = np.random.RandomState(args.seed)
    drop = {t for t in args.exclude_tasks.split(",") if t}
    by_group: dict = {}
    for ci, c in enumerate(sampler.clips):
        if c.get("task_name") in drop:
            continue
        by_group.setdefault(c.get("task_name") or c["_group"], []).append(ci)
    if not by_group:
        raise SystemExit(f"no clips left after excluding tasks {sorted(drop)}")
    for g in by_group.values():
        rng.shuffle(g)
    groups = sorted(by_group)
    picks = [by_group[g][i] for i in range(max(len(v) for v in by_group.values()))
             for g in groups if i < len(by_group[g])][: args.n * 3]

    index = []
    for ci in picks:
        if len(index) >= args.n:
            break
        clip = sampler.clips[int(ci)]
        try:
            smp = sampler.sample(clip, 0)
            act, msk = pack_actions(smp)
            frame = decode_frame(smp["video"], smp["video_frame"])
        except Exception as e:                      # bad video / missing stream: skip
            print(f"  skip {clip.get('clip_id')}: {type(e).__name__}: {e}")
            continue

        # Timebase: action step k lands on an absolute source-video frame, and the
        # `frame` column IS the decode index in v06 — so these tie every predicted
        # step back to a real instant of the clip the observation came from.
        fps = float(clip["fps"])
        step_frames = np.asarray(smp["video_frames_steps"], np.int64)
        step_seconds = step_frames / fps                      # clip timebase, seconds
        anchor_seconds = smp["video_frame"] / fps
        step_offsets = step_seconds - anchor_seconds          # == k / action_hz

        key = smp["clip_id"].replace("/", "__")
        Image.fromarray(frame).save(out / "frames" / f"{key}.png")
        np.savez_compressed(out / "actions" / f"{key}.npz", actions=act, mask=msk,
                            step_frames=step_frames, step_seconds=step_seconds,
                            step_offsets=step_offsets)

        narr = " ".join(n["text"] for n in smp["narratives"][:2]) or "no narration"
        index.append({
            "key": key,
            "clip_id": smp["clip_id"],
            "source": smp["source"],
            "task_name": clip.get("task_name"),
            "video": smp["video"],
            "fps": fps,
            "stride": int(clip["_stride"]),
            "anchor_row": int(smp["anchor_row"]),
            "anchor_frame": int(smp["video_frame"]),
            "anchor_seconds": round(anchor_seconds, 6),
            "step_frames": step_frames.tolist(),
            "step_seconds": [round(s, 6) for s in step_seconds.tolist()],
            "chunk_span_seconds": round(float(step_seconds[-1] - anchor_seconds), 6),
            "video_frame": int(smp["video_frame"]),
            "narrative": narr,
            "narrative_segments": [{"text": n["text"], "gen_model": n["gen_model"]}
                                   for n in smp["narratives"]],
            "joystick": smp["joystick"] or "stationary",
            "prompt": f"Task context: {narr}\nLocomotion: {smp['joystick'] or 'stationary'}",
        })
        print(f"  [{len(index):3d}/{args.n}] {smp['clip_id']}")

    meta = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "part": args.part, "sources": keys, "n_steps": args.n_steps,
        "action_hz": args.action_hz, "action_dim": int(act.shape[1]),
        "timebase": "step_frames are absolute source-video frame indices (v06 `frame` "
                    "column == decode index); step_seconds = step_frames / clip fps",
        "releases": {k: releases[k] for k in keys}, "samples": index,
    }
    (out / "index.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(index)} windows -> {out}")


if __name__ == "__main__":
    main()
