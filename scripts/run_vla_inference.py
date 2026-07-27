#!/usr/bin/env python
"""Run vla_core inference over the local sample dataset, write results to ./output/.

    python scripts/run_vla_inference.py                    # all samples in dataset/
    python scripts/run_vla_inference.py --limit 4 --tag smoke

Needs only ./dataset/<name>/ and the HF Qwen3.5 weights — no corpus loader, no
ffmpeg, no /mnt/SSD4.  Without --ckpt the ActionHead is randomly initialised, so
predicted actions are noise and only the narrative + shapes are meaningful.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VLA_CORE = ROOT / "third_party" / "02_vla_core"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "dataset" / "egodex_sample"))
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen3.5-0.8b-hf"))
    ap.add_argument("--ckpt", default=None, help="ActionHead state_dict (.pt); random init if unset")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--tag", default="run", help="subfolder name under --out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    sys.path.insert(0, str(VLA_CORE))
    from data.processing import VLAProcessor
    from model.config import VLAConfig
    from model.vla_model import VLAModel

    ds = Path(args.dataset)
    meta = json.loads((ds / "index.json").read_text())
    samples = meta["samples"][: args.limit]
    print(f"{len(samples)} samples from {ds}")

    device = torch.device(args.device)
    cfg = VLAConfig(qwen_model_id=args.model, train_dtype=args.dtype,
                    num_actions_chunk=meta["n_steps"], action_dim=meta["action_dim"])
    proc = VLAProcessor(args.model)
    model = VLAModel(cfg).to(device).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        missing = model.load_state_dict(sd.get("model", sd), strict=False)
        print("loaded ckpt:", args.ckpt, "| missing:", len(missing.missing_keys))
    model.print_trainable_summary()

    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now().astimezone()
    preds, gts, masks, records, steps = [], [], [], [], []
    for i, s in enumerate(samples):
        img = Image.open(ds / "frames" / f"{s['key']}.png").convert("RGB")
        inputs = proc.build_inference_inputs([img], task=s["prompt"], device=device)
        qwen_kwargs = {k: inputs[k] for k in
                       ("input_ids", "attention_mask", "pixel_values",
                        "image_grid_thw", "mm_token_type_ids") if k in inputs}

        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate_narrative(**qwen_kwargs, max_new_tokens=64)
            narrative = proc.processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True).strip()
            act = model.predict_action(**qwen_kwargs)
        dt = time.perf_counter() - t0

        npz = np.load(ds / "actions" / f"{s['key']}.npz")
        if "step_frames" not in npz.files:
            raise SystemExit(f"{ds} predates action timestamps — re-run "
                             f"scripts/prepare_sample_dataset.py to regenerate it")
        pred = act[0].float().cpu().numpy()
        preds.append(pred)
        gts.append(npz["actions"])
        masks.append(npz["mask"])
        steps.append(npz["step_frames"])

        # Untrained head -> this is a sanity number, not a metric.
        err = float(np.abs((pred - npz["actions"]) * npz["mask"]).sum() / max(npz["mask"].sum(), 1))
        records.append({"key": s["key"], "clip_id": s["clip_id"],
                        "prompt_narrative": s["narrative"],
                        "generated_narrative": narrative,
                        "masked_mae": err, "seconds": round(dt, 3),
                        # each predicted step -> the clip instant it targets
                        "video": s["video"], "fps": s["fps"],
                        "anchor_frame": s["anchor_frame"],
                        "anchor_seconds": s["anchor_seconds"],
                        "step_frames": npz["step_frames"].tolist(),
                        "step_seconds": [round(v, 6) for v in npz["step_seconds"].tolist()],
                        "predicted_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        print(f"[{i+1:3d}/{len(samples)}] {dt:5.2f}s  mae={err:7.4f}  {narrative[:70]!r}")

    P = np.stack(preds)
    np.savez_compressed(out_dir / "predictions.npz", predictions=P,
                        actions_gt=np.stack(gts), action_mask=np.stack(masks),
                        step_frames=np.stack(steps),
                        step_seconds=np.stack(steps) / np.array(
                            [[r["fps"]] for r in records], np.float64),
                        keys=np.array([r["key"] for r in records]))
    summary = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "dataset_generated_at": meta.get("generated_at"),
        "timebase": meta.get("timebase"),
        "model": args.model, "ckpt": args.ckpt, "dataset": str(ds),
        "n_samples": len(records), "action_shape": list(P.shape[1:]),
        "trained_action_head": bool(args.ckpt),
        "mean_masked_mae": float(np.mean([r["masked_mae"] for r in records])),
        "mean_seconds": float(np.mean([r["seconds"] for r in records])),
        "results": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    assert P.shape[1:] == (meta["n_steps"], meta["action_dim"]), P.shape
    assert np.isfinite(P).all(), "non-finite predictions"
    print(f"\nwrote {out_dir}/predictions.npz + summary.json")
    print(f"mean {summary['mean_seconds']:.2f}s/sample, action shape {tuple(P.shape[1:])}")
    if not args.ckpt:
        print("NOTE: ActionHead is random-init (no --ckpt) — actions are noise.")


if __name__ == "__main__":
    main()
