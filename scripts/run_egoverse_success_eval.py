#!/usr/bin/env python
"""Score vla_core open-loop predictions on egoverse and record per-task success.

    python scripts/prepare_sample_dataset.py --sources egoverse --part val \
        --n 24 --out dataset/egoverse_sample
    python scripts/run_egoverse_success_eval.py --tag egoverse_eval

WHAT "SUCCESS" MEANS HERE
-------------------------
egoverse_v06 ships no success/reward label — it is offline human ego video, there
is no environment to roll out in.  So success is an OPEN-LOOP TRAJECTORY-MATCH
proxy, per 1.6 s / 16-step window, computed only on `valid`-masked components:

    head_pos_err   mean L2 over steps of the chunk-anchored head delta   (m)
    head_rot_err   mean geodesic angle of the head delta rotation        (deg)
    hand_pos_err   mean L2 of camera-relative wrist position, valid only (m)
    hand_rot_err   mean geodesic angle of camera-relative wrist rotation (deg)
    kp_err         mean L2 of the 21 wrist-relative keypoints, valid only(m)

    success = all enabled criteria under their tolerance

plus a separate narrative check (token-F1 of the generated narrative against the
window's ground-truth segment text >= --narrative-f1).

This is a proxy, NOT task completion.  Real task success needs a simulator, a
robot, or a trained success classifier — none exist in this release.

UNITS WARNING: the corpus stores raw metric actions; the flow-matching head is
specified to work in q01/q99-normalized space (ACTION_SPEC "Normalization").
Pass --action-norm configs/action_norm_human_ego.json once it exists, otherwise
predictions and ground truth live in different spaces and the metres/degrees
below are not comparable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VLA_CORE = ROOT / "third_party" / "02_vla_core"

# ACTION_SPEC 153-dim layout: head d_pos(3) d_rot6d(6) | L pos(3) rot6d(6) kp21(63) | R same
HEAD_POS = slice(0, 3)
HEAD_ROT = slice(3, 9)
HANDS = {"left": 9, "right": 9 + 72}
H_POS, H_ROT, H_KP = slice(0, 3), slice(3, 9), slice(9, 72)


def rot6d_to_matrix(a: np.ndarray) -> np.ndarray:
    """(...,6) first-two-columns 6D rep -> (...,3,3) via Gram-Schmidt (Zhou et al.)."""
    c0, c1 = a[..., 0:3], a[..., 3:6]
    b0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-8)
    c1 = c1 - (b0 * c1).sum(-1, keepdims=True) * b0
    b1 = c1 / (np.linalg.norm(c1, axis=-1, keepdims=True) + 1e-8)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=-1)


def geodesic_deg(a6: np.ndarray, b6: np.ndarray) -> np.ndarray:
    """(...,6),(...,6) -> (...) rotation angle between the two frames, degrees."""
    Ra, Rb = rot6d_to_matrix(a6), rot6d_to_matrix(b6)
    tr = np.einsum("...ij,...ij->...", Ra, Rb)          # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def masked_mean(err: np.ndarray, valid: np.ndarray) -> float:
    """err (T,), valid (T,) bool -> mean over valid steps, nan if none."""
    n = valid.sum()
    return float(err[valid].mean()) if n else float("nan")


def window_errors(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """pred/gt/mask are (T,153).  Returns per-window error dict (m / deg)."""
    out = {
        "head_pos_err": float(np.linalg.norm(pred[:, HEAD_POS] - gt[:, HEAD_POS], axis=-1).mean()),
        "head_rot_err": float(geodesic_deg(pred[:, HEAD_ROT], gt[:, HEAD_ROT]).mean()),
    }
    pos, rot, kp, seen = [], [], [], 0
    for off in HANDS.values():
        blk = slice(off, off + 72)
        p, g, m = pred[:, blk], gt[:, blk], mask[:, blk]
        valid = m[:, 0] > 0                                  # per-step hand validity
        if not valid.any():
            continue
        seen += 1
        pos.append(masked_mean(np.linalg.norm(p[:, H_POS] - g[:, H_POS], axis=-1), valid))
        rot.append(masked_mean(geodesic_deg(p[:, H_ROT], g[:, H_ROT]), valid))
        kp_e = np.linalg.norm((p[:, H_KP] - g[:, H_KP]).reshape(len(p), 21, 3), axis=-1).mean(-1)
        kp.append(masked_mean(kp_e, valid))
    out["hand_pos_err"] = float(np.mean(pos)) if pos else float("nan")
    out["hand_rot_err"] = float(np.mean(rot)) if rot else float("nan")
    out["kp_err"] = float(np.mean(kp)) if kp else float("nan")
    out["n_hands_valid"] = seen
    return out


_TOK = re.compile(r"[a-z0-9]+")


def token_f1(pred: str, ref: str) -> float:
    """Bag-of-tokens F1 — enough to tell 'Fold the yellow t shirt' from 'Yes.'"""
    p, r = Counter(_TOK.findall(pred.lower())), Counter(_TOK.findall(ref.lower()))
    hit = sum((p & r).values())
    if not hit:
        return 0.0
    prec, rec = hit / sum(p.values()), hit / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "dataset" / "egoverse_sample"))
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen3.5-0.8b-hf"))
    ap.add_argument("--ckpt", default=None, help="ActionHead state_dict; random init if unset")
    ap.add_argument("--action-norm", default=None,
                    help="q01/q99 stats json; predictions are de-normalized to metres with it")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--tag", default="egoverse_eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    # success tolerances
    ap.add_argument("--head-pos-tol", type=float, default=0.05, help="metres")
    ap.add_argument("--head-rot-tol", type=float, default=10.0, help="degrees")
    ap.add_argument("--hand-pos-tol", type=float, default=0.05, help="metres")
    ap.add_argument("--hand-rot-tol", type=float, default=15.0, help="degrees")
    ap.add_argument("--kp-tol", type=float, default=0.03, help="metres")
    ap.add_argument("--narrative-f1", type=float, default=0.5)
    args = ap.parse_args()

    sys.path.insert(0, str(VLA_CORE))
    from data.processing import VLAProcessor
    from model.config import VLAConfig
    from model.vla_model import VLAModel

    ds = Path(args.dataset)
    meta = json.loads((ds / "index.json").read_text())
    samples = meta["samples"][: args.limit]
    if not samples:
        raise SystemExit(f"no samples in {ds}")
    print(f"{len(samples)} windows from {ds} (sources={meta['sources']}, part={meta['part']})")

    norm = None
    if args.action_norm:
        n = json.loads(Path(args.action_norm).read_text())
        norm = (np.asarray(n["q01"], np.float32), np.asarray(n["q99"], np.float32))
        print("de-normalizing predictions with", args.action_norm)

    device = torch.device(args.device)
    cfg = VLAConfig(qwen_model_id=args.model, train_dtype=args.dtype,
                    num_actions_chunk=meta["n_steps"], action_dim=meta["action_dim"])
    proc = VLAProcessor(args.model)
    model = VLAModel(cfg).to(device).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        info = model.load_state_dict(sd.get("model", sd), strict=False)
        print(f"loaded {args.ckpt} | missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")

    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    crit = {"head_pos_err": args.head_pos_tol, "head_rot_err": args.head_rot_tol,
            "hand_pos_err": args.hand_pos_tol, "hand_rot_err": args.hand_rot_tol,
            "kp_err": args.kp_tol}

    started = datetime.now().astimezone()
    records, preds, gts, masks, steps = [], [], [], [], []
    for i, s in enumerate(samples):
        img = Image.open(ds / "frames" / f"{s['key']}.png").convert("RGB")
        inputs = proc.build_inference_inputs([img], task=s["prompt"], device=device)
        kw = {k: inputs[k] for k in ("input_ids", "attention_mask", "pixel_values",
                                     "image_grid_thw", "mm_token_type_ids") if k in inputs}

        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate_narrative(**kw, max_new_tokens=64)
            narrative = proc.processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True).strip()
            act = model.predict_action(**kw)
        dt = time.perf_counter() - t0

        npz = np.load(ds / "actions" / f"{s['key']}.npz")
        if "step_frames" not in npz.files:
            raise SystemExit(f"{ds} predates action timestamps — re-run "
                             f"scripts/prepare_sample_dataset.py to regenerate it")
        gt, mask = npz["actions"].astype(np.float32), npz["mask"].astype(np.float32)
        pred = act[0].float().cpu().numpy()
        if norm is not None:                                  # [-1,1] -> metric
            q01, q99 = norm
            pred = (pred + 1.0) / 2.0 * (q99 - q01) + q01

        errs = window_errors(pred, gt, mask)
        # nan (no valid hand in this window) must not silently count as a pass
        checks = {k: (False if np.isnan(errs[k]) else errs[k] <= tol) for k, tol in crit.items()}
        ref = " ".join(n["text"] for n in s.get("narrative_segments", [])
                       if n["gen_model"] != "joystick_v1") or s["narrative"]
        f1 = token_f1(narrative, ref)

        records.append({
            "key": s["key"], "clip_id": s["clip_id"], "task_name": s.get("task_name"),
            "generated_narrative": narrative, "reference_narrative": ref,
            "narrative_f1": round(f1, 4), "narrative_success": bool(f1 >= args.narrative_f1),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in errs.items()},
            **{f"pass_{k}": v for k, v in checks.items()},
            "action_success": bool(all(checks.values())),
            "seconds": round(dt, 3),
            # each scored step -> the clip instant it targets
            "video": s["video"], "fps": s["fps"],
            "anchor_frame": s["anchor_frame"], "anchor_seconds": s["anchor_seconds"],
            "step_frames": npz["step_frames"].tolist(),
            "step_seconds": [round(v, 6) for v in npz["step_seconds"].tolist()],
            "predicted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        preds.append(pred); gts.append(gt); masks.append(mask)
        steps.append(npz["step_frames"])
        r = records[-1]
        print(f"[{i+1:3d}/{len(samples)}] {s.get('task_name','?')[:28]:28s} "
              f"succ={int(r['action_success'])} hp={errs['head_pos_err']:6.3f}m "
              f"hr={errs['head_rot_err']:6.1f}d kp={errs['kp_err']:6.3f}m f1={f1:.2f}")

    # ── aggregate ────────────────────────────────────────────────────
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_name"] or "unknown"].append(r)

    def rate(rs, key):
        return round(sum(bool(r[key]) for r in rs) / len(rs), 4)

    def mean(rs, key):
        v = [r[key] for r in rs if not np.isnan(r[key])]
        return round(float(np.mean(v)), 6) if v else None

    per_task = {
        t: {"n": len(rs), "action_success_rate": rate(rs, "action_success"),
            "narrative_success_rate": rate(rs, "narrative_success"),
            **{k: mean(rs, k) for k in crit}}
        for t, rs in sorted(by_task.items())
    }
    summary = {
        "success_definition": "open-loop trajectory match within tolerance; egoverse has "
                              "no ground-truth task-completion label",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "dataset_generated_at": meta.get("generated_at"),
        "timebase": meta.get("timebase"),
        "dataset": str(ds), "model": args.model, "ckpt": args.ckpt,
        "action_norm": args.action_norm,
        "trained_action_head": bool(args.ckpt),
        "metrics_comparable": bool(args.ckpt and args.action_norm),
        "tolerances": {**crit, "narrative_f1": args.narrative_f1},
        "n_windows": len(records),
        "action_success_rate": rate(records, "action_success"),
        "narrative_success_rate": rate(records, "narrative_success"),
        **{f"mean_{k}": mean(records, k) for k in crit},
        "mean_seconds": round(float(np.mean([r["seconds"] for r in records])), 3),
        "per_task": per_task,
        "results": records,
    }
    (out_dir / "success.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out_dir / "predictions.npz", predictions=np.stack(preds),
                        actions_gt=np.stack(gts), action_mask=np.stack(masks),
                        step_frames=np.stack(steps),
                        step_seconds=np.stack(steps) / np.array(
                            [[r["fps"]] for r in records], np.float64),
                        keys=np.array([r["key"] for r in records]))

    print(f"\nper-task success ({len(records)} windows)")
    for t, v in per_task.items():
        print(f"  {t[:38]:38s} n={v['n']:3d} action={v['action_success_rate']:.2f} "
              f"narrative={v['narrative_success_rate']:.2f}")
    print(f"overall action={summary['action_success_rate']:.3f} "
          f"narrative={summary['narrative_success_rate']:.3f} -> {out_dir}/success.json")
    if not summary["metrics_comparable"]:
        print("NOTE: no --ckpt and/or no --action-norm — predictions and ground truth are "
              "in different spaces; success rates are a plumbing check, not a result.")


def _selftest() -> None:
    """python scripts/run_egoverse_success_eval.py --selftest"""
    rng = np.random.RandomState(0)
    T = 16
    gt = rng.randn(T, 153).astype(np.float32)
    gt[:, HEAD_ROT] = np.array([1, 0, 0, 0, 1, 0], np.float32)          # identity frame
    for off in HANDS.values():
        gt[:, off + 3:off + 9] = np.array([1, 0, 0, 0, 1, 0], np.float32)
    mask = np.ones_like(gt)

    e = window_errors(gt.copy(), gt, mask)                               # perfect match
    assert e["head_pos_err"] < 1e-6 and e["head_rot_err"] < 1e-4, e
    assert e["hand_pos_err"] < 1e-6 and e["kp_err"] < 1e-6, e

    p = gt.copy()
    p[:, HEAD_POS] += np.array([0.1, 0.0, 0.0], np.float32)              # 10 cm off
    assert abs(window_errors(p, gt, mask)["head_pos_err"] - 0.1) < 1e-5

    p = gt.copy()                                                        # 90 deg about z
    p[:, HEAD_ROT] = np.array([0, 1, 0, -1, 0, 0], np.float32)
    assert abs(window_errors(p, gt, mask)["head_rot_err"] - 90.0) < 1e-3

    m2 = mask.copy()                                                     # right hand gone
    m2[:, HANDS["right"]:HANDS["right"] + 72] = 0.0
    assert window_errors(gt.copy(), gt, m2)["n_hands_valid"] == 1

    m3 = np.zeros_like(mask)                                             # no hands at all
    m3[:, :9] = 1.0
    assert np.isnan(window_errors(gt.copy(), gt, m3)["hand_pos_err"])

    assert token_f1("fold the yellow t shirt", "fold the yellow t shirt") == 1.0
    assert token_f1("yes", "fold the yellow t shirt on the table") < 0.3
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
