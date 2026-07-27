#!/usr/bin/env python3
"""Inspect ``predictions.npz`` written by ``scripts/run_vla_inference.py``.

    python src/inspect_vla_output.py
    python src/inspect_vla_output.py output/smoke/predictions.npz --show 3

The file contains predicted actions, ground truth, an action mask, and sample
keys. The command validates their shapes and values, then prints summary stats
and samples with the highest masked action error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ARRAYS = {"predictions", "actions_gt", "action_mask", "keys"}


def load_predictions(predictions_path: Path) -> dict[str, np.ndarray]:
    """Load prediction arrays and reject malformed inference output."""
    if not predictions_path.is_file():
        raise ValueError(f"file does not exist: {predictions_path}")
    with np.load(predictions_path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(f"{predictions_path} missing arrays: {', '.join(sorted(missing))}")
        arrays = {name: archive[name] for name in REQUIRED_ARRAYS}

    predictions = arrays["predictions"]
    actions_gt = arrays["actions_gt"]
    action_mask = arrays["action_mask"]
    keys = arrays["keys"]
    if predictions.ndim != 3:
        raise ValueError(f"predictions must have shape (samples, steps, actions), got {predictions.shape}")
    if actions_gt.shape != predictions.shape or action_mask.shape != predictions.shape:
        raise ValueError("predictions, actions_gt, and action_mask must have identical shapes")
    if keys.shape != (predictions.shape[0],):
        raise ValueError(f"keys shape {keys.shape} does not match {predictions.shape[0]} samples")
    if not all(np.isfinite(array).all() for array in (predictions, actions_gt, action_mask)):
        raise ValueError("predictions, actions_gt, and action_mask must be finite")
    if np.any(action_mask < 0) or np.any(action_mask > 1):
        raise ValueError("action_mask values must be in [0, 1]")
    return arrays


def masked_mae(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return masked MAE for every sample; empty masks are invalid output."""
    mask = arrays["action_mask"]
    denominator = mask.sum(axis=(1, 2))
    if np.any(denominator <= 0):
        raise ValueError("action_mask contains a sample with no valid actions")
    return np.abs((arrays["predictions"] - arrays["actions_gt"]) * mask).sum(
        axis=(1, 2)
    ) / denominator


def inspect(predictions_path: Path, show: int) -> None:
    """Print a compact inspection report for one ``predictions.npz`` file."""
    arrays = load_predictions(predictions_path)
    errors = masked_mae(arrays)
    predictions = arrays["predictions"]
    actions_gt = arrays["actions_gt"]
    mask = arrays["action_mask"]
    print(f"File: {predictions_path.resolve()}")
    print(f"Samples: {predictions.shape[0]} | action shape: {tuple(predictions.shape[1:])}")
    print(f"Dtype: predictions={predictions.dtype}, actions_gt={actions_gt.dtype}")
    print(f"Mean masked MAE: {errors.mean():.6f}")
    print(f"Masked action values: {mask.mean():.2%}")
    print(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"Ground-truth range: [{actions_gt.min():.6f}, {actions_gt.max():.6f}]")

    if show:
        print("\nWorst masked-MAE samples:")
        for index in np.argsort(errors)[::-1][:show]:
            print(f"  {arrays['keys'][index]}: mae={errors[index]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "predictions",
        nargs="?",
        type=Path,
        default=ROOT / "output" / "smoke" / "predictions.npz",
        help="path to predictions.npz",
    )
    parser.add_argument("--show", type=int, default=5, help="number of worst samples to print")
    args = parser.parse_args()
    if args.show < 0:
        parser.error("--show must be non-negative")

    try:
        inspect(args.predictions, args.show)
    except (OSError, ValueError) as error:
        sys.exit(f"Cannot inspect predictions: {error}")


if __name__ == "__main__":
    main()
