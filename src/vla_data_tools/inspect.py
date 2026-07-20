from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Iterable

import numpy as np

from .canonical import CanonicalEpisode, validate_episode


def inspect_episodes(episodes: Iterable[CanonicalEpisode]) -> tuple[dict[str, object], int]:
    items = list(episodes)
    if not items:
        raise ValueError("no episodes found")
    lengths = [episode.length for episode in items]
    rates: list[float] = []
    camera_counts: Counter[str] = Counter()
    errors: list[str] = []
    warnings: list[str] = []
    for episode in items:
        result = validate_episode(episode)
        errors.extend(f"episode {episode.episode_id}: {value}" for value in result.errors)
        warnings.extend(f"episode {episode.episode_id}: {value}" for value in result.warnings)
        camera_counts.update(set(episode.image_references) | set(episode.observation_images))
        if episode.length > 1:
            delta = np.diff(episode.timestamp.astype(np.float64))
            if np.all(delta > 0):
                rates.append(float(1.0 / np.median(delta)))
    instructions = sum(bool(item.task.get("language_instruction")) for item in items)
    expected_state_shape = items[0].observation_state.shape[1:]
    expected_action_shape = items[0].action_raw.shape[1:]
    expected_cameras = set(items[0].image_references) | set(items[0].observation_images)
    for episode in items[1:]:
        if episode.observation_state.shape[1:] != expected_state_shape:
            errors.append(f"episode {episode.episode_id}: inconsistent state shape")
        if episode.action_raw.shape[1:] != expected_action_shape:
            errors.append(f"episode {episode.episode_id}: inconsistent action shape")
        cameras = set(episode.image_references) | set(episode.observation_images)
        if cameras != expected_cameras:
            errors.append(f"episode {episode.episode_id}: inconsistent or missing camera modality")

    feature_specs = items[0].source_metadata.get("feature_specs", {})
    image_shapes = {
        key: spec.get("shape")
        for key, spec in feature_specs.items()
        if spec.get("dtype") in {"video", "image"}
    }
    image_shapes.update(
        {key: list(value.shape[1:]) for key, value in items[0].observation_images.items()}
    )
    image_shapes.update(
        {
            key: reference.get("shape")
            for key, reference in items[0].image_references.items()
            if reference.get("shape") is not None
        }
    )
    summary: dict[str, object] = {
        "episodes": len(items),
        "steps": {"min": min(lengths), "median": median(lengths), "max": max(lengths)},
        "control_rate_hz_median": median(rates) if rates else None,
        "cameras": sorted(camera_counts),
        "image_shapes": image_shapes,
        "state_shape": ["T", items[0].observation_state.shape[1]],
        "state_dtype": str(items[0].observation_state.dtype),
        "action_shape": ["T", items[0].action_raw.shape[1]],
        "action_dtype": str(items[0].action_raw.dtype),
        "instruction_coverage_percent": round(100.0 * instructions / len(items), 2),
        "action_convention": items[0].action_spec.get("representation", "unknown"),
        "validation_errors": errors,
        "validation_warnings": warnings,
    }
    return summary, 0 if not errors else 1
