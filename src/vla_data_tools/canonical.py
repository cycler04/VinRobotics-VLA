from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CanonicalEpisode:
    """Canonical v0.1 episode used between source readers and output writers.

    Images are references by default, not decoded pixels. Each reference records
    the source asset and the timestamp range needed to recover the episode.
    """

    episode_id: str
    dataset_name: str
    timestamp: np.ndarray
    observation_state: np.ndarray
    action_raw: np.ndarray
    is_first: np.ndarray
    is_last: np.ndarray
    is_terminal: np.ndarray
    robot: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    observation_images: dict[str, np.ndarray] = field(default_factory=dict)
    image_references: dict[str, dict[str, Any]] = field(default_factory=dict)
    action_spec: dict[str, Any] = field(default_factory=dict)
    state_spec: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.timestamp.shape[0])


@dataclass(slots=True)
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_episode(episode: CanonicalEpisode) -> ValidationResult:
    result = ValidationResult()
    arrays = {
        "timestamp": episode.timestamp,
        "observation.state": episode.observation_state,
        "action.raw": episode.action_raw,
        "is_first": episode.is_first,
        "is_last": episode.is_last,
        "is_terminal": episode.is_terminal,
    }
    arrays.update(
        {f"observation.images.{name}": value for name, value in episode.observation_images.items()}
    )
    lengths = {name: int(value.shape[0]) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        result.errors.append(f"step lengths differ: {lengths}")
        return result

    if episode.length == 0:
        result.errors.append("episode has no steps")
        return result

    if episode.observation_state.ndim != 2:
        result.errors.append("observation.state must have shape [T, Ds]")
    if episode.action_raw.ndim != 2:
        result.errors.append("action.raw must have shape [T, Da]")
    if not bool(episode.is_first[0]) or int(np.count_nonzero(episode.is_first)) != 1:
        result.errors.append("is_first must be true only at the first step")
    if not bool(episode.is_last[-1]) or int(np.count_nonzero(episode.is_last)) != 1:
        result.errors.append("is_last must be true only at the last step")
    if np.any(np.diff(episode.timestamp.astype(np.float64)) <= 0):
        result.errors.append("timestamps must be strictly increasing")

    for name in ("timestamp", "observation.state", "action.raw"):
        value = arrays[name]
        if not np.issubdtype(value.dtype, np.number):
            result.errors.append(f"{name} must be numeric, got {value.dtype}")
        elif not np.all(np.isfinite(value)):
            result.errors.append(f"{name} contains NaN or Inf")

    for name, value in episode.observation_images.items():
        if value.ndim != 4:
            result.errors.append(f"observation.images.{name} must have shape [T, H, W, C]")
        elif value.shape[-1] not in (1, 3, 4):
            result.warnings.append(
                f"observation.images.{name} has unusual channel count {value.shape[-1]}"
            )

    if not episode.action_spec:
        result.warnings.append("action semantics are undocumented")
    if not episode.state_spec:
        result.warnings.append("state semantics are undocumented")
    return result
