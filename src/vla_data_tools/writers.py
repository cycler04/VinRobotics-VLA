from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .canonical import CanonicalEpisode, validate_episode


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_hdf5(episodes: Iterable[CanonicalEpisode], output: str | Path) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with h5py.File(path, "w") as handle:
        handle.attrs["canonical_schema_version"] = "0.1"
        root = handle.create_group("episodes", track_order=True)
        for episode in episodes:
            validation = validate_episode(episode)
            if not validation.valid:
                raise ValueError(f"episode {episode.episode_id}: {'; '.join(validation.errors)}")
            group = root.create_group(episode.episode_id, track_order=True)
            group.attrs["dataset_name"] = episode.dataset_name
            for name in (
                "robot", "task", "image_references", "action_spec", "state_spec", "source_metadata"
            ):
                group.attrs[name] = _json(getattr(episode, name))
            steps = group.create_group("steps", track_order=True)
            steps.create_dataset("timestamp", data=episode.timestamp, compression="gzip")
            observation = steps.create_group("observation", track_order=True)
            observation.create_dataset("state", data=episode.observation_state, compression="gzip")
            if episode.observation_images:
                images = observation.create_group("images", track_order=True)
                for camera, frames in episode.observation_images.items():
                    images.create_dataset(camera, data=frames, compression="gzip", chunks=True)
            action = steps.create_group("action", track_order=True)
            action.create_dataset("raw", data=episode.action_raw, compression="gzip")
            steps.create_dataset("is_first", data=episode.is_first, compression="gzip")
            steps.create_dataset("is_last", data=episode.is_last, compression="gzip")
            steps.create_dataset("is_terminal", data=episode.is_terminal, compression="gzip")
            count += 1
        handle.attrs["episode_count"] = count
    return count


def write_parquet(episodes: Iterable[CanonicalEpisode], output: str | Path) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    count = 0
    for episode in episodes:
        validation = validate_episode(episode)
        if not validation.valid:
            raise ValueError(f"episode {episode.episode_id}: {'; '.join(validation.errors)}")
        image_references = dict(episode.image_references)
        if episode.observation_images:
            assets = path.parent / f"{path.stem}_assets" / f"episode_{episode.episode_id}"
            assets.mkdir(parents=True, exist_ok=True)
            for camera, frames in episode.observation_images.items():
                asset_path = assets / f"{camera}.npy"
                np.save(asset_path, frames)
                image_references[camera] = {
                    "path": str(asset_path.relative_to(path.parent)),
                    "format": "npy",
                    "shape": list(frames.shape),
                }
        metadata = {
            "dataset_name": episode.dataset_name,
            "robot": episode.robot,
            "task": episode.task,
            "image_references": image_references,
            "action_spec": episode.action_spec,
            "state_spec": episode.state_spec,
            "source_metadata": episode.source_metadata,
        }
        for index in range(episode.length):
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "frame_index": index,
                    "timestamp": float(episode.timestamp[index]),
                    "observation.state": episode.observation_state[index].tolist(),
                    "action.raw": episode.action_raw[index].tolist(),
                    "is_first": bool(episode.is_first[index]),
                    "is_last": bool(episode.is_last[index]),
                    "is_terminal": bool(episode.is_terminal[index]),
                    "episode_metadata_json": _json(metadata),
                }
            )
        count += 1
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return count
