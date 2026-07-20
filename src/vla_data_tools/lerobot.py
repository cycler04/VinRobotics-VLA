from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq

from .canonical import CanonicalEpisode


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _python_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _jsonl_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
    return rows


def _read_tasks(root: Path) -> dict[int, str]:
    paths = sorted((root / "meta").glob("tasks*.parquet"))
    paths += sorted((root / "meta").glob("tasks/**/*.parquet"))
    rows = _python_rows(paths)
    rows += _jsonl_rows(sorted((root / "meta").glob("tasks*.jsonl")))
    tasks: dict[int, str] = {}
    for row in rows:
        index = int(row.get("task_index", len(tasks)))
        text = row.get("task") or row.get("language_instruction") or ""
        tasks[index] = str(text)
    return tasks


def _read_episode_metadata(root: Path) -> dict[int, dict[str, Any]]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    rows = _python_rows(paths)
    rows += _jsonl_rows(sorted((root / "meta").glob("episodes*.jsonl")))
    return {int(row["episode_index"]): row for row in rows}


def _data_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no LeRobot Parquet files found below {root / 'data'}")
    return paths


def _as_matrix(values: list[Any], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a fixed-size vector per frame, got {array.shape}")
    return array


def _image_references(
    root: Path,
    info: dict[str, Any],
    episode_meta: dict[str, Any],
    episode_index: int,
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    video_keys = [
        key for key, spec in info.get("features", {}).items() if spec.get("dtype") == "video"
    ]
    version = str(info.get("codebase_version", "unknown"))
    template = info.get("video_path")
    for key in video_keys:
        if not template:
            continue
        if version.startswith("v2"):
            chunk = episode_index // int(info.get("chunks_size", 1000))
            relative = template.format(
                episode_chunk=chunk, episode_index=episode_index, video_key=key
            )
            references[key] = {"path": relative, "from_timestamp": 0.0}
            continue

        base = f"videos/{key}"
        chunk = int(episode_meta.get(f"{base}/chunk_index", 0))
        file_index = int(episode_meta.get(f"{base}/file_index", 0))
        relative = template.format(video_key=key, chunk_index=chunk, file_index=file_index)
        references[key] = {
            "path": relative,
            "from_timestamp": float(episode_meta.get(f"{base}/from_timestamp", 0.0)),
            "to_timestamp": float(episode_meta.get(f"{base}/to_timestamp", 0.0)),
        }
    return references


class LeRobotReader:
    """Read LeRobot v2 episode Parquet or v3 file-based Parquet."""

    def __init__(self, path: str | Path):
        self.root = Path(path).expanduser().resolve()
        self.info = _read_json(self.root / "meta" / "info.json")
        self.tasks = _read_tasks(self.root)
        self.episode_metadata = _read_episode_metadata(self.root)

    def episodes(self, max_episodes: int | None = None) -> Iterator[CanonicalEpisode]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for path in _data_paths(self.root):
            table = pq.read_table(path)
            required = {"episode_index", "timestamp", "observation.state", "action"}
            missing = required.difference(table.column_names)
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
            for row in table.to_pylist():
                grouped[int(row["episode_index"])].append(row)

        for ordinal, episode_index in enumerate(sorted(grouped)):
            if max_episodes is not None and ordinal >= max_episodes:
                break
            rows = sorted(grouped[episode_index], key=lambda row: int(row.get("frame_index", 0)))
            length = len(rows)
            timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float32)
            state = _as_matrix([row["observation.state"] for row in rows], "observation.state")
            action = _as_matrix([row["action"] for row in rows], "action")
            first = np.zeros(length, dtype=np.bool_)
            last = np.zeros(length, dtype=np.bool_)
            first[0] = True
            last[-1] = True
            terminal = np.asarray(
                [bool(row.get("next.done", False)) for row in rows], dtype=np.bool_
            )
            task_index = int(rows[0].get("task_index", 0))
            episode_meta = self.episode_metadata.get(episode_index, {})
            task_text = self.tasks.get(task_index, "")
            if not task_text and episode_meta.get("tasks"):
                task_text = str(episode_meta["tasks"][0])
            success_values = [bool(row.get("next.success", False)) for row in rows]

            feature_specs = self.info.get("features", {})
            yield CanonicalEpisode(
                episode_id=str(episode_index),
                dataset_name=self.root.name,
                timestamp=timestamps,
                observation_state=state,
                action_raw=action,
                is_first=first,
                is_last=last,
                is_terminal=terminal,
                robot={
                    "name": self.info.get("robot_type", "unknown"),
                    "embodiment": self.info.get("robot_type", "unknown"),
                    "control_frequency_hz": self.info.get("fps"),
                },
                task={"language_instruction": task_text, "success": any(success_values)},
                image_references=_image_references(
                    self.root, self.info, episode_meta, episode_index
                ),
                action_spec={
                    "representation": "unknown",
                    "frame": "unknown",
                    "units": "unknown",
                    "source_feature": feature_specs.get("action", {}),
                },
                state_spec={
                    "representation": "unknown",
                    "source_feature": feature_specs.get("observation.state", {}),
                },
                source_metadata={
                    "format": "lerobot",
                    "codebase_version": self.info.get("codebase_version", "unknown"),
                    "episode_index": episode_index,
                    "task_index": task_index,
                    "episode_metadata": episode_meta,
                    "feature_specs": feature_specs,
                },
            )
