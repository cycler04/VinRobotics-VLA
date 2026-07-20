from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .canonical import CanonicalEpisode


DEFAULT_RATES_HZ = {
    "r2d2_faceblur": 15.0,
    "asu_table_top_converted_externally_to_rlds": 125.0,
}


def _configure_tensorflow_gpu_memory() -> None:
    """Avoid reserving all GPU memory for the mostly CPU-bound RLDS reader."""
    import tensorflow as tf

    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            # The runtime was already initialized by another TensorFlow user.
            pass


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_text(value.item())
    return str(value) if value is not None else ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return _decode_text(value)
    return value


def _numeric_vector(value: Any) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim == 0 or not np.issubdtype(array.dtype, np.number):
        return None
    return array.astype(np.float32).reshape(-1)


def _state_from_observation(observation: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    for key in ("state", "robot_state", "proprio", "proprioception"):
        if key in observation:
            vector = _numeric_vector(observation[key])
            if vector is not None:
                return vector, [key]

    droid_keys = [key for key in ("joint_position", "gripper_position") if key in observation]
    if droid_keys:
        vectors = [_numeric_vector(observation[key]) for key in droid_keys]
        if all(vector is not None for vector in vectors):
            return np.concatenate(vectors), droid_keys  # type: ignore[arg-type]

    ignored = {"timestamp", "natural_language_instruction"}
    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in observation.items():
        if key in ignored or "image" in key:
            continue
        vector = _numeric_vector(value)
        if vector is not None and vector.size <= 256:
            candidates.append((key, vector))
    if not candidates:
        raise ValueError(f"cannot identify state in observation keys {sorted(observation)}")
    return np.concatenate([value for _, value in candidates]), [key for key, _ in candidates]


def _step_timestamp(step: dict[str, Any], index: int, rate_hz: float) -> tuple[float, bool]:
    for container in (step, step.get("observation", {})):
        for key in ("timestamp", "time", "time_stamp"):
            if key in container:
                value = np.asarray(container[key])
                if value.size == 1 and np.issubdtype(value.dtype, np.number):
                    return float(value.reshape(-1)[0]), False
    return index / rate_hz, True


class RLDSReader:
    """Read a prepared RLDS/TFDS directory into canonical episodes."""

    def __init__(
        self,
        path: str | Path,
        control_rate_hz: float | None = None,
        decode_images: bool = True,
    ):
        try:
            import tensorflow_datasets as tfds
        except ImportError as error:
            raise RuntimeError("RLDS support requires tensorflow and tensorflow-datasets") from error

        _configure_tensorflow_gpu_memory()

        root = Path(path).expanduser().resolve()
        candidates = sorted(root.rglob("dataset_info.json"), key=lambda item: len(item.parts))
        if not candidates:
            raise FileNotFoundError(f"dataset_info.json not found below {root}")
        self.dataset_dir = candidates[0].parent
        self.builder = tfds.builder_from_directory(str(self.dataset_dir))
        self.dataset_name = self.builder.info.name
        if control_rate_hz is not None:
            self.control_rate_hz = float(control_rate_hz)
            self.control_rate_source = "cli_override"
        elif self.dataset_name in DEFAULT_RATES_HZ:
            self.control_rate_hz = DEFAULT_RATES_HZ[self.dataset_name]
            self.control_rate_source = "dataset_profile"
        else:
            self.control_rate_hz = 10.0
            self.control_rate_source = "default_assumption"
        self.decode_images = decode_images
        self.tfrecord_paths = sorted(self.dataset_dir.glob("*.tfrecord-*"))
        observation_features = self.builder.info.features["steps"].feature["observation"]
        self.image_feature_shapes = {
            key: list(feature.shape)
            for key, feature in observation_features.items()
            if "image" in key and len(feature.shape) == 3
        }
        self.decoders = (
            None
            if self.decode_images
            else {
                "steps": {
                    "observation": {
                        key: tfds.decode.SkipDecoding() for key in self.image_feature_shapes
                    }
                }
            }
        )
        self.features_json = json.loads(
            (self.dataset_dir / "features.json").read_text(encoding="utf-8")
        )

    def episodes(self, max_episodes: int | None = None) -> Iterator[CanonicalEpisode]:
        import tensorflow_datasets as tfds

        split_names = list(self.builder.info.splits)
        expected_shards = sum(
            len(self.builder.info.splits[name].shard_lengths) for name in split_names
        )
        partial_release = bool(self.tfrecord_paths) and len(self.tfrecord_paths) < expected_shards
        if partial_release:
            import tensorflow as tf

            raw_dataset = tf.data.TFRecordDataset(
                [str(path) for path in self.tfrecord_paths],
                num_parallel_reads=tf.data.AUTOTUNE,
                buffer_size=8 * 1024 * 1024,
            )
            dataset = raw_dataset.map(
                lambda serialized: self.builder.info.features.deserialize_example(
                    serialized, decoders=self.decoders
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            datasets = [
                self.builder.as_dataset(split=name, decoders=self.decoders) for name in split_names
            ]
            dataset = datasets[0]
            for additional_dataset in datasets[1:]:
                dataset = dataset.concatenate(additional_dataset)
        split_label = "partial_local_shards" if partial_release else "+".join(split_names)
        if max_episodes is not None:
            dataset = dataset.take(max_episodes)
        for episode_index, episode in enumerate(dataset):
            top_level = tfds.as_numpy({key: value for key, value in episode.items() if key != "steps"})
            steps = list(tfds.as_numpy(episode["steps"]))
            if not steps:
                continue

            state_rows: list[np.ndarray] = []
            action_rows: list[np.ndarray] = []
            timestamps: list[float] = []
            first: list[bool] = []
            last: list[bool] = []
            terminal: list[bool] = []
            image_rows: dict[str, list[np.ndarray]] = {}
            image_shapes: dict[str, list[int]] = dict(self.image_feature_shapes)
            state_keys: list[str] = []
            instruction = ""
            timestamps_synthesized = False
            for step_index, step in enumerate(steps):
                observation = step.get("observation", {})
                state, state_keys = _state_from_observation(observation)
                action = _numeric_vector(step.get("action"))
                if action is None:
                    raise ValueError("RLDS step action is missing or non-numeric")
                timestamp, synthesized = _step_timestamp(step, step_index, self.control_rate_hz)
                timestamps_synthesized = timestamps_synthesized or synthesized
                state_rows.append(state)
                action_rows.append(action)
                timestamps.append(timestamp)
                first.append(bool(step.get("is_first", step_index == 0)))
                last.append(bool(step.get("is_last", step_index == len(steps) - 1)))
                terminal.append(bool(step.get("is_terminal", False)))
                if not instruction:
                    instruction = _decode_text(
                        step.get("language_instruction")
                        or observation.get("natural_language_instruction")
                        or ""
                    )
                for key, value in observation.items():
                    array = np.asarray(value)
                    if "image" in key and array.ndim == 3:
                        image_shapes[key] = list(array.shape)
                        if self.decode_images:
                            image_rows.setdefault(key, []).append(array)

            images = {
                key: np.stack(values)
                for key, values in image_rows.items()
                if len(values) == len(steps)
            }
            source_flags = {
                "is_first_true_indices": [index for index, value in enumerate(first) if value],
                "is_last_true_indices": [index for index, value in enumerate(last) if value],
                "is_terminal_true_indices": [index for index, value in enumerate(terminal) if value],
            }
            canonical_first = np.zeros(len(steps), dtype=np.bool_)
            canonical_last = np.zeros(len(steps), dtype=np.bool_)
            canonical_first[0] = True
            canonical_last[-1] = True
            image_references = (
                {}
                if self.decode_images
                else {
                    key: {
                        "path": str(self.dataset_dir),
                        "format": "tfrecord",
                        "shape": shape,
                        "decode_required": True,
                    }
                    for key, shape in image_shapes.items()
                }
            )
            yield CanonicalEpisode(
                episode_id=str(episode_index),
                dataset_name=self.dataset_name,
                timestamp=np.asarray(timestamps, dtype=np.float32),
                observation_state=np.stack(state_rows).astype(np.float32),
                action_raw=np.stack(action_rows).astype(np.float32),
                is_first=canonical_first,
                is_last=canonical_last,
                is_terminal=np.asarray(terminal, dtype=np.bool_),
                robot={
                    "name": "unknown",
                    "embodiment": "unknown",
                    "control_frequency_hz": self.control_rate_hz,
                },
                task={"language_instruction": instruction, "success": None},
                observation_images=images,
                image_references=image_references,
                action_spec={
                    "representation": (
                        "joint_velocity_gripper_position"
                        if self.dataset_name == "r2d2_faceblur"
                        else "source_defined"
                    ),
                    "frame": "unknown",
                    "units": "unknown",
                    "source_feature": "action",
                },
                state_spec={
                    "representation": "+".join(state_keys),
                    "source_features": state_keys,
                },
                source_metadata={
                    "format": "rlds",
                    "tfds_name": self.dataset_name,
                    "tfds_version": str(self.builder.info.version),
                    "split": split_label,
                    "episode_index": episode_index,
                    "episode_metadata": _jsonable(top_level),
                    "timestamps_synthesized": timestamps_synthesized,
                    "control_rate_source": self.control_rate_source,
                    "source_flags": source_flags,
                    "decode_images": self.decode_images,
                    "partial_release": partial_release,
                    "available_shards": len(self.tfrecord_paths),
                    "expected_shards": expected_shards,
                },
            )
