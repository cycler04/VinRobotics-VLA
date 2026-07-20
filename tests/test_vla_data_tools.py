from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vla_data_tools.canonical import validate_episode
from vla_data_tools.__main__ import _default_output_path, build_parser
from vla_data_tools.inspect import inspect_episodes
from vla_data_tools.lerobot import LeRobotReader
from vla_data_tools.writers import write_hdf5, write_parquet


def _make_lerobot_v3(root: Path) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "pusht",
        "fps": 10,
        "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.image": {"dtype": "video", "shape": [96, 96, 3]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 0, "task": "push the T block"}]),
        root / "meta" / "tasks.parquet",
    )
    episode_rows = []
    frame_rows = []
    global_index = 0
    for episode_index, length in ((0, 3), (1, 2)):
        episode_rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
                "tasks": ["push the T block"],
                "videos/observation.image/chunk_index": 0,
                "videos/observation.image/file_index": 0,
                "videos/observation.image/from_timestamp": float(episode_index),
                "videos/observation.image/to_timestamp": float(episode_index + length / 10),
            }
        )
        for frame_index in range(length):
            frame_rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": frame_index / 10,
                    "observation.state": [float(frame_index), 1.0],
                    "action": [2.0, float(frame_index)],
                    "next.done": frame_index == length - 1,
                    "next.success": frame_index == length - 1,
                    "task_index": 0,
                }
            )
            global_index += 1
    pq.write_table(
        pa.Table.from_pylist(episode_rows),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(frame_rows), root / "data" / "chunk-000" / "file-000.parquet"
    )
    return root


def test_read_inspect_and_convert(tmp_path: Path) -> None:
    source = _make_lerobot_v3(tmp_path / "pusht")
    episodes = list(LeRobotReader(source).episodes())

    assert [episode.length for episode in episodes] == [3, 2]
    assert episodes[0].task["language_instruction"] == "push the T block"
    assert episodes[0].image_references["observation.image"]["path"].endswith(
        "file-000.mp4"
    )
    assert all(validate_episode(episode).valid for episode in episodes)

    summary, status = inspect_episodes(episodes)
    assert status == 0
    assert summary["episodes"] == 2
    assert summary["steps"] == {"min": 2, "median": 2.5, "max": 3}
    assert summary["instruction_coverage_percent"] == 100.0

    hdf5_path = tmp_path / "converted" / "pusht.hdf5"
    parquet_path = tmp_path / "converted" / "pusht.parquet"
    episodes[0].observation_images["decoded_front"] = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    assert write_hdf5(iter(episodes), hdf5_path) == 2
    assert write_parquet(iter(episodes), parquet_path) == 2
    with h5py.File(hdf5_path) as handle:
        assert handle.attrs["episode_count"] == 2
        assert handle["episodes/0/steps/action/raw"].shape == (3, 2)
        assert handle["episodes/0/steps/observation/images/decoded_front"].shape == (3, 4, 5, 3)
    table = pq.read_table(parquet_path)
    assert table.num_rows == 5
    assert table.column("is_first").to_pylist() == [True, False, False, True, False]
    assert (tmp_path / "converted" / "pusht_assets" / "episode_0" / "decoded_front.npy").exists()


def test_max_episodes(tmp_path: Path) -> None:
    source = _make_lerobot_v3(tmp_path / "pusht")
    episodes = list(LeRobotReader(source).episodes(max_episodes=1))
    assert len(episodes) == 1


def test_read_lerobot_v2_jsonl_metadata(tmp_path: Path) -> None:
    root = tmp_path / "pusht_v2"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.0",
        "fps": 10,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.image": {"dtype": "video", "shape": [96, 96, 3]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "push"}) + "\n", encoding="utf-8"
    )
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 2, "tasks": ["push"]}) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "frame_index": index,
                    "timestamp": index / 10,
                    "observation.state": [0.0, 1.0],
                    "action": [1.0, 2.0],
                    "next.done": index == 1,
                    "task_index": 0,
                }
                for index in range(2)
            ]
        ),
        root / "data" / "chunk-000" / "episode_000000.parquet",
    )

    episode = next(LeRobotReader(root).episodes())
    assert episode.task["language_instruction"] == "push"
    assert episode.image_references["observation.image"]["path"] == (
        "videos/chunk-000/observation.image/episode_000000.mp4"
    )


def test_convert_defaults_to_output_folder() -> None:
    args = build_parser().parse_args(
        [
            "convert",
            "--input-format",
            "lerobot",
            "--input",
            "dataset/lerobot_pusht",
            "--output-format",
            "hdf5",
        ]
    )
    assert args.output is None
    assert args.output_dir == Path("output")
    assert _default_output_path(args.input, args.output_dir, "hdf5") == Path(
        "output/lerobot_pusht/lerobot_pusht.hdf5"
    )
