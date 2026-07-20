from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inspect import inspect_episodes
from .lerobot import LeRobotReader
from .rlds import RLDSReader
from .writers import write_hdf5, write_parquet


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("must be true or false")


def _default_output_path(input_path: Path, output_dir: Path, suffix: str) -> Path:
    dataset_name = input_path.name
    return output_dir / dataset_name / f"{dataset_name}.{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vla-data-tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect and validate a sample")
    inspect_parser.add_argument("--format", choices=("lerobot", "rlds"), required=True)
    inspect_parser.add_argument("--path", type=Path, required=True)
    inspect_parser.add_argument("--max-episodes", type=_positive_int)
    inspect_parser.add_argument("--control-rate-hz", type=float)
    inspect_parser.add_argument("--decode-images", type=_boolean, default=False)

    convert_parser = subparsers.add_parser("convert", help="convert via canonical schema")
    convert_parser.add_argument("--input-format", choices=("lerobot", "rlds"), required=True)
    convert_parser.add_argument("--output-format", choices=("hdf5", "parquet"), required=True)
    convert_parser.add_argument("--input", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path)
    convert_parser.add_argument("--output-dir", type=Path, default=Path("output"))
    convert_parser.add_argument("--max-episodes", type=_positive_int)
    convert_parser.add_argument("--control-rate-hz", type=float)
    convert_parser.add_argument("--decode-images", type=_boolean, default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.path if args.command == "inspect" else args.input
    reader = (
        LeRobotReader(input_path)
        if (args.format if args.command == "inspect" else args.input_format) == "lerobot"
        else RLDSReader(
            input_path,
            control_rate_hz=args.control_rate_hz,
            decode_images=args.decode_images,
        )
    )
    episodes = reader.episodes(max_episodes=args.max_episodes)
    if args.command == "inspect":
        summary, status = inspect_episodes(episodes)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return status

    writer = write_hdf5 if args.output_format == "hdf5" else write_parquet
    suffix = "hdf5" if args.output_format == "hdf5" else "parquet"
    output = args.output or _default_output_path(input_path, args.output_dir, suffix)
    count = writer(episodes, output)
    print(json.dumps({"episodes_written": count, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
