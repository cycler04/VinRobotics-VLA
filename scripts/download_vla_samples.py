#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


SOURCES = {
    "droid": {
        "bucket": "gresearch",
        "prefix": "robotics/droid_100/",
        "destination": "droid_100",
    },
    "oxe": {
        "bucket": "gdm-robotics-open-x-embodiment",
        "prefix": "asu_table_top_converted_externally_to_rlds/0.1.0/",
        "destination": "oxe/asu_table_top_converted_externally_to_rlds",
    },
    "oxe200": {
        "bucket": "gdm-robotics-open-x-embodiment",
        "prefix": "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds/0.1.0/",
        "destination": "oxe/utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    },
}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def list_gcs_objects(bucket: str, prefix: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        query = {"prefix": prefix, "maxResults": "1000"}
        if token:
            query["pageToken"] = token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?"
            + urllib.parse.urlencode(query)
        )
        with urllib.request.urlopen(url) as response:
            page = json.load(response)
        items.extend(item for item in page.get("items", []) if int(item.get("size", 0)) > 0)
        token = page.get("nextPageToken")
        if not token:
            return items


def gcs_object_metadata(bucket: str, name: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(name, safe="")
    with urllib.request.urlopen(
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded}"
    ) as response:
        return json.load(response)


def download_object(bucket: str, item: dict[str, Any], destination: Path, prefix: str) -> Path:
    relative = Path(item["name"].removeprefix(prefix))
    output = destination / relative
    expected_size = int(item["size"])
    if output.exists() and output.stat().st_size == expected_size:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        "https://storage.googleapis.com/"
        f"{bucket}/{urllib.parse.quote(item['name'], safe='/')}"
    )
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = "ab" if offset and response.status == 206 else "wb"
        with partial.open(mode) as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
    if partial.stat().st_size != expected_size:
        raise IOError(
            f"size mismatch for {item['name']}: {partial.stat().st_size} != {expected_size}"
        )
    partial.replace(output)
    return output


def write_manifest(destination: Path) -> None:
    lines = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        if ".cache" in path.parts or path.name == "sample_manifest.tsv":
            continue
        lines.append(f"{path.relative_to(destination)}\t{path.stat().st_size}\n")
    (destination / "sample_manifest.tsv").write_text("".join(lines), encoding="utf-8")


def download_gcs_source(name: str, dataset_root: Path, workers: int) -> None:
    source = SOURCES[name]
    destination = dataset_root / source["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    items = list_gcs_objects(source["bucket"], source["prefix"])
    total = sum(int(item["size"]) for item in items)
    print(f"{name}: {len(items)} files, {total / 1024**2:.1f} MiB", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_object,
                source["bucket"],
                item,
                destination,
                source["prefix"],
            ): item
            for item in items
        }
        completed = 0
        for future in as_completed(futures):
            path = future.result()
            completed += 1
            print(f"{name}: [{completed}/{len(items)}] {path.name}", flush=True)
    write_manifest(destination)


def download_droid_target(dataset_root: Path, workers: int, target_episodes: int) -> None:
    bucket = "gresearch"
    prefix = "robotics/droid/1.0.0/"
    destination = dataset_root / f"droid_{target_episodes}"
    info_name = prefix + "dataset_info.json"
    with urllib.request.urlopen(f"https://storage.googleapis.com/{bucket}/{info_name}") as response:
        dataset_info = json.load(response)
    split = dataset_info["splits"][0]
    lengths = [int(value) for value in split["shardLengths"]]
    selected_count = 0
    selected_shards = 0
    for length in lengths:
        selected_count += length
        selected_shards += 1
        if selected_count >= target_episodes:
            break

    base = "r2d2_faceblur-train.tfrecord"
    object_names = [prefix + "CC-BY-4.0", info_name, prefix + "features.json"]
    object_names += [
        prefix + f"{base}-{index:05d}-of-{len(lengths):05d}"
        for index in range(selected_shards)
    ]
    items = [gcs_object_metadata(bucket, name) for name in object_names]
    total = sum(int(item["size"]) for item in items)
    print(
        f"droid200: {selected_shards} shards, {selected_count} available episodes, "
        f"{total / 1024**3:.2f} GiB",
        flush=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_object, bucket, item, destination, prefix): item
            for item in items
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path = future.result()
            print(f"droid200: [{completed}/{len(items)}] {path.name}", flush=True)
    write_manifest(destination)


def download_lerobot(dataset_root: Path, token: str | None) -> None:
    destination = dataset_root / "lerobot_pusht"
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="lerobot/pusht",
        repo_type="dataset",
        local_dir=destination,
        token=token,
    )
    write_manifest(destination)
    print(f"lerobot: ready at {destination}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download three small Sprint 1 VLA samples")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--only",
        choices=("all", "lerobot", "droid", "droid200", "oxe", "oxe200"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-episodes", type=int, default=200)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be greater than zero")
    if args.target_episodes < 1:
        parser.error("--target-episodes must be greater than zero")

    env = {**read_env(args.env_file), **os.environ}
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    selected = ("lerobot", "droid200", "oxe200") if args.only == "all" else (args.only,)
    for name in selected:
        if name == "lerobot":
            download_lerobot(args.dataset_root, env.get("HF_TOKEN"))
        elif name == "droid200":
            download_droid_target(args.dataset_root, args.workers, args.target_episodes)
        else:
            download_gcs_source(name, args.dataset_root, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
