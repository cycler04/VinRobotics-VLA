#!/usr/bin/env python3
"""Validate the workspace agent infrastructure, optionally including code smoke checks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / ".agents"


def markdown_files() -> list[Path]:
    return [ROOT / "AGENTS.md", *sorted(AGENTS.rglob("*.md"))]


def validate_links(errors: list[str]) -> None:
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for file in markdown_files():
        text = file.read_text(encoding="utf-8")
        for target in link_re.findall(text):
            if "://" in target or target.startswith("#"):
                continue
            path = (file.parent / target.split("#", 1)[0]).resolve()
            if not path.exists():
                errors.append(f"broken link: {file.relative_to(ROOT)} -> {target}")


def validate_memory(errors: list[str]) -> None:
    memory = AGENTS / "memory"
    names: dict[str, Path] = {}
    for file in memory.glob("*.md"):
        text = file.read_text(encoding="utf-8")
        match = re.search(r"^name:\s*(.+?)\s*$", text, re.MULTILINE)
        if not match:
            errors.append(f"missing memory name: {file.relative_to(ROOT)}")
            continue
        name = match.group(1).strip("\"'")
        names[name] = file
        if file.stem != name:
            errors.append(f"memory name mismatch: {file.name} != {name}")

    for file in memory.glob("*.md"):
        text = file.read_text(encoding="utf-8")
        for target in re.findall(r"\[\[([^\]]+)\]\]", text):
            if target not in names:
                errors.append(f"broken wikilink: {file.name} -> [[{target}]]")


def validate_skills(errors: list[str]) -> None:
    for skill_dir in sorted((AGENTS / "skills").iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            errors.append(f"missing SKILL.md: {skill_dir.relative_to(ROOT)}")
            continue
        text = skill_file.read_text(encoding="utf-8")
        match = re.search(r"^name:\s*(.+?)\s*$", text, re.MULTILINE)
        if not match or match.group(1).strip("\"'") != skill_dir.name:
            errors.append(f"skill name mismatch: {skill_dir.relative_to(ROOT)}")
        if "description:" not in text.split("---", 2)[1]:
            errors.append(f"missing skill description: {skill_dir.relative_to(ROOT)}")


def validate_codex_config(errors: list[str]) -> None:
    files = [ROOT / ".codex" / "config.toml"]
    files.extend(sorted((ROOT / ".codex" / "agents").glob("*.toml")))
    for file in files:
        try:
            data = tomllib.loads(file.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"invalid TOML: {file.relative_to(ROOT)}: {exc}")
            continue
        if file.parent.name == "agents":
            for key in ("name", "description", "developer_instructions"):
                if not data.get(key):
                    errors.append(f"missing {key}: {file.relative_to(ROOT)}")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def validate_codex_discovery() -> None:
    command = ["codex", "debug", "prompt-input", "inspect dataset"]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    def strings(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)

    visible = "\n".join(strings(payload))
    required = ("# Hướng dẫn cho AI agent", "inspect-vla-dataset", "write-research-report")
    missing = [item for item in required if item not in visible]
    if missing:
        raise SystemExit(f"Codex discovery missing: {', '.join(missing)}")


def full_checks() -> None:
    python = ROOT / ".venv" / "bin" / "python"
    pytest = ROOT / ".venv" / "bin" / "pytest"
    if not python.exists() or not pytest.exists():
        raise SystemExit(".venv is missing; run the setup commands in .agents/04_commands.md")
    run([str(pytest), "-q"])
    run([str(python), "-m", "compileall", "-q", "src", "scripts", "tests"])
    run(["bash", "-n", "scripts/activate_vla_env.sh", "scripts/download_vla_sample.sh"])
    run([str(python), "-m", "vla_data_tools", "--help"])
    validate_codex_discovery()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also run code smoke checks")
    args = parser.parse_args()

    errors: list[str] = []
    validate_links(errors)
    validate_memory(errors)
    validate_skills(errors)
    validate_codex_config(errors)
    if (ROOT / ".claude").exists():
        errors.append("unexpected .claude directory")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print(
        f"OK: {len(markdown_files())} Markdown files, "
        f"{len(list((AGENTS / 'skills').glob('*/SKILL.md')))} skills, "
        f"{len(list((ROOT / '.codex' / 'agents').glob('*.toml')))} Codex agents"
    )
    if args.full:
        full_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
