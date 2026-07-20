#!/usr/bin/env bash
set -euo pipefail

exec .venv/bin/python scripts/download_vla_samples.py "$@"
