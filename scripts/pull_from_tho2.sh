#!/usr/bin/env bash
# Sync code from tho2 to this workspace. Only src/ and scripts/ are transferred.

set -euo pipefail

readonly REMOTE_HOST="vinrobotics"
readonly REMOTE_ROOT="/home/tho2/Dung_Workspace/VinRobotics"
readonly LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/pull_from_tho2.sh [--apply]

Preview changes from SSH host `vinrobotics` (user tho2) by default.
Pass --apply to transfer files.
Only src/ and scripts/ are included. This script never deletes files.
EOF
}

apply=false
case "${1:-}" in
    "") ;;
    --apply) apply=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

rsync_options=(
    -a
    --itemize-changes
    --protect-args
    --exclude="__pycache__/"
    --exclude="*.py[cod]"
)
if ! "$apply"; then
    rsync_options+=(--dry-run)
fi

rsync "${rsync_options[@]}" "$REMOTE_HOST:$REMOTE_ROOT/src/" "$LOCAL_ROOT/src/"
rsync "${rsync_options[@]}" "$REMOTE_HOST:$REMOTE_ROOT/scripts/" "$LOCAL_ROOT/scripts/"
