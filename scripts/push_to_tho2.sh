#!/usr/bin/env bash
# Sync this workspace's code to tho2. Only src/ and scripts/ are transferred.

set -euo pipefail

readonly REMOTE_HOST="vinrobotics"
readonly REMOTE_ROOT="/home/tho2/Dung_Workspace/VinRobotics"
readonly LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/push_to_tho2.sh [--apply]

Preview changes on SSH host `vinrobotics` (user tho2) by default.
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

if "$apply"; then
    ssh -o BatchMode=yes "$REMOTE_HOST" \
        "mkdir -p '$REMOTE_ROOT/src' '$REMOTE_ROOT/scripts'"
else
    ssh -o BatchMode=yes "$REMOTE_HOST" \
        "test -d '$REMOTE_ROOT/src' && test -d '$REMOTE_ROOT/scripts'"
fi
rsync "${rsync_options[@]}" "$LOCAL_ROOT/src/" "$REMOTE_HOST:$REMOTE_ROOT/src/"
rsync "${rsync_options[@]}" "$LOCAL_ROOT/scripts/" "$REMOTE_HOST:$REMOTE_ROOT/scripts/"
