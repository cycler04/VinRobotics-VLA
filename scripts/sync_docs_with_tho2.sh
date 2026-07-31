#!/usr/bin/env bash
# Sync docs/ between this workspace and tho2.

set -euo pipefail

readonly REMOTE_HOST="vinrobotics"
readonly REMOTE_ROOT="/home/tho2/Dung_Workspace/VinRobotics"
readonly LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/sync_docs_with_tho2.sh <pull|push> [--mirror] [--apply]

Examples:
  bash scripts/sync_docs_with_tho2.sh pull
  bash scripts/sync_docs_with_tho2.sh pull --apply
  bash scripts/sync_docs_with_tho2.sh push --mirror
  bash scripts/sync_docs_with_tho2.sh push --mirror --apply

Preview changes by default. Pass --apply to transfer files.
Only docs/ is included.

By default, files are added or updated without deleting destination-only files.
Pass --mirror to make the destination exactly match the source, including
deleting destination-only files and directories:
  pull --mirror: tho2 docs/ -> local docs/
  push --mirror: local docs/ -> tho2 docs/

Always preview a mirror operation before repeating it with --apply.
EOF
}

direction="${1:-}"

case "$direction" in
    pull|push) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
shift

apply=false
mirror=false
while (($#)); do
    case "$1" in
        --apply)
            apply=true
            ;;
        --mirror)
            mirror=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
    shift
done

rsync_options=(-a --itemize-changes --protect-args)
if ! "$apply"; then
    rsync_options+=(--dry-run)
fi
if "$mirror"; then
    # Delete only after a successful transfer so a partial failure does not
    # immediately remove destination-only files.
    rsync_options+=(--delete-delay)
fi

if [[ "$direction" == "pull" ]]; then
    rsync \
        "${rsync_options[@]}" \
        "$REMOTE_HOST:$REMOTE_ROOT/docs/" \
        "$LOCAL_ROOT/docs/"
else
    if "$apply"; then
        ssh -o BatchMode=yes "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/docs'"
    else
        ssh -o BatchMode=yes "$REMOTE_HOST" "test -d '$REMOTE_ROOT/docs'"
    fi
    rsync \
        "${rsync_options[@]}" \
        "$LOCAL_ROOT/docs/" \
        "$REMOTE_HOST:$REMOTE_ROOT/docs/"
fi
