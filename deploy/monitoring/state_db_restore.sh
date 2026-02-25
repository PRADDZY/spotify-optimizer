#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <backup-file.db.gz>"
  exit 1
fi

BACKUP_FILE="$1"
STATE_DB_PATH="${STATE_DB_PATH:-/opt/spotify-optimizer/backend/data/state.db}"

if [[ ! -f "$BACKUP_FILE" ]]; then
  echo "Backup file not found: $BACKUP_FILE"
  exit 1
fi

mkdir -p "$(dirname "$STATE_DB_PATH")"
tmp_file="$(mktemp)"
gzip -cd "$BACKUP_FILE" > "$tmp_file"
cp "$tmp_file" "$STATE_DB_PATH"
rm -f "$tmp_file"

echo "State DB restored to: $STATE_DB_PATH"
