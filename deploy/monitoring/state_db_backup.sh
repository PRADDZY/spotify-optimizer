#!/usr/bin/env bash
set -euo pipefail

STATE_DB_PATH="${STATE_DB_PATH:-/opt/spotify-optimizer/backend/data/state.db}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/spotify-optimizer}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"

mkdir -p "$BACKUP_DIR"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
backup_file="${BACKUP_DIR}/state-${timestamp}.db"

if [[ ! -f "$STATE_DB_PATH" ]]; then
  echo "State DB not found: $STATE_DB_PATH"
  exit 1
fi

cp "$STATE_DB_PATH" "$backup_file"
gzip -f "$backup_file"
find "$BACKUP_DIR" -type f -name "state-*.db.gz" -mtime +"$RETENTION_DAYS" -delete

echo "Backup written: ${backup_file}.gz"
