#!/usr/bin/env bash
set -euo pipefail

REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}
BACKUP_DIR=${BACKUP_DIR:-/var/backups/redis}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$BACKUP_DIR/redis-$TIMESTAMP.rdb"

mkdir -p "$BACKUP_DIR"

redis-cli -u "$REDIS_URL" --rdb "$OUT_FILE"

echo "Backup written to $OUT_FILE"
