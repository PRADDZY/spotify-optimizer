#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: redis_restore.sh /path/to/backup.rdb"
  exit 1
fi

BACKUP_FILE="$1"
REDIS_DATA_DIR=${REDIS_DATA_DIR:-/var/lib/redis}
TARGET_FILE="$REDIS_DATA_DIR/dump.rdb"

if [ ! -f "$BACKUP_FILE" ]; then
  echo "Backup file not found: $BACKUP_FILE"
  exit 1
fi

sudo systemctl stop redis-server
sudo cp "$BACKUP_FILE" "$TARGET_FILE"
sudo chown redis:redis "$TARGET_FILE"
sudo systemctl start redis-server

echo "Restore complete from $BACKUP_FILE"
