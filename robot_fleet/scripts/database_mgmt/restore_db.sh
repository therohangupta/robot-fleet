#!/bin/bash
# Restore the Docker Postgres database from a backup file.
# Reads DB config from packages/config.py (single source of truth).
#
# Usage: ./restore_db.sh <backup_file.sql>
# Example: ./restore_db.sh ../db_backups/robot_fleet_20240123_143052.sql

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file.sql>" >&2
    echo "Example: $0 ../db_backups/robot_fleet_20240123_143052.sql" >&2
    exit 1
fi

BACKUP_FILE="$1"
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Extract DB config from Python config (single source of truth)
DB_HOST=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from packages.config import DB_HOST; print(DB_HOST)")
DB_PORT=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from packages.config import DB_PORT; print(DB_PORT)")
DB_USER=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from packages.config import DB_USER; print(DB_USER)")
DB_NAME=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from packages.config import DB_NAME; print(DB_NAME)")

echo "Restoring $DB_NAME on $DB_HOST:$DB_PORT from $BACKUP_FILE..."
# Use psql from the container to avoid version mismatch
docker compose exec -T db psql -U "$DB_USER" -d "$DB_NAME" < "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "Restore complete."
else
    echo "Restore failed!" >&2
    exit 1
fi
