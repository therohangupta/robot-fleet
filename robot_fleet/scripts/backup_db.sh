#!/bin/bash
# Backup the Docker Postgres database.
# Reads DB config from packages/config.py (single source of truth).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Extract DB config from Python config (single source of truth)
DB_HOST=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from packages.config import DB_HOST; print(DB_HOST)")
DB_PORT=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from packages.config import DB_PORT; print(DB_PORT)")
DB_USER=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from packages.config import DB_USER; print(DB_USER)")
DB_NAME=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from packages.config import DB_NAME; print(DB_NAME)")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$REPO_ROOT/db_backups"
mkdir -p "$BACKUP_DIR"

echo "Backing up $DB_NAME from $DB_HOST:$DB_PORT..."
pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" > "$BACKUP_DIR/${DB_NAME}_$TIMESTAMP.sql"

if [ $? -eq 0 ]; then
    echo "Backed up to $BACKUP_DIR/${DB_NAME}_$TIMESTAMP.sql"
else
    echo "Backup failed!" >&2
    exit 1
fi
