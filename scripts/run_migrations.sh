#!/bin/bash
set -e
# Ensure Alembic environment is initialized (if not already done)
if [ ! -d "alembic" ]; then
  echo "Initializing Alembic..."
  alembic init alembic
fi

# Upgrade the database to the latest version
echo "Running Alembic migrations..."
if alembic upgrade head; then
  echo "Migrations completed successfully."
else
  echo "Migrations failed!" >&2
  exit 1
fi

echo "Populating the database..."
python /app/scripts/populate_db.py

# Run the main application
exec "$@"