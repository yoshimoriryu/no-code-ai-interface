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

# Populate the database
echo "Populating the database..."
python /app/scripts/populate_db.py

# Run the main application
if [ "$ENV" == "development" ]; then
    echo "Starting the application in development mode..."
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Starting the application in production mode..."
    exec gunicorn -k uvicorn.workers.UvicornWorker -w 1 --bind 0.0.0.0:8000 app.main:app --log-level info --access-logfile - --error-logfile -
fi