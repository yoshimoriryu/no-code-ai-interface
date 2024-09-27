#!/bin/bash

# Ensure Alembic environment is initialized (if not already done)
if [ ! -d "alembic" ]; then
  echo "Initializing Alembic..."
  alembic init alembic
fi

# Upgrade the database to the latest version
echo "Running Alembic migrations..."
alembic upgrade head

# Run the main application
exec "$@"