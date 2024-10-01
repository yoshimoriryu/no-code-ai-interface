# Use the official Uvicorn-Gunicorn-FastAPI image with Python 3.10
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10


# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY ./app /app

# Set the working directory
WORKDIR /app

# Copy the migration script into the container
COPY scripts/run_migrations.sh /app/scripts/run_migrations.sh

# Make the migration script executable
RUN chmod +x /app/scripts/run_migrations.sh

EXPOSE 80

# Run migrations and then start the application with Gunicorn
ENTRYPOINT ["/app/scripts/run_migrations.sh"]
# CMD bash /app/scripts/run_migrations.sh
