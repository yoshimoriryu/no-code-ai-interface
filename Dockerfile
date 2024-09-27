# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app


# Copy the migration script into the container
COPY scripts/run_migrations.sh /app/scripts/run_migrations.sh
RUN chmod +x /app/scripts/run_migrations.sh

# Expose the necessary ports
EXPOSE 8000

# Run migrations and start the application
CMD ["/app/scripts/run_migrations.sh", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
