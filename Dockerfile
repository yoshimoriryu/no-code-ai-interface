FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

WORKDIR /app

COPY scripts/run_migrations.sh /app/scripts/run_migrations.sh

RUN chmod +x /app/scripts/run_migrations.sh

EXPOSE 80

ENTRYPOINT ["/app/scripts/run_migrations.sh"]
# CMD bash /app/scripts/run_migrations.sh
