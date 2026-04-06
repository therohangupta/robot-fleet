# Build from repo root: docker build -f services/telemetry/Dockerfile -t telemetry .
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
EXPOSE 9000

CMD ["uvicorn", "services.telemetry.src.main:app", "--host", "0.0.0.0", "--port", "9000"]
