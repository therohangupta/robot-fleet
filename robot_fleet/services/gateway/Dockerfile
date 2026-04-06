# Build from repo root: docker build -f services/gateway/Dockerfile -t gateway .
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
WORKDIR /app/services/gateway
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
