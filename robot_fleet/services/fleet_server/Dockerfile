# Build from repo root: docker build -f services/fleet_server/Dockerfile -t fleet-server .
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
EXPOSE 50051

CMD ["python", "-m", "services.fleet_server.src", "-v"]
