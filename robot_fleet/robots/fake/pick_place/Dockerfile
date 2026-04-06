# Build from repo root: docker build -f robots/fake/pick_place/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install the repo as editable so "packages" resolves via normal Python path (no PYTHONPATH hacks)
RUN pip install --no-cache-dir -e .

ENV ROBOT_ID=pick_place
ENV CONFIG_FILE=/app/robots/fake/pick_place/pick_place.yaml
ENV PORT=8003

EXPOSE 8003

CMD ["python", "/app/robots/fake/pick_place/server.py", "--config", "/app/robots/fake/pick_place/pick_place.yaml"]
