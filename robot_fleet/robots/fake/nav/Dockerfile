# Build from repo root: docker build -f robots/fake/nav/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install the repo as editable so "packages" resolves via normal Python path (no PYTHONPATH hacks)
RUN pip install --no-cache-dir -e .

ENV ROBOT_ID=nav
ENV CONFIG_FILE=/app/robots/fake/nav/nav.yaml
ENV PORT=8002

EXPOSE 8002

CMD ["python", "/app/robots/fake/nav/server.py", "--config", "/app/robots/fake/nav/nav.yaml"]
