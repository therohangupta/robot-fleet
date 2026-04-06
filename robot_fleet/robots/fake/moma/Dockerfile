# Build from repo root: docker build -f robots/fake/moma/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

# Copy whole repo so bind-mount in dev can override without rebuild
COPY . /app

# Install the repo as editable so "packages" resolves via normal Python path (no PYTHONPATH hacks)
RUN pip install --no-cache-dir -e .

ENV ROBOT_ID=moma
ENV CONFIG_FILE=/app/robots/fake/moma/moma.yaml
ENV PORT=8001

EXPOSE 8001

CMD ["python", "/app/robots/fake/moma/server.py", "--config", "/app/robots/fake/moma/moma.yaml"]
