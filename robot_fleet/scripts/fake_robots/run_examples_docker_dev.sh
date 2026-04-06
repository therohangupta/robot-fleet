#!/bin/bash
set -e

# Dev-mode runner for fake robot containers (v2).
#
# Bind-mounts the repo into /app so edits to server.py do NOT require image rebuild.
# Restart the container after code changes: docker restart <container_name>
#
# Usage (from repo root robot_fleet):
#   ./scripts/fake_robots/rebuild_examples_docker.sh   # only when Dockerfile or deps change
#   ./scripts/fake_robots/run_examples_docker_dev.sh
#
# Robots push heartbeats to the Telemetry service (gateway reads health from Telemetry).
# Set TELEMETRY_URL if telemetry is not on host:9000:
#   -e TELEMETRY_URL=http://host.docker.internal:9000  (default from container)
#
# If robot containers don't show in 'docker ps', they may have exited. Check:
#   docker ps -a
#   docker logs pick_place_robot   # (or nav_robot, moma_robot_1, moma_robot_2)
# To run one robot in foreground: docker run --rm -p 8001:8001 -v "${ROOT_DIR}:/app" -e TELEMETRY_URL=http://host.docker.internal:9000 moma_robot

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Using repo bind mount: ${ROOT_DIR} -> /app"

# Telemetry URL: robots run in separate containers and reach telemetry via host.
# Default matches packages/config.py TELEMETRY_PORT (9000).
export TELEMETRY_URL="${TELEMETRY_URL:-http://host.docker.internal:9000}"

# No --rm: if a container exits, it stays so you can run 'docker logs <name>' to debug
# Health is matched by host:port (from heartbeat), so you can register with any name in the fleet.
echo "Running pick_place_robot on port 8003 (dev mount)..."
docker run -d \
  -p 8003:8003 \
  -v "${ROOT_DIR}:/app" \
  -e TELEMETRY_URL="${TELEMETRY_URL}" \
  --name pick_place_robot \
  pick_place_robot

echo "Running nav_robot on port 8002 (dev mount)..."
docker run -d \
  -p 8002:8002 \
  -v "${ROOT_DIR}:/app" \
  -e TELEMETRY_URL="${TELEMETRY_URL}" \
  --name nav_robot \
  nav_robot

echo "Running first moma_robot on port 8001 (dev mount)..."
docker run -d \
  -p 8001:8001 \
  -v "${ROOT_DIR}:/app" \
  -e TELEMETRY_URL="${TELEMETRY_URL}" \
  --name moma_robot_1 \
  moma_robot

# Second moma: host port from variable (container listens on 8001); TASK_SERVER_PORT so fleet registration at that port matches
MOMA_2_HOST_PORT=8004
echo "Running second moma on host port ${MOMA_2_HOST_PORT} (dev mount)..."
docker run -d \
  -p "${MOMA_2_HOST_PORT}:8001" \
  -v "${ROOT_DIR}:/app" \
  -e TELEMETRY_URL="${TELEMETRY_URL}" \
  -e TASK_SERVER_PORT="${MOMA_2_HOST_PORT}" \
  --name moma_robot_2 \
  moma_robot

echo "All example robot containers are running in detached mode (dev mount enabled)."
