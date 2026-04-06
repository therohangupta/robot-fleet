#!/bin/bash
set -e

# Rebuild Docker images for fake robots. Run from repo root (robot_fleet).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Building pick_place_robot..."
docker build -t pick_place_robot -f robots/fake/pick_place/Dockerfile .

echo "Building nav_robot..."
docker build -t nav_robot -f robots/fake/nav/Dockerfile .

echo "Building moma_robot..."
docker build -t moma_robot -f robots/fake/moma/Dockerfile .

echo "All fake robot Docker images rebuilt. Use run_examples_docker_dev.sh to run with bind mount (no rebuild needed for server.py changes)."
