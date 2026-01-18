#!/bin/bash
set -e

# Run all example robot containers as detached Docker containers with correct ports

# pick_place_robot: port 8003 (from pick_place.yaml)
echo "Running pick_place_robot on port 8003..."
docker run -d --rm -p 8003:8003 --name pick_place_robot pick_place_robot

# nav_robot: port 8002 (from nav.yaml)
echo "Running nav_robot on port 8002..."
docker run -d --rm -p 8002:8002 --name nav_robot nav_robot

# moma_robot: port 8001 (from moma.yaml)
echo "Running first moma_robot on port 8001..."
docker run -d --rm -p 8001:8001 --name moma_robot_1 moma_robot

# moma_robot: port 8004 (from moma.yaml)
echo "Running second moma_robot on port 8001..."
docker run -d --rm -p 8004:8001 --name moma_robot_2 moma_robot

echo "All example robot containers are running in detached mode." 