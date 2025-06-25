#!/bin/bash
set -e

# Change to the project root directory
cd "$(dirname "$0")/../.."

# Rebuild Docker images for all example robots

echo "Building pick_place_robot..."
docker build -t pick_place_robot -f robot_fleet/robots/examples/pick_place/Dockerfile .

echo "Building nav_robot..."
docker build -t nav_robot -f robot_fleet/robots/examples/nav/Dockerfile .

echo "Building moma_robot..."
docker build -t moma_robot -f robot_fleet/robots/examples/moma/Dockerfile .

echo "All example robot Docker images rebuilt successfully." 