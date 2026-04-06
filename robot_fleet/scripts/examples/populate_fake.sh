#!/bin/bash
set -e

echo "=== Robot Fleet Demo Population Script ==="
echo "This script registers example robots, sets up world state, and creates goals."
echo

# Robot host: host.docker.internal on Mac Docker Desktop resolves to 127.0.0.1
# from both the host and inside containers, so fleet-server can reach robots.
ROBOT_HOST="${ROBOT_HOST:-host.docker.internal}"

# Resolve repo root relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Wait a moment for containers to be fully ready
echo "Waiting for robot containers to be ready..."
sleep 3

echo "=== Registering Robots ==="

echo "Registering pick-place robot (kitchen countertop)..."
robotctl register "${REPO_ROOT}/robot_fleet/robots/fake/pick_place/pick_place.yaml" pick_place-1 --host "${ROBOT_HOST}" --port 8003

echo "Registering navigation robot (mobile with basket)..."
robotctl register "${REPO_ROOT}/robot_fleet/robots/fake/nav/nav.yaml" nav-1 --host "${ROBOT_HOST}" --port 8002

echo "Registering mobile manipulator robot 1..."
robotctl register "${REPO_ROOT}/robot_fleet/robots/fake/moma/moma.yaml" moma-1 --host "${ROBOT_HOST}" --port 8001

echo
echo "=== Setting Up World State ==="

echo "Adding world statements..."
robotctl world add "The house contains a kitchen with a countertop, toaster, sink with dirty dishes, and refrigerator; a dining room adjacent to the kitchen; and a living room."
robotctl world add "The pick-place robot is affixed to the kitchen countertop and can manipulate small objects."
robotctl world add "The navigation robot has a large basket and can move throughout the house."
robotctl world add "The mobile manipulator robot can both navigate and manipulate objects with its gripper."
robotctl world add "The pick-place robot is in the kitchen, while the navigation robot and mobile manipulator are in unknown locations around the house."

echo
echo "=== Creating Goals ==="

echo "Creating goals..."
robotctl goal add "prepare breakfast toast"
robotctl goal add "clean up dirty dishes"

echo
echo "=== Summary ==="
echo "Registered robots (host=${ROBOT_HOST}):"
echo "  - pick_place-1: Kitchen countertop robot (port 8003)"
echo "  - nav-1: Mobile navigation robot with basket (port 8002)"
echo "  - moma-1: Mobile manipulator robot (port 8001)"
echo
echo "World state set up with house layout and robot capabilities."
echo "Goals created: 'prepare breakfast toast' and 'clean up dirty dishes'"
echo
echo "Ready to create plans! Try:"
echo "  robotctl plan create dag llm 1,2"
echo "  robotctl plan create monolithic cost_based 1,2"
echo
echo "=== Demo Setup Complete ==="
