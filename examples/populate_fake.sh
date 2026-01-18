#!/bin/bash
set -e

echo "=== Robot Fleet Demo Population Script ==="
echo "This script registers example robots, sets up world state, and creates goals."
echo

# Wait a moment for containers to be fully ready
echo "Waiting for robot containers to be ready..."
sleep 3

echo "=== Registering Robots ==="

# Register pick-place robot (affixed to kitchen countertop)
echo "Registering pick-place robot (kitchen countertop)..."
robotctl register /Users/rohangupta/Desktop/Workspace/glamor/robot-fleet/robot_fleet/robots/examples/pick_place/pick_place.yaml pick_place-1 --host localhost --port 8003

# Register navigation robot (mobile with basket)
echo "Registering navigation robot (mobile with basket)..."
robotctl register /Users/rohangupta/Desktop/Workspace/glamor/robot-fleet/robot_fleet/robots/examples/nav/nav.yaml nav-1 --host localhost --port 8002

# Register mobile manipulator robots
echo "Registering mobile manipulator robot 1..."
robotctl register /Users/rohangupta/Desktop/Workspace/glamor/robot-fleet/robot_fleet/robots/examples/moma/moma.yaml moma-1 --host localhost --port 8001

echo
echo "=== Setting Up World State ==="

# Add world statements describing the environment and robot locations
echo "Adding world statements..."
robotctl world add "The house contains a kitchen with a countertop, toaster, sink with dirty dishes, and refrigerator; a dining room adjacent to the kitchen; and a living room."
robotctl world add "The pick-place robot is affixed to the kitchen countertop and can manipulate small objects."
robotctl world add "The navigation robot has a large basket and can move throughout the house."
robotctl world add "The mobile manipulator robot can both navigate and manipulate objects with its gripper."
robotctl world add "The pick-place robot is in the kitchen, while the navigation robot and mobile manipulator are in unknown locations around the house."

echo
echo "=== Creating Goals ==="

# Add the two goals for the demo
echo "Creating goals..."
robotctl goal add "prepare breakfast toast"
robotctl goal add "clean up dirty dishes"

echo
echo "=== Summary ==="
echo "Registered robots:"
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