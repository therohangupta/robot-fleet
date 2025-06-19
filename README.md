# Robot Fleet Management System

A distributed system for managing a fleet of robots with PostgreSQL backend.

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL 14+
- Docker (for robot deployment)

### Python Dependencies

Install all required packages:

```bash
pip install grpc-tools protobuf sqlalchemy[asyncio] asyncpg docker pyyaml click mcp
```

### Database Setup

1. Install PostgreSQL:
```bash
# For macOS
brew install postgresql@14
brew services start postgresql@14

# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql-14
sudo systemctl start postgresql
```

2. Create the database and user:
```bash
# Connect to PostgreSQL
psql postgres

# In the PostgreSQL prompt:
CREATE DATABASE robot_fleet;
CREATE USER robot_user WITH PASSWORD 'secret';
GRANT ALL PRIVILEGES ON DATABASE robot_fleet TO robot_user;
\c robot_fleet
GRANT ALL ON SCHEMA public TO robot_user;
\q
```

3. Verify the connection:
```bash
psql -U robot_user -d robot_fleet -h localhost
# Enter password when prompted: secret
\q
```

## Running the Server

From the project root directory:

```bash
# Make sure to kill any existing server instances first
pkill -f "python3 -m robot_fleet.server"

# Start the server
python3 -m robot_fleet.server
```

The server will start on port 50051 by default. Check for successful initialization:
- The server should print "Starting Robot Fleet Management Server on port 50051"
- Initial database tables will be created automatically
- Test writes will confirm database access

## Building Docker Images for Robot Types

### Building the Pick-Place Robot Image

The example YAML configuration in `robot_fleet/robots/examples/pick_place/pick_place.yaml` references a Docker image `robotfleet/pick-place:1.0`. Here's how to build it:

1. Navigate to the example directory:

```bash
cd robot_fleet/robots/examples/pick_place
```

2. Build the Docker image from the project root:

```bash
# From the project root directory
docker build -t robotfleet/pick-place:1.0 -f robot_fleet/robots/examples/pick_place/Dockerfile .
```

3. Verify the image is built:
```bash
docker images | grep robotfleet
```

You should see the image `robotfleet/pick-place:1.0` in the list.

## Using the CLI

The `robotctl` command is the main interface for interacting with the robot fleet.

### Register a Robot

```bash
# Navigate to a directory with the robot config file
cd robot_fleet/robots/examples/pick_place

# Register a robot
robotctl register pick_place.yaml my-robot-1
```

### List Robots

```bash
# List all registered robots
robotctl list
```

### Check Robot Status

```bash
# Get the status of a specific robot
robotctl status my-robot-1
```

### Deploy a Robot

```bash
# Deploy a registered robot
robotctl deploy my-robot-1
```

### Undeploy a Robot

```bash
# Undeploy a robot
robotctl undeploy my-robot-1
```

### Unregister a Robot

```bash
# Unregister a robot
robotctl unregister my-robot-1
```

### Manage Goals

```bash
# Add a new goal
robotctl goal add "Move all boxes from warehouse A to B"

# List all goals
robotctl goal list

# Get details of a specific goal
robotctl goal get 1

# Delete a goal
robotctl goal delete 1
```

### Manage Tasks

```bash
# Add a new task (with optional goal association)
robotctl task add --robot-id robot1 "Move to coordinates (x,y)"
robotctl task add --robot-id robot1 --goal-id 1 "Pick up box from location A"

# List all tasks
robotctl task list

# List tasks for a specific robot
robotctl task list --robot-id robot1

# List tasks for a specific goal
robotctl task list --goal-id 1

# Get details of a specific task
robotctl task get 1

# Delete a task
robotctl task delete 1
```

## Troubleshooting

### Server Issues

If you encounter issues with the server:

1. Check for existing server processes:
```bash
lsof -i :50051
```

2. Kill any existing server processes:
```bash
pkill -f "python3 -m robot_fleet.server"
```

3. Check PostgreSQL connection:
```bash
psql -U robot_user -d robot_fleet -h localhost
```

4. Check server logs for detailed error messages

### Docker Issues

If deployment fails:

1. Verify Docker is running:
```bash
docker ps
```

2. Check if the image exists:
```bash
docker images | grep robotfleet
```

3. Check if the container is already running:
```bash
docker ps -a | grep my-robot-1
```

4. Make sure Docker daemon is accessible:
```bash
# For local Docker
docker info

# For remote Docker (specified in deployment config)
docker -H tcp://hostname:port info
```

### Database Reset

To completely reset the database:

```bash
psql postgres

# In PostgreSQL prompt:
DROP DATABASE robot_fleet;
CREATE DATABASE robot_fleet;
GRANT ALL PRIVILEGES ON DATABASE robot_fleet TO robot_user;
\c robot_fleet
GRANT ALL ON SCHEMA public TO robot_user;
\q
```


# current directory structure
/agents — for any LLM/VLM based agents that robots can run
/data — any kind of data, from episodes to vector databases
/models —  not sure yet but maybe stuff in between agents and end-to-end policies for skills
/robot_control — how robots communicate and have low-level control over their actions. Likely to implement ROS of some kind for each robot
/robots — a directory containing all the information for supported robots, will maintain information for each robot that allows the highest level of abstraction in all other modules
---- /bimanual
---- /locobot
---- /toyota_hri
/skills — a directory of policies trained on specific tasks for each robot and are tagged with relevant keywords for that skill
/utils — all random useful things

server.py - for running the central server
start_robots.py — for running each individual robot 