# RobotFleet: An Open-Source Framework for Centralized Multi-Robot Task Planning

RobotFleet is an open-source framework for centralized multi-robot task planning and scheduling, designed to coordinate heterogeneous fleets using modular components and LLM-based planning. Whether you're working with mobile manipulators, navigation robots, or custom agents, RobotFleet helps you scale multi-robot operations with ease.


## Key Features

- Centralized Task Planning with support for multi-goal missions.
- LLM-Driven Planning: Converts natural language goals into dependency-aware task plans.
- Modular Architecture: Planner, allocator, and executors are all easily swappable.
- Containerized Robots: Deploy each robot as a Docker service for scalable fleet management.
- Dynamic World State: Maintain and update a declarative world model in real-time.
- Supports Replanning: React to execution feedback and dynamically reallocate tasks.

## Architecture
RobotFleet consists of three major components:

1. Task Planner: 
Converts high-level goals into task DAGs using different LLM prompting strategies:
- Monolithic Prompt
- Big DAG
- Per-Goal DAG

2. Task Allocator: 
Assigns tasks within DAGs to robots using:
- LLM-based reasoning
- Mixed-Integer Linear Programming (MILP)

3. Task Status and Schedule Manager:
The actual management of the task plan/schedule that sends natural language commands to each robot and maintains the status of each robot. 

4. On-Robot Task Executors
A standard set of functions that can run on robots to execute the tasks they are given. Effectively, a task-to-action module that returns statuses of task successes and failures back to the central planner. See `robots/demo/README.md` for how different robots implement this.

![RobotFleet Diagram](/Diagram.png)


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

start a venv and make sure all requirements are installed 
```bash
source venv/bin/activate

pip install -r requirements

# Start the server
python3 -m robot_fleet.server
```

The server will start on port 50051 by default. Check for successful initialization:
- The server should print "Starting Robot Fleet Management Server on port 50051"
- Initial database tables will be created automatically

## Building and Running Example Robot Docker Containers

To build and run all example robot containers for testing and development, use the provided scripts. This is the recommended way to set up the example robots.

### 1. Make Scripts Executable

Before running the scripts, ensure they are executable:

```bash
chmod +x robot_fleet/scripts/*
```

### 2. Build All Example Robot Docker Images

Run the following script to build Docker images for all example robots (pick_place, nav, moma):

```bash
robot_fleet/scripts/rebuild_examples_docker.sh
```

This will build and tag each image appropriately for use with the fleet manager.

### 3. Start All Example Robot Containers

To launch all example robot containers in detached mode, run:

```bash
robot_fleet/scripts/run_examples_docker.sh
```

This will start each robot container and map the appropriate ports as defined in their configuration files.

### 4. Stopping Example Containers

To stop the running example containers, you can use:

```bash
docker stop pick_place_robot nav_robot moma_robot
```

Or stop all running containers (use with caution):

```bash
docker stop $(docker ps -q)
```

## Using the CLI

The `robotctl` command is the main interface for interacting with the robot fleet.

**For a full CLI reference, see [`robot_fleet/cli/ROBOTCTL_CLI.md`](robot_fleet/cli/ROBOTCTL_CLI.md).**

### Register a Robot (Pick Place)

```bash
# Navigate to a directory with the robot config file
cd robot_fleet/robots/examples/pick_place

# Register a robot
robotctl register pick_place.yaml pick_place
# Or bulk register
robotctl register pick_place.yaml --num 3
```

### Create Plans

Once you have the docker containers running in the example, and all 3 robots are registered, you can start to generate plans with the robots and the world state items.

Here is an example if you had two goal's with goal ids 1 and 2, and you wanted to use the dag planning strategy and llm allocation strategy

```bash
robotctl plan create dag llm 1,2
```

### Start Plans

Run the following command with the appropriate plan id, and the plan will start executing and sending the natural language tasks one after another to the robot servers, waiting on dependencies.

```bash
robotctl plan start 1
```

### Other CLI Actions
### List Robots

```bash
robotctl list
```

### Unregister a Robot

```bash
robotctl unregister pick_place
```

### Manage Goals

```bash
robotctl goal add "Move all boxes from warehouse A to B"
robotctl goal list
robotctl goal get 1 --verbose
robotctl goal delete 1
```

### Manage Tasks

```bash
robotctl task add "Move to coordinates (x,y)" --goal-id 1 --plan-id 1 --robot-id robot1 --robot-type nav
robotctl task list
robotctl task list --robot-id robot1
robotctl task list --goal-id 1
robotctl task get 1 --verbose
robotctl task delete 1
```


### Manage World Statements

```bash
robotctl world add "Box A is at location X"
robotctl world list
robotctl world get 123
robotctl world delete 123
```

### Manage Plans

```bash
robotctl plan create dag llm 1,2,3
robotctl plan list
robotctl plan get 1 --verbose
robotctl plan get 1 --analyze-idle
robotctl plan delete 1
```

### Database Reset

To completely reset the database, restart the server like this:

```bash
python3 -m robot_fleet.server --reset-db
```