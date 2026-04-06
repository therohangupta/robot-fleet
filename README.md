<h1 align="center">RobotFleet: An Open-Source Framework for Centralized Multi-Robot Task Planning</h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2510.10379">Paper</a> &bull;
  <a href="https://youtu.be/1L4maFDmo-o">Video Overview + Demo</a> &bull;
  <a href="#quick-start-docker-compose">Quick Start</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#dashboard-frontend">Dashboard</a> &bull;
  <a href="#cli-reference">CLI</a>
</p>

RobotFleet is an open-source framework for centralized multi-robot task planning and scheduling. It coordinates heterogeneous fleets using modular LLM-based planning, dependency-aware DAG execution, and a real-time dashboard.

## Key Features

- **LLM-Driven Planning** &mdash; converts natural language goals into dependency-aware task DAGs.
- **Modular Architecture** &mdash; planners, allocators, and executors are all swappable.
- **Dependency-Aware Execution** &mdash; topological task ordering with parallel multi-robot dispatch.
- **Real-Time Dashboard** &mdash; React frontend with live WebSocket updates, DAG visualization, and execution monitoring.
- **Containerized Robots** &mdash; each robot runs as a Docker service for scalable fleet management.
- **Dynamic World State** &mdash; maintain and update a declarative world model in real-time.
- **Replanning** &mdash; react to execution failures and dynamically reallocate tasks.

## Architecture

RobotFleet is composed of five services:

| Service | Port | Role |
|---------|------|------|
| **Fleet Server** (gRPC) | 50051 | Orchestration: planning, allocation, DAG execution |
| **Gateway** (REST/WS) | 8000 | Client-facing API surface, real-time WebSocket fanout |
| **Telemetry** | 9000 | Robot heartbeat ingest, health monitoring |
| **Dashboard** (Vite/React) | 5173 | Web UI for fleet management, plan creation, execution monitoring |
| **PostgreSQL** | 5432 | Persistent storage for plans, tasks, robots, goals, world state |

The **Gateway** is the only client-facing API. The dashboard, CLI, and any external client all talk to the Gateway, which bridges to Fleet Server via gRPC and reads health from Telemetry.

**Execution flow:**
1. Goals are defined (natural language descriptions).
2. A **planner** (Big DAG, Per-Goal DAG, or Monolithic) converts goals into a task DAG.
3. An **allocator** (LLM-based or algorithmic) assigns tasks to robots.
4. The **executor** dispatches tasks in topological order, respecting dependencies, running robots in parallel.
5. The dashboard shows live progress via WebSocket.

![RobotFleet Diagram](/Diagram.png)

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Docker and Docker Compose
- PostgreSQL 14+ (for bare-metal; Docker Compose includes one)

## Quick Start (Docker Compose)

This is the recommended way to run RobotFleet. Docker Compose starts the database, fleet server, gateway, and telemetry. The frontend and fake robots run on the host.

### 1. Install the Python package (for CLI and robot scripts)

```bash
cd robot_fleet && pip install -e . && cd ..
```

### 2. Set up environment

Create a `.env` file at the repo root with your OpenAI key (required for LLM planners/allocators):

```
OPENAI_API_KEY=sk-...
```

### 3. Start backend services

```bash
cd robot_fleet
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

This starts PostgreSQL (port 5433 on host), Fleet Server (50051), Gateway (8000), and Telemetry (9000).

### 4. Start the frontend

In a separate terminal:

```bash
cd robot_fleet/services/dashboard-web
npm install
npm run dev
```

The dashboard is available at **http://localhost:5173**.

### 5. Start fake robots

In a separate terminal:

```bash
./robot_fleet/scripts/fake_robots/run_examples_docker.sh
```

### 6. Populate demo data

```bash
./robot_fleet/scripts/examples/populate_fake.sh
```

This registers 3 robots, sets up world state, and creates two goals ("prepare breakfast toast" and "clean up dirty dishes").

### 7. Create and execute a plan

Via the dashboard: navigate to **Plans**, click **Create Plan with AI**, select goals and methods, then execute.

Or via CLI:

```bash
robotctl plan create dag llm 1,2
robotctl plan start <plan_id>
```

## Dashboard Frontend

The dashboard at `robot_fleet/services/dashboard-web` is a React + TypeScript app (Vite, TailwindCSS, React Query, ReactFlow).

### Key Pages

- **Robots** &mdash; fleet overview, health status, send individual tasks, register/unregister robots.
- **Plans** &mdash; create plans (AI or manual), allocate tasks to robots, filter by status (unallocated/allocated/executing/completed/failed).
- **Execution Monitor** &mdash; live task progress with pills for completed/executing/pending/failed/cancelled, per-robot fleet table, task dependency DAG, event log. Retry failed plans with "Copy & Retry".
- **Goals** &mdash; manage natural language goals.
- **World** &mdash; manage world state statements.
- **Methods** &mdash; view available planners and allocators.

### Running the Frontend

```bash
cd robot_fleet/services/dashboard-web
npm install    # first time only
npm run dev    # starts on http://localhost:5173
```

The Vite dev server proxies `/api/*` and `/ws/*` to the Gateway at `localhost:8000`.

## Bare-Metal Setup (without Docker Compose)

If you prefer running services directly:

### Database Setup

```bash
# macOS
brew install postgresql@14 && brew services start postgresql@14

# Create database
psql postgres -c "CREATE DATABASE robot_fleet;"
psql postgres -c "CREATE USER robot_user WITH PASSWORD 'secret';"
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE robot_fleet TO robot_user;"
psql robot_fleet -c "GRANT ALL ON SCHEMA public TO robot_user;"
```

### Start Services (4 terminals)

```bash
# Terminal 1 – Fleet Server (gRPC)
cd robot_fleet && python -m services.fleet_server.src -v

# Terminal 2 – Telemetry
cd robot_fleet && python -m services.telemetry.src --port 9000

# Terminal 3 – Gateway
cd robot_fleet/services/gateway && uvicorn src.main:app --reload --port 8000

# Terminal 4 – Frontend
cd robot_fleet/services/dashboard-web && npm run dev
```

## Examples

Demo scripts live in `robot_fleet/scripts/examples/`.

### populate_fake.sh

Registers 3 fake robots, sets up world state, and creates goals:

```bash
./robot_fleet/scripts/examples/populate_fake.sh
```

| Robot | Type | Port | Description |
|-------|------|------|-------------|
| moma-1 | Mobile Manipulator | 8001 | Navigate and manipulate objects |
| nav-1 | Navigation | 8002 | Mobile with basket |
| pick_place-1 | Pick & Place | 8003 | Kitchen countertop manipulator |

After populating, create plans via the dashboard or CLI.

## Fake Robot Docker Containers

```bash
# Build images (first time or after Dockerfile changes)
./robot_fleet/scripts/fake_robots/rebuild_examples_docker.sh

# Run containers
./robot_fleet/scripts/fake_robots/run_examples_docker.sh

# Run with bind mounts (dev mode — code changes without rebuild)
./robot_fleet/scripts/fake_robots/run_examples_docker_dev.sh

# Stop containers
docker stop pick_place_robot nav_robot moma_robot
```

## Configuration

All shared configuration (ports, hosts, URLs) lives in `robot_fleet/packages/config.py` as the single source of truth. Docker Compose overrides values via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet` | Database connection |
| `GRPC_SERVER_PORT` | `50051` | Fleet Server gRPC port |
| `GATEWAY_PORT` | `8000` | Gateway HTTP port |
| `TELEMETRY_PORT` | `9000` | Telemetry service port |
| `DEFAULT_ROBOT_HOST` | `localhost` | Default host for robot task servers |
| `OPENAI_API_KEY` | (from `.env`) | Required for LLM planners/allocators |

## CLI Reference

The `robotctl` CLI communicates with the Gateway.

### Robots

```bash
robotctl register <yaml_path> <robot_id> [--host HOST] [--port PORT]
robotctl list
robotctl unregister <robot_id>
```

### Plans

```bash
robotctl plan create <strategy> <allocator> <goal_ids>   # e.g. dag llm 1,2
robotctl plan list
robotctl plan get <id> [--verbose] [--analyze-idle]
robotctl plan start <id>
robotctl plan delete <id>
```

### Goals

```bash
robotctl goal add "<description>"
robotctl goal list
robotctl goal get <id> [--verbose]
robotctl goal delete <id>
```

### Tasks

```bash
robotctl task add "<description>" --goal-id <id> --plan-id <id> --robot-id <id> --robot-type <type>
robotctl task list [--robot-id <id>] [--goal-id <id>]
robotctl task get <id> [--verbose]
robotctl task delete <id>
```

### World State

```bash
robotctl world add "<statement>"
robotctl world list
robotctl world get <id>
robotctl world delete <id>
```

### Database

```bash
# Reset database (destroys all data)
cd robot_fleet && python -m services.fleet_server.src --reset-db
```

## Project Structure

```
robot_fleet/
├── packages/              # Shared libraries
│   ├── config.py          # Single source of truth for all configuration
│   ├── metrics.py         # Structured observability
│   ├── proto/             # gRPC protobuf definitions
│   ├── fleet_sdk/         # DB models, instance registry (used by services)
│   ├── robot_sdk/         # Robot client for task dispatch
│   └── client_sdk/        # TypeScript + Python SDKs for Gateway API
├── services/
│   ├── fleet_server/      # gRPC orchestration: planning, allocation, execution
│   ├── gateway/           # REST/WebSocket API (client-facing)
│   ├── telemetry/         # Robot health monitoring
│   └── dashboard-web/     # React frontend
├── robots/
│   ├── fake/              # Simulated robots for testing (moma, nav, pick_place)
│   └── real/              # Real robot integrations
├── scripts/
│   ├── examples/          # Demo population scripts
│   ├── fake_robots/       # Docker build/run scripts for fake robots
│   ├── database_mgmt/     # DB backup/restore scripts
│   └── grpc_gen.sh        # Regenerate protobuf stubs
├── docs/                  # Architecture and design documentation
├── cli/                   # robotctl CLI
├── docker-compose.yml     # Production Docker Compose
└── docker-compose.dev.yml # Dev overlay (bind mounts, live reload)
```

## License

See [LICENSE](LICENSE) for details.
