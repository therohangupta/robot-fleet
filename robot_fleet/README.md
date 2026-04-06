# robot_fleet

Canonical Python package for multi-robot fleet orchestration. This directory contains all services, shared libraries, robots, scripts, and the CLI.

## Components

### Services

| Service | Directory | Port | Role |
|---------|-----------|------|------|
| **Fleet Server** | `services/fleet_server/` | 50051 (gRPC) | Control plane: planning, allocation, DAG execution, DB truth |
| **Gateway** | `services/gateway/` | 8000 (REST/WS) | Client-facing API surface, gRPC bridge to fleet, WebSocket fanout |
| **Telemetry** | `services/telemetry/` | 9000 (HTTP) | Robot heartbeat ingest, health monitoring, event push to gateway |
| **Dashboard** | `services/dashboard-web/` | 5173 (Vite) | React frontend: fleet management, plan creation, execution monitoring, DAG visualization |

### Shared Packages

| Package | Path | Purpose |
|---------|------|---------|
| **config** | `packages/config.py` | Single source of truth for all ports, hosts, and URLs |
| **metrics** | `packages/metrics.py` | Structured observability and logging |
| **fleet_sdk** | `packages/fleet_sdk/` | SQLAlchemy models, instance registry (DB operations) |
| **robot_sdk** | `packages/robot_sdk/` | Robot client for task dispatch, server base class, YAML schema |
| **client_sdk** | `packages/client_sdk/` | TypeScript SDK (used by dashboard) and Python SDK for Gateway API |
| **proto** | `packages/proto/` | gRPC protobuf definitions and generated stubs |

### Robots

| Directory | Description |
|-----------|-------------|
| `robots/fake/` | Simulated robots for testing: `moma` (8001), `nav` (8002), `pick_place` (8003) |
| `robots/real/` | Real robot integrations: `hsr`, `locobot` |

Each robot directory contains a YAML config, Dockerfile, server implementation, and tools.

### CLI

`cli/robotctl.py` &mdash; command-line interface for fleet operations (register robots, create plans, manage goals/tasks/world state).

## Quick Start

### Docker Compose (recommended)

From this directory (`robot_fleet/`):

```bash
# Backend services (fleet server, gateway, telemetry, postgres)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Frontend (separate terminal)
cd services/dashboard-web && npm install && npm run dev

# Fake robots (separate terminal, from repo root)
./robot_fleet/scripts/fake_robots/run_examples_docker.sh

# Demo data (separate terminal, from repo root)
./robot_fleet/scripts/examples/populate_fake.sh
```

### Bare-metal (4 terminals)

```bash
# Install package (one-time, from this directory)
pip install -e .

# Terminal 1 – Fleet Server (gRPC)
python -m services.fleet_server.src -v

# Terminal 2 – Telemetry
python -m services.telemetry.src --port 9000

# Terminal 3 – Gateway
cd services/gateway && uvicorn src.main:app --reload --port 8000

# Terminal 4 – Frontend
cd services/dashboard-web && npm install && npm run dev
```

Dashboard: http://localhost:5173 | Gateway: http://localhost:8000 | Telemetry: http://localhost:9000

## Architecture

```
Browser ──REST/WS──► Gateway ──gRPC──► Fleet Server ──► Postgres
                        │                    │
                        │ HTTP query         │ HTTP /do_task
                        ▼                    ▼
                    Telemetry ◄─heartbeat── Robots
```

- **Gateway** is the only client-facing API. The dashboard, CLI, and any external client talk exclusively to the Gateway.
- **Fleet Server** orchestrates plans, dispatches tasks to robots, and owns DB truth.
- **Telemetry** ingests robot heartbeats and pushes health-change events to the Gateway for real-time UI updates.
- **Robots** expose `/do_task` and `/health` endpoints. They push heartbeats to Telemetry.

### Event-Driven Updates

State changes flow from Fleet Server to Gateway to frontend without polling:
1. Fleet Server mutations trigger HTTP POST to `GATEWAY_EVENT_URL` (via `services/fleet_server/src/events.py`).
2. Gateway's WebSocket handler (`services/gateway/src/routers/websocket.py`) broadcasts invalidation signals to connected clients via per-subscriber event queues.
3. Frontend React Query hooks receive invalidation and refetch only when data actually changes.

### Execution Flow

1. Goals are defined (natural language descriptions).
2. A **planner** (Big DAG, Per-Goal DAG, or Monolithic) converts goals into a task DAG.
3. An **allocator** (LLM, LP, or Cost-Based) assigns tasks to robots.
4. The **executor** dispatches tasks in topological order (Kahn's algorithm), running robots in parallel while respecting DAG dependencies.
5. On non-replan failure, remaining tasks are cancelled and the plan is marked failed.

## Configuration

All shared configuration lives in `packages/config.py`. Docker Compose overrides values via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet` | Database connection |
| `GRPC_SERVER_PORT` | `50051` | Fleet Server gRPC port |
| `GATEWAY_PORT` | `8000` | Gateway HTTP port |
| `TELEMETRY_PORT` | `9000` | Telemetry service port |
| `GATEWAY_EVENT_URL` | `http://localhost:8000/internal/events` | Where fleet/telemetry POST state-change events |
| `DEFAULT_ROBOT_HOST` | `localhost` | Default host for robot task servers |
| `OPENAI_API_KEY` | (from `.env`) | Required for LLM planners/allocators |

## Directory Structure

```
robot_fleet/
├── cli/                          # robotctl CLI
│   ├── robotctl.py
│   └── printer.py
├── docs/                         # Architecture and design documentation
│   ├── COMPONENT_FLOWS.md        # Who talks to whom: gateway, fleet, robots
│   ├── COMMUNICATION_REVIEW.md   # Polling vs event-driven analysis
│   ├── DESIGN.md                 # Architecture decisions and goals
│   ├── GATEWAY_VS_TELEMETRY_SPLIT.md  # BFF vs telemetry separation
│   ├── IMPLEMENTATION_PLAN.md    # Phased implementation roadmap
│   ├── OPENAPI_CONTRACT.md       # OpenAPI schema notes
│   ├── REPO_LAYOUT.md            # Full directory inventory and responsibilities
│   ├── REQUIRED_TO_REMOVE_POLLING.md  # Event-driven migration checklist
│   ├── RUN.md                    # Step-by-step run and test guide
│   ├── TELEMETRY_AND_DATA_FLOW.md     # Telemetry pipeline design
│   ├── TELEMETRY_STORE_AND_EVENTS.md  # Telemetry storage patterns
│   └── TODO.md                   # Deferred work items
├── packages/
│   ├── config.py                 # Centralized configuration (ports, hosts, URLs)
│   ├── metrics.py                # Structured observability
│   ├── proto/                    # gRPC protobuf definitions
│   ├── fleet_sdk/                # DB models, instance registry
│   ├── robot_sdk/                # Robot client, server base, YAML schema
│   └── client_sdk/               # TypeScript + Python SDKs for Gateway
├── services/
│   ├── fleet_server/             # gRPC orchestration server
│   │   └── src/
│   │       ├── service.py        # gRPC service implementation
│   │       ├── events.py         # Event emission to gateway
│   │       ├── executor/         # DAG execution engine
│   │       ├── planners/         # Planning strategies (dag, big_dag, monolithic)
│   │       └── allocators/       # Allocation strategies (lp, llm, cost_based)
│   ├── gateway/                  # REST/WebSocket API
│   │   └── src/
│   │       ├── app.py            # FastAPI application
│   │       ├── grpc_bridge.py    # Fleet Server gRPC client
│   │       └── routers/          # API routes (plans, robots, tasks, websocket, etc.)
│   ├── telemetry/                # Health monitoring service
│   │   └── src/
│   │       ├── app.py            # FastAPI application
│   │       ├── heartbeat_store.py # Per-robot heartbeat ring buffer
│   │       └── publishing.py     # Event push to gateway
│   └── dashboard-web/            # React + TypeScript frontend
│       └── src/
│           ├── pages/            # Plans, Execution, PlanDetails, Robots, Goals, World, etc.
│           ├── components/       # Reusable UI (DAGVisualization, StatusBadge, modals)
│           └── lib/              # API client, utilities
├── robots/
│   ├── fake/                     # Simulated robots (moma, nav, pick_place)
│   └── real/                     # Real robot integrations (hsr, locobot)
├── scripts/
│   ├── fake_robots/              # Docker build/run scripts for fake robots
│   ├── examples/                 # Demo population scripts
│   ├── database_mgmt/            # DB backup/restore
│   └── grpc_gen.sh               # Regenerate protobuf stubs
├── docker-compose.yml            # Production Docker Compose
└── docker-compose.dev.yml        # Dev overlay (bind mounts, live reload)
```

## Documentation

All architecture and design docs are in `docs/`. Key references:

| Document | Description |
|----------|-------------|
| [docs/RUN.md](docs/RUN.md) | Step-by-step guide to running and testing all services |
| [docs/COMPONENT_FLOWS.md](docs/COMPONENT_FLOWS.md) | Data flow diagrams: who talks to whom |
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture goals and decisions |
| [docs/REPO_LAYOUT.md](docs/REPO_LAYOUT.md) | Full directory inventory with responsibilities |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phased roadmap with implementation status |
