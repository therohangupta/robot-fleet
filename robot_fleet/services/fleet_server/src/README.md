# Fleet Server

Core orchestration service for multi-robot task planning, allocation, and execution. Runs as a gRPC server on port 50051.

## Components

### Planner
Generates task plans from high-level goals.

**Strategies:**
- **Monolithic** &mdash; single sequential plan for all goals.
- **Per-Goal DAG** &mdash; separate DAG per goal, then aggregated.
- **Big DAG** &mdash; single comprehensive DAG across multiple goals.

### Allocator
Assigns tasks from plans to specific robots.

**Strategies:**
- **LP** &mdash; linear programming optimization (load-balanced).
- **LLM** &mdash; OpenAI GPT-4 allocation (contextual reasoning).
- **Cost-Based** &mdash; capability-weighted assignment.

### Executor
Dispatches tasks to robots in dependency order (topological sort via Kahn's algorithm). Runs robots in parallel, respects DAG dependencies, and handles failure cascading (abort remaining tasks on non-replan failure).

## Running

```bash
# Bare-metal (from workspace root)
cd robot_fleet && python -m services.fleet_server.src -v

# Docker Compose (from robot_fleet/)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | see `packages/config.py` | PostgreSQL connection string |
| `OPENAI_API_KEY` | (from `.env`) | Required for LLM planners/allocators |
| `GATEWAY_EVENT_URL` | `http://localhost:8000/internal/events` | Where to POST state-change events |
| `DEFAULT_ROBOT_HOST` | `localhost` | Default host for reaching robot task servers |

## Strategy Selection Guide

| Scenario | Planning | Allocation |
|----------|----------|------------|
| Sequential goals | Monolithic | LP |
| Independent goals | DAG | LLM |
| Complex interdependencies | Big DAG | Cost-Based |
| Load balancing priority | Any | LP |
| Contextual reasoning | Any | LLM |
