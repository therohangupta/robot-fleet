# robot_fleet (canonical repo)

This folder is the **canonical robot_fleet package** that reflects component ownership:

- **Fleet Server (control plane)** &mdash; orchestration, DB truth, DAG execution, robot command/control.
- **Gateway/BFF (client-facing edge)** &mdash; one API surface for all clients (web/mobile/CLI), real-time WebSocket fanout, telemetry proxying.
- **Dashboard (React frontend)** &mdash; fleet management UI with plan creation, execution monitoring, DAG visualization.
- **Robots** &mdash; per-robot bundles that co-locate YAML config + Dockerfile + server code.
- **Packages** &mdash; shared libraries used by multiple services/robots (DB models, protos, SDKs, config).
- **Scripts** &mdash; build/run helpers for fake robots, database management, and demo population.
- **CLI** &mdash; `robotctl` command-line interface for fleet operations.

See `REPO_LAYOUT.md` for the full directory inventory and responsibilities.

### Running services

See **[docs/RUN.md](docs/RUN.md)** for step-by-step commands to start the fleet server, gateway, telemetry, and frontend, and how to test them.

**Quick start (Docker Compose + host frontend):**

```bash
# Backend services (from robot_fleet/)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Frontend (separate terminal, from robot_fleet/services/dashboard-web/)
npm install && npm run dev

# Fake robots (separate terminal, from repo root)
./robot_fleet/scripts/fake_robots/run_examples_docker.sh

# Demo data (separate terminal, from repo root)
./robot_fleet/scripts/examples/populate_fake.sh
```

Dashboard: http://localhost:5173 | Gateway: http://localhost:8000 | Telemetry: http://localhost:9000
