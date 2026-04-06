# How to run and test robot_fleet

Everything runs from the **robot_fleet** directory (this package). No old_robot_fleet.

**Layout:** Your workspace root is the directory that **contains** the `robot_fleet/` folder. Paths below use that convention.

**Postgres:** Defaults for bare-metal Python services match `packages/config.py`: host `localhost`, port **5432**, database `robot_fleet`, user `robot_user`, password `secret`. Override with `DATABASE_URL` or `DB_*` env vars. **Docker Compose** maps the container to the host as **localhost:5433** → container `5432`, so if only the DB runs in Compose and the stack runs on the host, set `DB_PORT=5433` (or a full `DATABASE_URL` using port 5433).

Optional: for Compose, `docker-compose.yml` loads `../.env` (the directory **above** `robot_fleet/`, i.e. the same workspace root that contains `robot_fleet`) for extra env on **fleet-server**.

You need **four terminals** for bare-metal dev (or use Docker Compose). From the **workspace root** (the directory that contains `robot_fleet`):

---

## 1. Install the repo (one-time)

```bash
cd robot_fleet
pip install -e .
cd ..
```

This editable install exposes the `packages`, `services`, and `cli` packages (including the `robotctl` entry point).

---

## 2. Terminal 1 – Fleet server (gRPC, port 50051)

```bash
cd robot_fleet
python -m services.fleet_server.src -v
```

Leave it running. It uses the same DB as the gateway.

---

## 3. Terminal 2 – Telemetry service (heartbeat ingest, port 9000)

```bash
cd robot_fleet
python -m services.telemetry.src --port 9000
```

Leave it running. Robots POST heartbeats here; the gateway queries it for health.

---

## 4. Terminal 3 – Gateway (REST + WebSocket, port 8000)

```bash
cd robot_fleet/services/gateway
uvicorn src.main:app --reload --port 8000
```

Leave it running. The Vite dev server proxies `/api` and `/ws` to this port (see `services/dashboard-web/vite.config.ts`).

---

## 5. Terminal 4 – Frontend (Vite, port 5173)

```bash
cd robot_fleet/services/dashboard-web
npm install   # once per clone
npm run dev
```

Open **http://localhost:5173** in the browser.

---

## Fake robots and demo data (optional)

From the **workspace root** (directory containing `robot_fleet`):

- **Fake example robots (Docker):** `./robot_fleet/scripts/fake_robots/run_examples_docker.sh`  
  Build images first when needed: `./robot_fleet/scripts/fake_robots/rebuild_examples_docker.sh`  
  For bind-mounted robot code during dev, use `./robot_fleet/scripts/fake_robots/run_examples_docker_dev.sh` instead (see `robots/fake/README.md`).
- **Populate demo DB data:** `./robot_fleet/scripts/examples/populate_fake.sh`

---

## Quick check

- **Gateway**: `curl -s http://localhost:8000/health` → `{"status":"healthy",...}`
- **Telemetry**: `curl -s http://localhost:9000/healthz` → `{"status":"ok"}`
- **Frontend**: http://localhost:5173 loads the dashboard.

---

## Summary

| Component     | Command (from workspace root)                                                                 | Port  |
|--------------|-------------------------------------------------------------------------------------------------|-------|
| Fleet server  | `cd robot_fleet && python -m services.fleet_server.src -v`                                   | 50051 |
| Telemetry     | `cd robot_fleet && python -m services.telemetry.src --port 9000`                             | 9000  |
| Gateway       | `cd robot_fleet/services/gateway && uvicorn src.main:app --reload --port 8000`               | 8000  |
| Frontend      | `cd robot_fleet/services/dashboard-web && npm run dev`                                       | 5173  |

All four run against the **robot_fleet** tree; use `pip install -e .` from `robot_fleet/` first for Python imports and `robotctl`.

---

## Docker (scalable)

Compose files live in **`robot_fleet/`**: `docker-compose.yml` and `docker-compose.dev.yml`.

From the **workspace root**:

```bash
cd robot_fleet
docker compose up --build
```

This starts **Postgres**, **fleet-server** (gRPC 50051), **telemetry** (HTTP 9000), and **gateway** (HTTP 8000) in separate containers. They use the Compose network (e.g. `gateway:8000`, `telemetry:9000`, `fleet-server:50051`, `db:5432` inside the network).

- **Gateway**: http://localhost:8000 (e.g. `curl http://localhost:8000/health`)
- **Telemetry**: http://localhost:9000 (e.g. `curl http://localhost:9000/healthz`)
- **Fleet gRPC**: localhost:50051 (for CLI/scripts)
- **DB**: Postgres is reachable from the **host** at **localhost:5433** (maps to `5432` in the container). Data uses the persistent Docker volume `db_data`.

Run the **frontend** on the host for dev: `cd robot_fleet/services/dashboard-web && npm install && npm run dev`, then open http://localhost:5173 (proxies to the gateway on port 8000).

---

## Importing existing data into the Docker Postgres (one-time)

The containerized Postgres uses a **persistent Docker volume** (`db_data`). Data survives container restarts. It starts empty until you migrate or import.

To import from a Postgres instance on your **host** (typical bare-metal port **5432**) into the **Compose** database (**5433** on the host):

```bash
# 1. Make sure the db container is running
cd robot_fleet
docker compose up db -d

# 2. Export from your HOST Postgres (adjust user/db to match your local instance)
pg_dump -h localhost -p 5432 -U YOUR_HOST_USER -d robot_fleet > /tmp/robot_fleet_dump.sql

# 3. Import into the CONTAINER Postgres (robot_user @ host port 5433)
psql -h localhost -p 5433 -U robot_user -d robot_fleet < /tmp/robot_fleet_dump.sql
#    Password: secret (default in docker-compose.yml / packages/config.py)

# 4. Verify
psql -h localhost -p 5433 -U robot_user -d robot_fleet -c "SELECT COUNT(*) FROM goals;"
```

After this, the container DB holds the imported data, and changes persist in the `db_data` volume.

**To completely reset the container DB** (start fresh):

```bash
cd robot_fleet
docker compose down -v   # -v removes volumes
docker compose up --build
```

---

## What to test end-to-end

- **DB**: Compose starts Postgres; fleet-server and gateway connect to it. No extra setup if you use the built-in `db` service.
- **Fleet server**: gRPC on 50051. From the host you can use the CLI: `robotctl world list`, `robotctl robots list`, etc. (requires `pip install -e .` from `robot_fleet/`; point at `localhost:50051` or use gateway-backed flows as appropriate.)
- **Telemetry service**: HTTP on 9000. Robots (or scripts) POST heartbeats here; the gateway queries for health.
- **Gateway**: HTTP on 8000. Try:
  - `curl -s http://localhost:8000/health`
  - `curl -s http://localhost:8000/api/robots`
  - `curl -s http://localhost:8000/api/robots/health/all`
  - `curl -s http://localhost:8000/api/goals`
- **Event-driven updates**: Create/update a goal or plan via API or CLI; the fleet server notifies the gateway, which pushes over WebSocket. With the frontend open on the host, you should see updates without polling.
- **Heartbeat flow (robots → telemetry → gateway → UI)**:
  1. Send a heartbeat to Telemetry:

     ```bash
     curl -X POST http://localhost:9000/ingest/heartbeat \
       -H "Content-Type: application/json" \
       -d '{"robot_id":"fake1","reachable":true}'
     ```

  2. Verify Telemetry stored it:

     ```bash
     curl http://localhost:9000/health/summary
     ```

  3. Verify Gateway reads from Telemetry:

     ```bash
     curl http://localhost:8000/api/robots/health/all
     ```

  4. If effective reachable changes (e.g. first heartbeat or timeout), Telemetry pushes `telemetry.health_changed` to the gateway, which pushes over WebSocket. The frontend reacts without polling.

---

## Live reload with Docker

- **Frontend**: Run it on the host (`npm run dev`). UI edits live-reload. The frontend does **not** need to be in Docker for dev.
- **Gateway / fleet-server / telemetry in Docker**: By default, code is baked into the image. A code change requires `docker compose up --build` (rebuild + restart).

To keep **gateway** and **telemetry** live reload while still using Compose for Postgres and all services:

```bash
cd robot_fleet
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

- **Gateway**: `./services/gateway/src` and `./packages` are mounted read-only; the container runs `uvicorn ... --reload`, so edits there apply without rebuilding the image.
- **Telemetry**: `./services/telemetry/src` is mounted read-only; the container runs `uvicorn services.telemetry.src.main:app ... --reload` for the same behavior.
- **Fleet-server**: `./services/fleet_server/src` and `./packages` are mounted read-only; **restart** the `fleet-server` container to pick up Python changes (no image rebuild needed).
- **Frontend**: Still run on the host; the app talks to `http://localhost:8000` via the Vite proxy. No container needed for local UI dev.

**Does the frontend need to be Dockerized?** For local dev, no—running `npm run dev` on the host keeps UI live reload and is simplest. For production you can serve the built static bundle from a container or a CDN later if you want.
