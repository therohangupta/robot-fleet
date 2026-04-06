# How to run and test robot_fleet

Everything runs from the **robot_fleet** repo only. No old_robot_fleet.

Assumptions: Postgres is running (e.g. database `robot_fleet`, user `robot_user`). Optional: set `DATABASE_URL` or use a `.env` in the workspace root.

You need **four terminals** for bare-metal dev (or use Docker Compose). From the **workspace root** (the directory that contains `robot_fleet`):

---

## 1. Install the repo (one-time)

```bash
cd robot_fleet
pip install -e .
cd ..
```

This makes `packages` and `services` importable.

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

Leave it running. Robots POST heartbeats here; gateway queries it for health.

---

## 4. Terminal 3 – Gateway (REST + WebSocket, port 8000)

```bash
cd robot_fleet/services/gateway
uvicorn src.main:app --reload --port 8000
```

Leave it running. The frontend proxies `/api` and `/ws` to this port.

---

## 5. Terminal 4 – Frontend (Vite, port 5173)

```bash
cd robot_fleet/services/dashboard-web
npm install
npm run dev
```

Open **http://localhost:5173** in the browser.

---

## Quick check

- **Gateway**: `curl -s http://localhost:8000/health` → `{"status":"healthy",...}`
- **Telemetry**: `curl -s http://localhost:9000/healthz` → `{"status":"ok"}`
- **Frontend**: http://localhost:5173 loads the dashboard.

---

## Summary

| Component     | Command (from workspace root)                                                                 | Port  |
|--------------|------------------------------------------------------------------------------------------------|-------|
| Fleet server  | `cd robot_fleet && python -m services.fleet_server.src -v`                                   | 50051 |
| Telemetry     | `cd robot_fleet && python -m services.telemetry.src --port 9000`                             | 9000  |
| Gateway       | `cd robot_fleet/services/gateway && uvicorn src.main:app --reload --port 8000`              | 8000  |
| Frontend      | `cd robot_fleet/services/dashboard-web && npm run dev`                                       | 5173  |

All four run from the **robot_fleet** repo only.

---

## Docker (scalable)

From **robot_fleet** repo root:

```bash
cd robot_fleet
docker compose up --build
```

This starts **Postgres**, **fleet-server** (gRPC 50051), **telemetry** (HTTP 9000), and **gateway** (HTTP 8000) in separate containers. They talk over the Compose network (e.g. `gateway:8000`, `telemetry:9000`, `fleet-server:50051`, `db:5432`).

- **Gateway**: http://localhost:8000 (e.g. `curl http://localhost:8000/health`)
- **Telemetry**: http://localhost:9000 (e.g. `curl http://localhost:9000/healthz`)
- **Fleet gRPC**: localhost:50051 (for CLI/scripts)
- **DB**: Container Postgres on port 5433 (host), uses persistent Docker volume `db_data`

Run the **frontend** on the host for dev: `cd robot_fleet/services/dashboard-web && npm run dev`, then open http://localhost:5173 (it proxies to the gateway on port 8000).

---

## Importing existing data into the Docker Postgres (one-time)

The containerized Postgres uses a **persistent Docker volume** (`db_data`). Data survives container restarts. But it starts empty.

To import your existing host Postgres data into the container:

```bash
# 1. Make sure the db container is running
cd robot_fleet
docker compose up db -d

# 2. Export from your HOST Postgres (rohangupta@localhost:5432/robot_fleet)
pg_dump -U rohangupta -d robot_fleet > /tmp/robot_fleet_dump.sql

# 3. Import into the CONTAINER Postgres (robot_user@localhost:5433/robot_fleet)
#    Note: container exposes port 5433 on the host
psql -h localhost -p 5433 -U robot_user -d robot_fleet < /tmp/robot_fleet_dump.sql
#    Password: secret

# 4. Verify
psql -h localhost -p 5433 -U robot_user -d robot_fleet -c "SELECT COUNT(*) FROM goals;"
```

After this, your container DB has your existing data, and any changes persist in the `db_data` volume.

**To completely reset the container DB** (start fresh):

```bash
docker compose down -v   # -v removes volumes
docker compose up --build
```

---

## What to test end-to-end

- **DB**: Compose starts Postgres; fleet-server and gateway connect to it. No extra setup if you use the built-in `db` service.
- **Fleet server**: gRPC on 50051. From the host you can use the CLI: `robotctl world list`, `robotctl robots list`, etc. (CLI must be installed from this repo and point at `localhost:50051` or the gateway.)
- **Telemetry service**: HTTP on 9000. Robots (or scripts) POST heartbeats here; gateway queries for health.
- **Gateway**: HTTP on 8000. Try:
  - `curl -s http://localhost:8000/health`
  - `curl -s http://localhost:8000/api/robots`
  - `curl -s http://localhost:8000/api/robots/health/all`
  - `curl -s http://localhost:8000/api/goals`
- **Event-driven updates**: Create/update a goal or plan via API or CLI; the fleet server sends an event to the gateway, which pushes over WebSocket. With the frontend open (on host), you should see updates without polling.
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

  4. If effective reachable changes (e.g. first heartbeat or timeout), Telemetry pushes `telemetry.health_changed` to Gateway, which pushes over WebSocket. The frontend reacts without polling.

---

## Live reload with Docker

- **Frontend**: Run it on the host (`npm run dev`). No change: your UI edits still live-reload. The frontend does **not** need to be in Docker for dev.
- **Gateway / fleet-server / telemetry in Docker**: By default, code is baked into the image. A code change would require `docker compose up --build` (rebuild + restart).

To keep **gateway and telemetry live reload** while using Docker for db + fleet-server + gateway + telemetry:

```bash
cd robot_fleet
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

- **Gateway**: Source is mounted and the process runs with `uvicorn ... --reload`, so edits to gateway (and shared `packages`) take effect without rebuilding.
- **Telemetry**: Source is mounted and the process runs with `uvicorn ... --reload`, so edits to telemetry take effect without rebuilding.
- **Fleet-server**: Source is mounted; restart the fleet-server container to pick up code changes (no rebuild needed).
- **Frontend**: Still run on the host; point the app at `http://localhost:8000` (gateway). No container needed for dev.

**Does the frontend need to be Dockerized?** For local dev, no—running `npm run dev` on the host keeps UI live reload and is simplest. For production you can serve the built static bundle from a container or a CDN later if you want.
