# Telemetry Service

Minimal telemetry server for heartbeat ingest and health read API.

## Responsibilities

- **Ingest**: Robots POST heartbeats to `POST /ingest/heartbeat`.
- **Store**: Keeps last N heartbeats per robot in memory (configurable via `MAX_HEARTBEATS_PER_ROBOT`).
- **Health read API**: Gateway queries `GET /health/summary` or `GET /health/{robot_id}` for derived health.
- **Event push**: When effective "reachable" changes, sends `telemetry.health_changed` event to the gateway so the UI updates without polling.

## Running locally (bare metal)

From repo root (`robot_fleet/`):

```bash
pip install -e .
python -m services.telemetry.src --port 9000
```

Or with uvicorn directly:

```bash
cd services/telemetry
uvicorn src.main:app --reload --port 9000
```

## Running in Docker

From repo root:

```bash
docker compose up --build
```

The `telemetry` service is defined in `docker-compose.yml`.

## Configuration (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_EVENT_URL` | `http://localhost:8000/internal/events` | Where to POST health_changed events |
| `TELEMETRY_PORT` | `9000` | Port for the Telemetry service |
| `MAX_HEARTBEATS_PER_ROBOT` | `20` | Ring buffer size per robot |
| `HEARTBEAT_REACHABLE_THRESHOLD_SECS` | `45.0` | Seconds since last heartbeat before "unreachable" |

## API

- `POST /ingest/heartbeat` — robot heartbeat ingest
- `GET /health/summary` — all robots' health
- `GET /health/{robot_id}` — single robot's health
- `GET /healthz` — liveness probe
