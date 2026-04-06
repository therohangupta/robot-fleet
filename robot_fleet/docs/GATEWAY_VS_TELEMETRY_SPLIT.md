# Gateway (BFF) vs Telemetry service

The fleet stack uses **two separate HTTP services**: a **gateway (BFF)** for browsers and clients, and a **telemetry** service for robot push and health-oriented reads. This document describes **today’s architecture** (where the code and Compose layout already are) and why those boundaries stay useful.

**Where things live (repo):**

| Piece | Path | Default port |
|-------|------|----------------|
| **Gateway (BFF)** | `robot_fleet/services/gateway/src/` | **8000** (`GATEWAY_PORT`) |
| **Telemetry service** | `robot_fleet/services/telemetry/src/` | **9000** (`TELEMETRY_PORT`) |
| **Fleet server** (gRPC) | `robot_fleet/services/fleet_server/src/` | **50051** |
| **Dashboard frontend** | `robot_fleet/services/dashboard-web/src/` | **5173** (dev) |

Docker Compose wires `TELEMETRY_URL` on the gateway to the telemetry container and runs all three backend services together (see `robot_fleet/docker-compose.yml`).

---

## The two roles

| Role | Who talks to it | What it does | Direction of data |
|------|------------------|--------------|--------------------|
| **BFF / API Gateway** | **Frontend (and other clients)** | Single entry point for control and queries. Proxies to **fleet server** (gRPC) for plans/tasks/robots. **Queries the Telemetry service** over HTTP for health summaries (`GET /health/summary`, per-robot health). Handles auth, CORS, and realtime fanout (WebSockets). | **Client → Gateway → Fleet** (control) and **Client → Gateway → Telemetry** (health for display) |
| **Telemetry service** | **Robots** (push) and **Gateway/BFF** (query) | **Ingest:** Receives push data from robots (today: **heartbeats** on `POST /ingest/heartbeat`). **Query:** Exposes read APIs so the **Gateway** can fetch health for the UI. Can **POST** `telemetry.health_changed`-style events to the Gateway’s `POST /internal/events` so WebSocket clients invalidate robot-health queries without polling. | **Robot → Telemetry → (store)**; **Gateway → Telemetry** (read) → **Gateway → Frontend** |

So:

- **BFF** = client-facing: route control to fleet, and **query telemetry** for what the UI needs (health today; joints/video when those APIs exist).
- **Telemetry** = robot-facing ingest + storage path for execution telemetry; **read API** for the Gateway. The **frontend does not call Telemetry directly**—only the Gateway does.

---

## Why this split matters (even though it is already split)

Separate processes give you:

- **Independent scaling** — many robots and high-rate streams can stress Telemetry without dragging down REST/WebSocket latency on the Gateway.
- **Clear ownership** — “control plane” (fleet + gateway) vs “telemetry plane” (ingest, retention, training-oriented data).
- **Security boundary** — robots need not reach the BFF; browsers need not reach Telemetry.

Early prototypes often folded a small heartbeat handler into the gateway to reduce moving parts. The **current repo has already moved** ingest and health storage to `services/telemetry`; the gateway uses `services/gateway/src/services/telemetry_client.py` to read health from Telemetry.

---

## Current deployment shape

### 1) API Gateway (BFF)

- **Owns:** Client-facing HTTP API and WebSockets (`/ws/global-updates`, `/ws/execution/{plan_id}`).
- **Talks to:** Fleet server (gRPC) and **Telemetry service** (`TELEMETRY_URL`, HTTP). Does **not** need robots to call it for heartbeats in the nominal path.
- **Responsibilities:**
  - REST for plans/tasks/robots → fleet via `grpc_bridge`.
  - Robot health for the UI → **HTTP to Telemetry** (`/health/summary`, `/health/{robot_id}`), merged with fleet robot metadata where needed (`routers/robots.py`).
  - **POST `/internal/events`** — internal endpoint: **Fleet** and **Telemetry** POST here; the Gateway pushes WebSocket invalidations using **per-subscriber queues** (`routers/websocket.py`).
- **Legacy note:** The gateway still includes `POST /api/telemetry/heartbeat` and an in-memory store in `routers/telemetry.py` for backward compatibility. **Fake robots and the intended path use Telemetry** (`TELEMETRY_URL/ingest/heartbeat`).

### 2) Telemetry service

- **Owns:** Robot heartbeat ingest and in-memory (or pluggable) last-seen state; health read API.
- **Talks to:** Robots on ingest; optionally **Gateway** via `GATEWAY_EVENT_URL` for health-changed fan-out.
- **Responsibilities:**
  - **Ingest:** `POST /ingest/heartbeat` (see `services/telemetry/src/routers/ingest.py`).
  - **Query:** `GET /health/summary`, `GET /health/{robot_id}` (see `routers/health.py`).
  - **Not** responsible for `/api/plans`, task execution, or gRPC to fleet.

### 3) End-to-end flows

- **Control:** Browser → **Gateway** → Fleet → (fleet calls robot `/do_task` when executing).
- **Telemetry ingest:** Robot → **Telemetry** (`/ingest/heartbeat` today).
- **Telemetry display:** Browser → **Gateway** → **Telemetry** (health) → Gateway → Browser.
- **Realtime UI:** Fleet (and Telemetry) → **POST Gateway `/internal/events`** → WebSocket clients get `invalidate` (and execution channels refresh task lists on events).

---

## Summary

| Question | Answer |
|----------|--------|
| Are Gateway and Telemetry separate today? | **Yes.** Telemetry is `services/telemetry` (port 9000); Gateway is `services/gateway` (port 8000). |
| What does the Gateway do for telemetry? | **Queries** Telemetry for health; forwards to the frontend. It does **not** replace Telemetry for robot push in the supported setup. |
| Is telemetry only heartbeats forever? | **Today, mostly yes** at the ingest layer. **Design intent** is heavier execution data (joints, video, logs) on Telemetry with real storage—see `IMPLEMENTATION_PLAN.md` Phase 4–5. |
| How does the frontend get telemetry? | **Frontend → Gateway → Telemetry** for health. No direct browser → Telemetry calls. |
| How do live updates work? | Fleet **HTTP callback** to Gateway; Telemetry can also notify Gateway on health changes; Gateway uses **event-driven WebSockets** (no 1s polling loops on those paths). |

If you extend the system, add new **read** routes on Telemetry and **proxy or aggregate** them from the Gateway so the browser’s single origin stays the BFF.
