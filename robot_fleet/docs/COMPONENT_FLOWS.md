# Who Talks to Whom: Gateway, Fleet Server, Telemetry, Robots

Clear picture of the main backend services and how data flows. Paths below are relative to the `robot_fleet/` directory (Python package root).

---

## The main components

| Component | What it is | Typical port |
|-----------|------------|--------------|
| **Fleet server** | Central control plane. Owns the database (plans, tasks, robots, goals). Runs plan execution (sends tasks to robots). Exposes **gRPC** only. | 50051 |
| **Gateway** | Single entry point for **all clients** (browser, mobile, CLI). Exposes **REST + WebSockets**. Proxies control and DB-backed reads to the fleet via gRPC; queries the **Telemetry** service over HTTP for heartbeat-derived health. | 8000 |
| **Telemetry** | Separate service for **robot push** ingest (heartbeats today; room for streams later) and **health read APIs** the Gateway calls. Does not own fleet control data. | 9000 |
| **Robot** | One per physical/simulated robot. Runs a small HTTP server: `/do_task`, `/health`, and (optionally) sends heartbeats **to the Telemetry service**. | 8001, 8002, … |

---

## Rule of thumb

- **Clients (e.g. browser)** → talk **only** to the **Gateway** (except the one-off “Send Task” shortcut below).
- **Gateway** → talks to the **Fleet server** (gRPC) for control and DB-backed lists; talks to **Telemetry** (HTTP) for ingest-derived health and related reads.
- **Fleet server** → talks to **Robots** (to send tasks: “do this”).
- **Robots** → talk to the **Telemetry service** (heartbeats). They do **not** send heartbeats to the Fleet server for that path.

Realtime UI invalidation: after mutations, the **Fleet server** POSTs events to the Gateway (`POST /internal/events`); the Gateway’s WebSocket handlers notify clients. **Telemetry** can also POST health-changed events to the same endpoint so the UI refreshes robot health without polling.

So: nothing in the fleet **control plane** is bypassed for normal API usage; **Telemetry** sits beside the Gateway for observation data from robots.

---

## Flow diagrams

### 1) Control: “Do something” (create plan, run task, register robot, …)

```
  Browser (or other client)
         │
         │  HTTP: POST /api/plans, /api/plans/123/start, /api/robots/register, etc.
         ▼
  ┌──────────────┐
  │   Gateway    │
  └──────────────┘
         │
         │  gRPC: CreatePlan, StartExecution, RegisterRobot, etc.  (via grpc bridge)
         ▼
  ┌──────────────┐
  │ Fleet Server │ ──────►  Database (Postgres): plans, tasks, robots, goals
  └──────────────┘
         │
         │  When executing a plan: HTTP POST to each robot’s /do_task
         ▼
  ┌──────────────┐
  │    Robot     │  (one per robot; Fleet calls them)
  └──────────────┘
```

- **Request flow:** Client → Gateway → Fleet server. Fleet server may then call Robots.
- **Data flow:** Client sends intent; Gateway forwards; Fleet updates DB and commands robots; Robot returns task result to **Fleet**; Fleet updates DB again.

---

### 2) Observe: “See what’s happening” (list plans, task status, robot list, …)

```
  Browser (or other client)
         │
         │  HTTP: GET /api/plans, /api/tasks, /api/robots, etc.
         │  WebSocket: /ws/global-updates, /ws/execution/123
         ▼
  ┌──────────────┐
  │   Gateway    │
  └──────────────┘
         │
         │  gRPC: ListPlans, ListTasks, ListRobots, GetPlan, …  (via grpc bridge)
         ▼
  ┌──────────────┐
  │ Fleet Server │ ──────►  Database: read plans, tasks, robots, goals
  └──────────────┘
```

- **Request flow:** Client → Gateway → Fleet server (which reads from DB).
- **Data flow:** Fleet returns data to Gateway; Gateway returns it to the client. **Robots are not in this path** for “list plans / tasks / robots.”
- **Robot health in the UI:** Gateway aggregates fleet robot records with **Telemetry** health (`GET /health/...` on the Telemetry service). Heartbeats are **not** stored inside the Gateway process.
- **WebSockets:** The fleet server emits mutation events (`services/fleet_server/src/events.py`); the Gateway receives them on `POST /internal/events` and fans out invalidation over `/ws/global-updates` (see `services/gateway/src/routers/websocket.py`). Telemetry may POST `telemetry.health_changed` to the same internal endpoint when heartbeat-derived status changes.

---

### 3) Robot → Telemetry: heartbeats (and future telemetry)

```
  ┌──────────────┐
  │    Robot     │
  └──────────────┘
         │
         │  HTTP POST: /ingest/heartbeat  (Telemetry service; identity is host:port)
         │  (every N seconds; no one polls the robot)
         ▼
  ┌──────────────┐
  │  Telemetry   │  stores last heartbeat; exposes GET /health/summary, /health/{id}
  └──────────────┘
         │
         │  optional: POST /internal/events on Gateway (health_changed) for WS clients
         ▼
  ┌──────────────┐
  │   Gateway    │  browser calls GET /api/robots/health/* → Gateway queries Telemetry
  └──────────────┘
```

- **Request flow:** Robot → Telemetry for ingest; Browser → Gateway → Telemetry for displayed health.
- **Data flow:** Robot pushes “I’m alive, busy/idle” to Telemetry. Gateway reads that state when the UI asks. Fleet server is **not** in the heartbeat ingest path.

---

### 4) “Send Task” from the UI to one robot (special case)

Today, the **browser** can send a single task directly to a robot (robot card → “Send Task”):

```
  Browser
     │
     │  HTTP POST to http://<robot_host>:<robot_port>/do_task
     │  (Browser uses robot’s host/port from GET /api/robots)
     ▼
  ┌──────────────┐
  │    Robot     │  returns { success, message, replan }
  └──────────────┘
```

- Here the browser talks **directly** to the robot (one-off). The Gateway is only used to get the robot’s address (from the list). This is the only case where the client skips the Gateway for the actual request.

---

## Summary table

| Who | Talks to Fleet server? | Talks to Gateway? | Talks to Telemetry? | Talks to Robot? |
|-----|------------------------|-------------------|---------------------|-----------------|
| **Browser** | No | Yes (all API + WS) | No (only via Gateway) | Yes, only “Send Task” |
| **Gateway** | Yes (gRPC) | — | Yes (HTTP read + receives internal events) | No |
| **Fleet server** | — | Yes (outbound: POST mutation events to Gateway) | No | Yes (during execution) |
| **Telemetry** | No | Yes (POST /internal/events for health fan-out) | — | No |
| **Robot** | No | No | Yes (heartbeat ingest) | — |

So:

- **Gateway** talks to **Fleet server** (gRPC) and **Telemetry** (HTTP), and receives **internal event POSTs** from Fleet and Telemetry.
- **Fleet server** talks to **Robots** (and DB).
- **Robots** push heartbeats to **Telemetry**; they respond to **Fleet server** (and optionally the browser) for `/do_task`.

---

## Where this lives in the repo

| Area | Path |
|------|------|
| Gateway HTTP routers (plans, robots, tasks, goals, telemetry proxy, etc.) | `services/gateway/src/routers/` (`plans.py`, `robots.py`, `tasks.py`, `goals.py`, `telemetry.py`, …) |
| Gateway ↔ Fleet gRPC bridge | `services/gateway/src/grpc_bridge.py` |
| Gateway WebSockets + `POST /internal/events` | `services/gateway/src/routers/websocket.py` |
| Fleet gRPC service implementation | `services/fleet_server/src/service.py` |
| Plan / task execution loop | `services/fleet_server/src/executor/executor.py` |
| Fleet → Gateway event emission | `services/fleet_server/src/events.py` |
| Planners | `services/fleet_server/src/planners/` |
| Allocators | `services/fleet_server/src/allocators/` |
| DB / instance registry (fleet_sdk) | `packages/fleet_sdk/src/instance_registry.py` |
| Shared ports and URLs | `packages/config.py` |
| Telemetry FastAPI app, heartbeat store, publishing | `services/telemetry/src/` (`app.py`, `heartbeat_store.py`, `publishing.py`, …) |
| Dashboard API client and helpers | `services/dashboard-web/src/lib/` (`api.ts`, `utils.ts`) |
| Dashboard pages | `services/dashboard-web/src/pages/` |
