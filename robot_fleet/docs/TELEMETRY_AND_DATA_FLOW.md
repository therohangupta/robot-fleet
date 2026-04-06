# Telemetry, Data Flow, and Robot → Storage Pipeline

Telemetry is **not** just lightweight heartbeats. Long term it includes **execution data** for collection and training: video, images, joint positions, logs. The **Telemetry service** is a **separate HTTP server** that ingests from robots, holds (today: in-memory) state, and exposes **read APIs** so the **Gateway (BFF)** can query it and pass data to the frontend for display. The frontend never talks to the Telemetry service directly.

---

## 1. What handles telemetry today (v2)

### Telemetry service (canonical path)

The **Telemetry service** is a dedicated FastAPI app under **`services/telemetry/src/`**:

| Area | Files / behavior |
|------|------------------|
| App entry, lifespan, timeout scanner | `app.py` — background task calls `HeartbeatStore.check_timeouts()` and publishes when robots go stale |
| Store | `heartbeat_store.py` — thread-safe in-memory ring buffer (last **N** heartbeats per robot; **N** = `MAX_HEARTBEATS_PER_ROBOT` in `packages/config.py`) |
| Ingest | `routers/ingest.py` — **`POST /ingest/heartbeat`** |
| Read API | `routers/health.py` — **`GET /health/summary`**, **`GET /health/{robot_id}`** |
| Events | `events.py` — `HealthChangedEvent` with `type: "telemetry.health_changed"` |
| Publish to Gateway | `publishing.py` — HTTP **`POST`** to `GATEWAY_EVENT_URL` (default `{GATEWAY_URL}/internal/events`) |

**Robots** should POST heartbeats to **`{TELEMETRY_URL}/ingest/heartbeat`** (Compose exposes Telemetry on **port 9000** by default; `TELEMETRY_URL` / `TELEMETRY_PORT` come from `packages/config.py`).

**Ingest body** (see `routers/ingest.py`): `host`, `port`, `reachable`, optional `busy`, optional `ts`. The store’s primary key is **`host:port`** (task server address), so the Gateway can join Telemetry data with Fleet robots using each robot’s `task_server_info`.

**Joint positions, video, and rich logs** are still **not** implemented on this path; the target architecture below still applies for those.

### Gateway (consumer + real-time fan-out)

- **`services/gateway/src/services/telemetry_client.py`** — fetches **`GET {TELEMETRY_URL}/health/summary`** and **`GET {TELEMETRY_URL}/health/{robot_id}`** when serving Gateway APIs.
- **`services/gateway/src/routers/robots.py`** — robot health endpoints default to **`source=telemetry`**: they merge Fleet’s robot list with Telemetry’s summary keyed by **`host:port`**.
- **`services/gateway/src/routers/websocket.py`** — accepts **`POST /internal/events`** from Fleet **and** Telemetry; maps **`telemetry.health_changed`** → invalidates **`robot-health`** for WebSocket clients (same event bus as fleet mutations).

### Legacy route on the Gateway (optional / migration)

**`services/gateway/src/routers/telemetry.py`** still defines **`POST /api/telemetry/heartbeat`** with a **`robot_id`**-shaped payload and an in-memory **`_heartbeats`** dict on the Gateway. That path is **legacy**; with Telemetry deployed, robots should use **`POST …/ingest/heartbeat`** on the Telemetry service so the Gateway reads one source of truth via **`telemetry_client`**.

---

## 2. Frontend ↔ one robot: what actually happens

You have two ways the frontend interacts with a single robot:

### A) “Send Task” from the Robot detail (UI → robot **directly**)

1. Frontend gets the robot list from **Gateway** (`GET /api/robots`), which comes from Fleet/DB (includes `task_server_info.host` and `task_server_info.port`).
2. User opens a robot, goes to “Send Task”, enters a description, and submits.
3. Frontend **POSTs straight to the robot**:
   - URL: `http://<robot_host>:<robot_port>/do_task`
   - Body: `{ "task_description": "..." }`
4. Robot responds with:
   - `{ "success": bool, "message": string, "replan": bool }` (task result).

So for this flow: **frontend → robot directly**. No telemetry: only the one-off task result in the HTTP response. The gateway is not in the path for this request.

### B) Plan execution (UI → Gateway → Fleet → robot)

1. User creates/allocates a plan and clicks “Execute” (or “Start”).
2. Frontend calls **Gateway** `POST /api/plans/{plan_id}/start`.
3. Gateway calls **Fleet** (gRPC); Fleet’s **Executor** runs the plan:
   - For each ready task, Fleet’s `RobotClient` **POSTs to that robot’s `/do_task`** (same as above, but from Fleet, not from the browser).
4. Robot returns the same `TaskResult` to Fleet; Fleet updates task status in the DB.
5. Frontend learns about progress by:
   - Polling (e.g. `GET /api/tasks?plan_id=...`) or
   - WebSocket `ws/execution/{plan_id}` (today still a 1s poll under the hood).

So for execution: **data “from the robot”** that the frontend sees is **task status and result text** stored in the DB and exposed via Gateway APIs/WS. No raw telemetry (joints, video) in this path.

---

## 3. Target: telemetry from robot → your storage (joints, video, etc.)

You want:

- **Each robot** to use its own stack (ROS1/2, Rust, etc.) to **read** joint positions, video, and other sensors.
- **Send** that data to a **central ingest** (no duplicate streams to many consumers).
- **Store** in a **place you specify** (e.g. object storage, time-series DB) for data collection and training.

A clean way to do that:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Robot                                                                      │
│  - Task server (FastAPI): /do_task, /health, optional /telemetry ingest      │
│  - Robot-specific collector: ROS1/2 or Rust → reads joints, camera, etc.  │
│  - Sends telemetry to central ingest (HTTP POST, WebSocket, or gRPC stream)│
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Central telemetry ingest                                                   │
│  **Today:** Telemetry service — heartbeats only (`POST /ingest/heartbeat`).  │
│  **Target:** same service (or additional routes) receives streams per      │
│    robot (tags: robot_id / host:port, plan_id, task_id, session_id).        │
│  - Normalizes / validates (e.g. canonical joint schema, chunked video).    │
│  - Forwards to storage you configure.                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Your storage (you specify)                                                 │
│  - Time-series DB (e.g. Influx, Timescale, Prometheus): joint positions,    │
│    small sensor streams.                                                     │
│  - Object storage (S3, MinIO, GCS): video files, trajectory dumps, logs.   │
│  - Postgres (or existing DB): metadata, indexes (robot_id, plan_id, task_id,│
│    timestamps, pointers to blobs in object storage).                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Displaying telemetry in the UI:** The **Telemetry service** exposes **read APIs** (today: health summary derived from heartbeats). The **Gateway (BFF)** calls those APIs and passes the response to the frontend. So: **Browser → Gateway → Telemetry service (query) → Gateway → Browser**. The frontend never talks to the Telemetry service directly.

- **Who handles telemetry (heartbeats):** **`services/telemetry/src/`** — ingest, store, read API, and push **`telemetry.health_changed`** to the Gateway. The Gateway **queries** Telemetry on demand when serving health APIs and reacts to events for WebSocket invalidation (see `TELEMETRY_STORE_AND_EVENTS.md`).
- **Flow of data (ingest, heartbeats):** robot → **`POST {TELEMETRY_URL}/ingest/heartbeat`** → Telemetry store → (optional) event → Gateway → WS → UI refetch.
- **Flow of data (display):** Frontend → Gateway → **`GET {TELEMETRY_URL}/health/...`** → Gateway → Frontend.

Concretely (heartbeats **implemented**; streams **future**):

1. **Robot side (your responsibility per robot)**
   - Keep the existing task server for **control** (`/do_task`, `/health`).
   - Send heartbeats to **`{TELEMETRY_URL}/ingest/heartbeat`** with **`host`/`port`** matching the task server Fleet knows about.
   - **Later:** add a **telemetry sender** (ROS node, Rust binary, etc.) that pushes structured streams to new ingest endpoints on the Telemetry service (or a sibling component).

2. **Central ingest (Telemetry service)**
   - **Implemented:** `POST /ingest/heartbeat`, read **`GET /health/summary`**, **`GET /health/{robot_id}`**, publish to Gateway **`POST /internal/events`**.
   - **Add later:** e.g. `POST /ingest/stream` (and/or WebSocket) for joints/logs; chunked or streaming video to object storage; metadata in Postgres.

3. **Storage you specify**
   - Configure the ingest (or media service) with time-series DB, object storage, and optional Postgres — **future** for high-volume telemetry; heartbeats today stay in-process on the Telemetry service unless you add Redis etc. (see `TELEMETRY_STORE_AND_EVENTS.md`).

4. **Frontend**
   - For **control**: unchanged (Send Task → robot; Execute Plan → Gateway → Fleet → robot).
   - For **telemetry**: no direct robot↔frontend telemetry. Frontend uses Gateway APIs; Gateway reads Telemetry and receives push invalidations over **`/ws/global-updates`**.

This keeps a single path for heartbeats: **robot → Telemetry → (events + read API) → Gateway → UI**, and leaves room to grow **robot → Telemetry → your storage** for training-scale data.

---

## 4. Summary

| Question | Answer |
|----------|--------|
| **What handles telemetry (heartbeats)?** | **Telemetry service** at `services/telemetry/src/`. Gateway **`telemetry_client`** reads **`/health/*`**; **`websocket`** handles **`telemetry.health_changed`**. Legacy **`routers/telemetry.py`** on the Gateway is optional. |
| **Frontend → one robot flow** | **Control:** (1) “Send Task” = frontend → robot `POST /do_task`. (2) Execute plan = frontend → Gateway → Fleet → robot `/do_task`; “data back” = task status/result via DB and Gateway APIs/WS. **Telemetry:** heartbeats → Telemetry ingest; UI sees health via Gateway (and WS invalidation). **Joints/video:** not implemented yet. |
| **Joint / video / storage** | Target: robot collector → Telemetry (or media) ingest → **your** time-series DB and object storage. **Today:** only heartbeat ingest + in-memory store on Telemetry. |

For the event-driven health path and store patterns, see **`TELEMETRY_STORE_AND_EVENTS.md`**.
