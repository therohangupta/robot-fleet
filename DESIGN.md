# DESIGN.md

### Goals
- **Prototype that scales**: keep production-shaped boundaries while staying lightweight.
- **Single UI boundary**: the browser should talk to exactly one service (the “Gateway/BFF”).
- **Control-plane correctness**: plan execution and task/plan truth must remain consistent and recoverable.
- **Data-plane scalability**: high-rate telemetry/video should not destabilize control-plane execution.

---

### Current repo reality (today)
- **Fleet control plane (gRPC server)**: `robot_fleet/services/fleet_server/src/service.py`
  - Owns execution orchestration via `robot_fleet/services/fleet_server/src/executor/executor.py`
  - Owns DB access through `robot_fleet/packages/fleet_sdk/src/instance_registry.py`
- **Gateway/BFF (FastAPI)**: `robot_fleet/services/gateway/*`
  - Exposes `/api/*` to the frontend and bridges to fleet via `robot_fleet/services/gateway/src/grpc_bridge.py`
  - Exposes event-driven WebSockets in `robot_fleet/services/gateway/src/routers/websocket.py`
- **Telemetry service (FastAPI)**: `robot_fleet/services/telemetry/*`
  - Heartbeat ingest and robot health summary
- **Frontend (Vite/React)**: `robot_fleet/services/dashboard-web/*`
  - Calls `/api/...` (already aligned with “single UI boundary”)
- **Robot task servers (examples)**: `robot_fleet/robots/fake/*`

---

### Recommended architecture (production-shaped, prototype-friendly)

#### Components
- **Robots (many)**: execute tasks + produce telemetry/video.
- **Fleet Server (Control Plane)**: orchestrates plans/tasks/allocations/execution; writes truth to DB; commands robots.
- **Gateway/BFF (Data Plane Edge for UI)**: the only browser-facing API; fans out realtime updates; (optionally) ingests telemetry/video or coordinates those services.
- **Postgres (Truth Store)**: plans/tasks/robots/goals + execution outcomes + metadata.

Optional later (add only when needed):
- **Event stream** (NATS/Redis Streams/Kafka): decouples fleet events from UI fanout/analytics.
- **Time-series store** (Prometheus/Influx/Timescale): stores high-rate telemetry history.
- **Media service** (WebRTC SFU + recorder): scalable livestream + recording.
- **Object storage** (S3/MinIO): videos, trajectories, large artifacts; Postgres stores metadata/pointers only.

---

### Control path vs Observe path (the two paths)

#### 1) Control path (“do something”)

```text
UI → Gateway/BFF → Fleet Server → Robot(s)
                    ↓
                 Postgres
```

Examples:
- register/unregister robot
- create/update plan/task, allocate robots
- start/cancel execution

Key rule:
- **UI never calls robots or fleet directly** (except in local dev prototypes if you explicitly choose to).

#### 2) Observe path (“see what’s happening”)
This splits into two categories:

##### A) Control-plane observability (authoritative state)

```text
Postgres (truth) ← Fleet Server → Gateway/BFF → UI
```

Includes:
- plans/tasks/goals/allocations
- task status + final task results
- plan execution status

##### B) Data-plane observability (telemetry + media)

```text
Robot(s) → Telemetry/Media Ingress (usually Gateway or a dedicated service) → UI
```

Includes:
- robot health heartbeat streams (if treated as telemetry)
- joint states, sensor streams, rich logs
- camera/video livestream

Key rule:
- **Avoid duplicating full-rate streams** to multiple consumers. Prefer “send once → fan out”.
- Fleet should consume **summaries** needed for decisions, not raw high-rate streams.

---

### “Desired vs Observed” + reconciliation (why both fleet & robot status matter)
The fleet server maintains:
- **Desired/commanded state**: what it dispatched and what the DB says should be happening.
Robots maintain:
- **Observed state**: what is actually happening (heartbeat, faults, estop, reboot, busy/idle).

Fleet computes an **effective** status by reconciling both:
- If `last_heartbeat` is stale → robot is **offline**, even if DB says tasks are in progress.
- If robot reports `fault/estop` → robot is **unavailable** even if “idle”.
- If fleet thinks busy but robot reports idle → tasks may be stale; recover/reassign.

This keeps execution robust under crashes, partitions, and restarts.

---

### Realtime: event-driven updates (implemented)
Current state:
- Fleet server POSTs events to the gateway on every mutation via `robot_fleet/services/gateway/src/routers/websocket.py`.
- Per-subscriber queues push invalidation signals immediately with no polling.
- WebSocket clients wake only when there is a real change.

---

### Suggested event schema (minimal set)
Use a small stable contract; the UI consumes:
- **snapshot** on connect
- **deltas/events** as they happen

Suggested event types:
- `task.state_changed`: `{ plan_id, task_id, status, robot_id, ts, result? }`
- `plan.state_changed`: `{ plan_id, status, ts }`
- `robot.heartbeat`: `{ robot_id, reachable, latency_ms?, battery?, faults?, ts }`
- `robot.state_changed` (effective): `{ robot_id, effective_state, reason?, ts }`
- `replan.occurred`: `{ old_plan_id, new_plan_id, failed_task_id?, reason, ts }`

Notes:
- Keep payloads small and stable.
- For heavy content (video/log blobs), send references/URLs and fetch/stream separately.

---

### Protocol recommendations (prototype → production)

#### Control plane
- **Fleet ⇄ Robots**: gRPC is ideal (or HTTP if needed).
- **Gateway ⇄ Fleet**: gRPC or HTTP; events via WS/SSE or streaming gRPC.
- **UI ⇄ Gateway**: REST for queries/mutations; WS/SSE for realtime.

#### Telemetry
- **Robot → Gateway/Telemetry service**:
  - low-rate: HTTP POST or WS
  - high-rate: WS or gRPC streaming
  - logs: WS stream
- **Video**:
  - robot publishes RTSP/WebRTC internally
  - gateway/media layer exposes **WebRTC** (or HLS) to browsers

---

### Naming & class design recommendations (extendable)
Current names work for a prototype, but clearer separation helps future growth.

#### Rename by responsibility (recommended)
- `FleetManagerService` → **`FleetControlService`**
  - communicates “this is the control plane”
- `Executor` → **`PlanExecutor`**
  - clearer that it executes a plan DAG, not a generic “executor”
- `RobotClient` → **`RobotTaskClient`**
  - clearer that this is task RPC, not telemetry/media
- `GRPCBridge` (dashboard/backend) → **`FleetGatewayClient`** or **`FleetControlClient`**
  - “bridge” is vague; “client” clarifies direction

#### Introduce explicit interfaces (even if implemented in one file today)
- **Control plane interfaces**:
  - `IRobotCommandClient`: send tasks/cancel/pause
  - `IPlanExecutor`: run plan execution loop
  - `IStateStore`: DB operations for plans/tasks/robots
- **Gateway interfaces**:
  - `IEventSubscriber`: subscribe to fleet events
  - `IEventBroadcaster`: broadcast to UI clients
  - `ITelemetryIngestor`: accept robot telemetry
  - `IMediaSessionManager`: negotiate video sessions (later)

#### Organize packages by plane (optional refactor later)
- `robot_fleet/control_plane/...`
- `robot_fleet/data_plane/...` (or keep telemetry in dashboard backend)
- `robot_fleet/storage/...` (DB models + registry)

---

### Concrete “best-practice” changes to make next (without breaking docker testing)
- Keep frontend calling `/api/*` (already true).
- Keep dashboard backend as Gateway/BFF (already exists).
- ~~Replace WS polling loops with event-driven updates~~ (done: fleet → gateway → UI via per-subscriber queues).
- Move robot health to a fleet-owned heartbeat/reconciliation model (avoid UI-driven constant probing).
- Add a documented event schema and keep it stable as features grow.
- Keep videos/trajectories out of Postgres; store metadata in Postgres and blobs in object storage later.

---

### Dev workflow (still works)
- You can continue using docker “fake robots” and the frontend.
- For faster iteration on robot server code, use bind mounts (dev runner script) and restart containers to reload code.

