# RobotFleet v2: Step-by-Step Implementation Plan

This plan orders the main architectural improvements we’ve analyzed and specifies how to test each step. It assumes the current v2 state: Fleet server, Gateway (BFF + telemetry ingest in one process), Dashboard frontend, and fake robots with heartbeat push.

---

## Summary of components we’ve analyzed

| Component | Role | Target state |
|-----------|------|--------------|
| **Fleet server** | Control plane: DB, execution, gRPC. | Emit events on mutations (task/plan/robot) so Gateway can push to UI without polling. |
| **Gateway (BFF)** | Client-facing: proxy to Fleet, WebSockets, (today) telemetry ingest. | Event-driven WS only; no sleep loops. Eventually: no ingest, only query Telemetry service for display. |
| **Telemetry service** | Ingest (robot → storage) + read API (Gateway → display). | Separate server: heartbeat, joints, video, images; storage; read API for Gateway. |
| **Frontend** | Single client talking only to Gateway. | No refetchInterval; rely on WS invalidation and Gateway-proxied telemetry. |
| **Robots** | Execute tasks (Fleet calls them); push telemetry. | Push to Telemetry service (heartbeat, later streams). |

---

## Phase 1: Event-driven updates (remove polling)

**Goal:** No timer-based polling. Fleet notifies Gateway on change; Gateway pushes once to WS clients; frontend refetches only on invalidation.

### Step 1.1 — Fleet server: emit events on mutations

- **Implement:**  
  - Add config: `GATEWAY_EVENT_URL` (e.g. `http://localhost:8000/internal/events`).  
  - After every relevant mutation (task status, plan status, robot register/unregister), call `POST <GATEWAY_EVENT_URL>` with a small JSON body, e.g. `{ "type": "task.state_changed", "plan_id": 1, "task_id": 2, "status": "completed" }`.  
  - Fire-and-forget (non-blocking); do not fail the gRPC call if the POST fails.  
  - Call sites: executor (`update_task_status`, `update_plan` execution_status), and gRPC handlers (CreateTask, UpdateTask, DeleteTask, CreatePlan, UpdatePlan, DeletePlan, RegisterRobot, UnregisterRobot, etc.).  
- **Test:**  
  - Start Fleet + Gateway.  
  - Trigger a mutation (e.g. create a task via API or run a plan that updates task status).  
  - Verify Gateway receives a POST to `/internal/events` (log or temporary debug endpoint).  
  - Optional: use a small script or curl to POST a fake event and confirm Gateway logs it.

### Step 1.2 — Gateway: event receiver and event-driven WebSocket

- **Implement:**  
  - Add `POST /internal/events` (or `/api/internal/events`): accept JSON `{ "type": "...", "plan_id": ?, "task_id": ?, ... }`.  
  - Maintain in-memory lists of WS connections: (1) global-updates, (2) per-plan `ws/execution/{plan_id}`.  
  - On event:  
    - For `task.state_changed` / `plan.state_changed`: broadcast to global-updates `{ type: "invalidate", queries: ["plans", "tasks", "robots", "robot-allocations"] }`; for matching `plan_id`, also fetch tasks for that plan and send `{ type: "tasks_update", plan_id, tasks }` to all connections subscribed to `ws/execution/{plan_id}`.  
    - For `robot.registered` / `robot.unregistered`: broadcast invalidation including `"robots"`, `"robot-allocations"`.  
  - **Remove** the two `while True` + `asyncio.sleep(1)` loops in `ws/global-updates` and `ws/execution/{plan_id}`.  
  - On WS connect: global-updates sends `{ type: "connected" }` once; execution sends one initial `tasks_update` then only on events.  
- **Test:**  
  - Open dashboard in browser; open DevTools → Network (WS).  
  - Connect to `ws/global-updates` and `ws/execution/1` (for plan_id 1).  
  - Trigger a task status change (e.g. run a plan or update a task via API).  
  - Verify: exactly one new WS message (invalidate or tasks_update) after the mutation, and no repeated messages every second.  
  - Verify: UI updates (e.g. task list or execution view) without needing a manual refresh.

### Step 1.3 — Frontend: remove refetch intervals; rely on WS

- **Implement:**  
  - In `main.tsx`: remove default `refetchInterval` (or set to `false`).  
  - In `Dashboard.tsx`: use `useRealtimeUpdates()`, remove `refetchInterval` from the robot-health query.  
  - In `Execution.tsx`: use `useRealtimeUpdates()` and ensure execution WS is connected when the page is mounted; remove `refetchInterval` for tasks and robots.  
  - Ensure `useRealtimeUpdates` invalidates query keys that match what the Gateway sends (e.g. `["tasks"]` and optionally `["tasks", planId]` for execution).  
- **Test:**  
  - Load Dashboard and Execution pages; confirm no refetch timer in React Query DevTools (or no 1s/2s/5s refetch).  
  - Trigger a plan execution or task update from another tab/API; confirm the open tab updates within a second via WS-driven invalidation.  
  - Confirm no console errors and that data (plans, tasks, robots) still appears correctly.

**Phase 1 done when:** No `asyncio.sleep` in Gateway WS handlers; no refetchInterval for live data; UI updates only when Fleet sends an event.

---

## Phase 2: Robot health from heartbeat only

**Goal:** Gateway stops polling each robot’s `/health`. Health shown in the UI comes only from the heartbeat store (robot → Gateway or, later, robot → Telemetry service).

### Step 2.1 — Gateway: health from heartbeat store

- **Implement:**  
  - Change `GET /api/robots/health/all` (and any per-robot health) to **read from the existing heartbeat store** (e.g. `get_all_heartbeats()` from `routers/telemetry.py`).  
  - Derive “reachable” from “last heartbeat within last N seconds” (e.g. 30–45 s).  
  - Remove or stop using the code that calls each robot’s HTTP `/health` on a schedule.  
- **Test:**  
  - Start a fake robot (with heartbeat enabled) and Gateway.  
  - Call `GET /api/robots/health/all`; confirm response includes that robot as reachable and uses timestamp from heartbeat.  
  - Stop the robot (or disable heartbeat); wait until “last seen” is older than threshold; call health again; confirm robot is reported unreachable.  
  - In the UI, open Dashboard and Robots; confirm “online” status matches heartbeat-based health and that no requests are sent to robot `:8001/health` (check Network tab).

**Phase 2 done when:** Robot health in the UI is driven only by heartbeat data; no HTTP calls from Gateway (or frontend) to robot `/health`.

---

## Phase 3: Telemetry service (separate server) — heartbeat + read API

**Goal:** Telemetry ingest and read API live in a **separate process**. Gateway no longer receives robot push; it **queries** the Telemetry service for health (and later for joints/video). Robots push to the Telemetry service.

### Step 3.1 — New service: Telemetry server

- **Implement:**  
  - New app under `services/telemetry/` (or similar): FastAPI app with (1) **ingest:** `POST /ingest/heartbeat` (same payload as today), store last heartbeat per robot in memory or Redis; (2) **read:** `GET /health/summary` → `{ robot_id: { last_seen, reachable } }` using a “last seen within N seconds” rule.  
  - Config: port (e.g. 8001 or 9000), optional Redis URL.  
  - No dependency on Fleet or Gateway code; Gateway will call this service over HTTP.  
- **Test:**  
  - Start the Telemetry service.  
  - `curl -X POST .../ingest/heartbeat -d '{"robot_id":"r1","reachable":true}'`; then `curl .../health/summary` → expect `r1` with last_seen and reachable.  
  - Stop sending heartbeats; after threshold, `GET /health/summary` should show `r1` unreachable.

### Step 3.2 — Robots push to Telemetry service

- **Implement:**  
  - Robots already have `GATEWAY_URL` for heartbeat. Introduce `TELEMETRY_URL` (e.g. `http://host.docker.internal:9000`).  
  - Robot heartbeat loop POSTs to `TELEMETRY_URL/ingest/heartbeat` instead of (or in addition to, during migration) `GATEWAY_URL/api/telemetry/heartbeat`.  
  - Once Gateway no longer exposes ingest, robots use only `TELEMETRY_URL`.  
- **Test:**  
  - Start Telemetry service and one fake robot with `TELEMETRY_URL` set.  
  - Verify Telemetry service’s heartbeat store updates (e.g. `GET /health/summary` shows that robot).  
  - Verify no heartbeat requests hit the Gateway (if ingest is removed from Gateway).

### Step 3.3 — Gateway (BFF) queries Telemetry service for health

- **Implement:**  
  - Gateway config: `TELEMETRY_SERVICE_URL` (e.g. `http://localhost:9000`).  
  - `GET /api/robots/health/all` (and any per-robot health) → Gateway calls `GET <TELEMETRY_SERVICE_URL>/health/summary` and maps the response to the shape the frontend expects.  
  - Remove ingest from Gateway: drop `POST /api/telemetry/heartbeat` from the Gateway (or leave it as deprecated and unused once all robots use Telemetry service).  
- **Test:**  
  - Start Fleet, Gateway, Telemetry service, and one robot (pushing to Telemetry).  
  - Open Dashboard; confirm robot health in the UI.  
  - In Network tab, confirm frontend only calls Gateway; Gateway calls Telemetry service (server-side).  
  - Confirm `POST /ingest/heartbeat` hits Telemetry service, not Gateway.

**Phase 3 done when:** Telemetry is a separate server; robots push heartbeat to it; Gateway gets health only by querying the Telemetry service; frontend still gets health via Gateway.

---

## Phase 4: Telemetry read API for display (Gateway → Telemetry → UI)

**Goal:** Gateway exposes endpoints that “display telemetry” (e.g. last N joint samples, video URL for a task) by querying the Telemetry service and returning the result to the frontend.

### Step 4.1 — Telemetry service: minimal read API for display

- **Implement:**  
  - Add read endpoints that the Gateway can call, e.g.  
    - `GET /telemetry/joints?robot_id=&task_id=&limit=100` (return empty list or stub until you have real joint storage).  
    - `GET /telemetry/artifacts?task_id=` (return list of artifact metadata, e.g. video URL; stub if no storage yet).  
  - Implement only enough so the Gateway can proxy and the frontend can show “no data” or placeholder.  
- **Test:**  
  - `curl` Gateway → Telemetry for these endpoints; verify response shape.  
  - Optional: add a simple UI (e.g. on Execution or Robot detail) that calls Gateway `/api/telemetry/joints?...` or `/api/telemetry/artifacts?...` and displays “No data” or a table.

### Step 4.2 — Gateway: proxy telemetry for display

- **Implement:**  
  - Add Gateway routes, e.g. `GET /api/telemetry/joints`, `GET /api/telemetry/artifacts`, that forward query params to the Telemetry service and return the response to the client.  
  - Frontend (or Postman) calls only Gateway; Gateway calls Telemetry service.  
- **Test:**  
  - From browser or Postman, `GET /api/telemetry/joints?robot_id=...` via Gateway; verify response comes from Telemetry service.  
  - Confirm frontend never calls the Telemetry service directly (only Gateway).

**Phase 4 done when:** Any “display telemetry” request from the UI goes Browser → Gateway → Telemetry service → Gateway → Browser; Telemetry service is the single source of truth for telemetry read API.

---

## Phase 5: Full telemetry ingest (joints, video, storage) — later

**Goal:** Robots send joint streams and video to the Telemetry service; Telemetry service writes to your storage (time-series DB, object storage); read API returns real data for display and for training/export.

### Step 5.1 — Telemetry service: ingest streams and storage

- **Implement:**  
  - `POST /ingest/stream` or WebSocket for joint positions (and optionally logs), with `robot_id`, optional `task_id`/`plan_id`/`session_id`.  
  - Chunked upload or stream for video/images; write to object storage (S3/MinIO); store metadata (and URLs) in DB or in-memory index.  
  - Config for storage: time-series DB URL, bucket, credentials.  
- **Test:**  
  - Send sample joint payloads and a small “video” chunk to the Telemetry service; verify they are stored (query DB or object storage).  
  - Call `GET /telemetry/joints?...` and `GET /telemetry/artifacts?task_id=...` and verify returned data matches what was ingested.

### Step 5.2 — Robot-side senders (per-robot)

- **Implement:**  
  - Per-robot: ROS1/2 or Rust node that reads joint state (and optionally camera), and POSTs or streams to `TELEMETRY_URL/ingest/stream` (and upload endpoint for video).  
  - Tag with `robot_id`, and when running a task, `task_id`/`plan_id`/`session_id` so you can query by execution.  
- **Test:**  
  - Run one robot with the sender; run a task; verify Telemetry service receives and stores data; query via Gateway and show in UI (or export for training).

**Phase 5** can be broken into smaller steps (e.g. joints only first, then video) and scheduled after Phases 1–4 are stable.

---

## Implementation order and testing summary

| Phase | What you implement | How you test |
|-------|--------------------|--------------|
| **1** | Event-driven updates: Fleet → Gateway callback; Gateway event-driven WS; Frontend no refetchInterval | Trigger mutation; verify single WS message and UI update; no 1s loops. |
| **2** | Robot health from heartbeat only; Gateway stops polling robot /health | Health from heartbeat store; no requests to robot :port/health. |
| **3** | Telemetry service (new server): ingest heartbeat + read health/summary; robots push to it; Gateway queries it for health | Telemetry service runs; robots POST to it; Gateway GET health from it; UI unchanged. |
| **4** | Telemetry read API for display; Gateway proxies to Telemetry service | Browser → Gateway → Telemetry for joints/artifacts; frontend never talks to Telemetry. |
| **5** | Full ingest: joints, video, storage; robot senders | Ingest → storage; read API returns real data; optional UI or export. |

Dependencies: 2 depends on heartbeat existing (already in place). 3 depends on 2 conceptually (health from “telemetry” source). 4 depends on 3 (Telemetry service exists and has read API). 5 depends on 4 (read API shape) and can be done incrementally (joints first, then video).

This is the full set of main architectural components and the order in which to implement and test them.
