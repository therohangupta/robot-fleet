# Two Roles in One Box: BFF vs Telemetry Service

The current “gateway” is doing **two different jobs**. It’s useful to name them and to know how to split them. The **Telemetry** side is a full **server** (separate process) that handles both ingest and query of execution data—not just lightweight heartbeats.

---

## The two roles

| Role | Who talks to it | What it does | Direction of data |
|------|------------------|--------------|--------------------|
| **BFF / API Gateway** | **Frontend (and other clients)** | Single entry point for control and queries. Proxies to **fleet server** (gRPC) for plans/tasks/robots. **Queries the Telemetry service** for telemetry data (health, joints, video URLs, etc.) and passes it to the frontend for display. Handles auth, CORS, realtime fanout. | **Client → Gateway → Fleet** (control) and **Client → Gateway → Telemetry service** (telemetry for display) |
| **Telemetry service** (another **server**) | **Robots** (push) and **Gateway/BFF** (query) | **Ingest:** Receives push data from robots: heartbeats, joint positions, video, images, logs—used for **execution data collection and training**, not just light health. Writes to your storage (time-series DB, object storage, metadata in DB). **Query:** Exposes read APIs (e.g. last N joint samples, video URL for a task, health summary) so the **Gateway can query** and pass results to the frontend for display. | **Robot → Telemetry service → storage**; **Gateway → Telemetry service** (read) → **Gateway → Frontend** |

So:

- **BFF** = client-facing: route control to fleet, and **query telemetry from the Telemetry service** to show in the UI.
- **Telemetry service** = its own **server**: robot-facing ingest (heavy data: video, images, joints) + storage, and a **read API** so the Gateway can fetch “what to display” and pass it to the frontend. The frontend never talks to the Telemetry service directly—only the Gateway does.

---

## Why they ended up in one process

In the current repo, both live under `services/gateway` because:

- One less service to run and deploy.
- Telemetry started small (one HTTP endpoint: heartbeat). No need for a separate process yet.
- The BFF already had to expose “robot health” to the UI; reading from an in-memory heartbeat store in the same process was the simplest way.

So it’s a **convenience merge**, not a requirement. As soon as you have many robots, high-rate streams, or separate scaling/ownership (e.g. “telemetry team” vs “control-plane team”), splitting makes sense.

---

## Clean split: two services

If you want clear separation of concerns:

### 1) **API Gateway (BFF only)** — one server

- **Owns:** All client-facing API and WebSockets.
- **Talks to:** Fleet server (gRPC) and **Telemetry service** (HTTP read API). Does **not** receive data directly from robots.
- **Responsibilities:**
  - REST: `/api/plans`, `/api/tasks`, `/api/robots`, etc. → proxy to fleet.
  - REST: `/api/robots/health`, `/api/telemetry/...` (e.g. joints, video URL for a task) → **query Telemetry service**, then return to frontend for display.
  - WebSockets: `/ws/global-updates`, `/ws/execution/{plan_id}` → fed by events from fleet (or by querying fleet/DB when an event arrives).
  - Auth, CORS, rate limiting.
- **Displaying telemetry to the user:** BFF calls the Telemetry service (e.g. `GET /health/summary`, `GET /telemetry/joints?robot_id=&limit=`, `GET /telemetry/video?task_id=` or similar). BFF then passes that data to the frontend. Frontend never talks to the Telemetry service directly.
- **Does not:** Expose robot-facing ingest endpoints (heartbeat, streams). Those live on the Telemetry service.

### 2) **Telemetry service** — another server (separate process)

- **Owns:** All robot-push (ingest) and the pipeline robot → storage; plus **read APIs** so the Gateway can fetch data for the UI.
- **Talks to:** Robots (they call it to push); time-series DB, object storage, and optionally Postgres for metadata. **Does not** talk to the fleet for control.
- **Responsibilities:**
  - **Ingest (robot → Telemetry service):**
    - `POST /ingest/heartbeat`: accept heartbeat; store last-seen.
    - `POST /ingest/stream` or WebSocket: joint positions, logs; chunked upload for **video, images**. Normalize and write to **your configured storage** (time-series for joints, object storage for video/images). Tag with robot_id, and optionally plan_id/task_id/session_id for execution data collection and training.
  - **Query (Gateway → Telemetry service):**
    - e.g. `GET /health/summary` → `{ robot_id: { last_seen, reachable } }`.
    - e.g. `GET /telemetry/joints?robot_id=&task_id=&limit=` → last N joint samples for display.
    - e.g. `GET /telemetry/video?task_id=` or `/telemetry/artifacts?task_id=` → URL or metadata for video/images so the Gateway can pass a link or blob URL to the frontend for display.
  - Telemetry is **not** just lightweight: it includes execution data (video, images, joint streams) for collection and training; storage is real (object storage, time-series DB), not just in-memory.
- **Does not:** Expose `/api/plans`, `/api/robots` (list from DB), or any control endpoints. No gRPC to fleet.

### 3) **Flow after split**

- **Control / observe (UI state):**  
  Browser → **API Gateway (BFF)** → Fleet server → DB.  
  Robots are only involved when the **fleet** calls them for `/do_task`.

- **Telemetry ingest (robot → storage):**  
  Robot → **Telemetry service** (heartbeat, joints, video, images).  
  Telemetry service → your storage (time-series, object storage, metadata).

- **Telemetry display (UI):**  
  Browser → **API Gateway (BFF)** → **Telemetry service** (query: health, joints, video URL, etc.) → BFF → Browser.  
  So the Gateway **queries** the Telemetry service and passes the results to the frontend for displaying to the user. The Telemetry service is **another server**; the frontend never talks to it directly.

So you have:

- **One server:** BFF (frontend ↔ fleet, and frontend ↔ telemetry *via BFF querying Telemetry service*).
- **One server:** Telemetry service (robot → ingest → storage; and BFF → query → data for display).

---

## Summary

| Question | Answer |
|----------|--------|
| Is the gateway doing too much? | Yes. It’s two roles: **BFF** (client → fleet, client → telemetry *for display*) and **telemetry ingest + storage** (robot → store). |
| Will telemetry be another server? | **Yes.** The Telemetry **service** is a separate server (process). It handles ingest from robots (heartbeat, joints, video, images) and writes to your storage; it also exposes **read APIs** so the Gateway can query it and pass data to the frontend for display. |
| Is telemetry only lightweight (heartbeats)? | **No.** Telemetry includes **execution data** for collection and training: video, images, joint positions, logs. Storage is real (object storage, time-series DB), not just in-memory. |
| How does the frontend get telemetry to display? | Frontend → **Gateway (BFF)** → **Telemetry service** (query) → Gateway → Frontend. The Gateway queries the Telemetry service (e.g. health summary, last N joints, video URL for a task) and passes the response to the frontend. The frontend never talks to the Telemetry service directly. |
| How to separate? | Split into **API Gateway (BFF)** and **Telemetry service** (another server). BFF talks to fleet (control) and to Telemetry service (query for display). Robots only push to the Telemetry service. |
| Do you have to split now? | No. You can keep one “gateway” process and still separate the code (e.g. `gateway/bff/` vs `gateway/telemetry/`) so the boundaries are clear and you can split into two processes later without rewriting logic. |

If you want, the next step can be: (1) rename or document the two parts inside `services/gateway` (e.g. routers and roles), and (2) add a one-page “deploy split” note (two Docker images, two ports, robots point to the Telemetry service URL, frontend points to the BFF URL).
