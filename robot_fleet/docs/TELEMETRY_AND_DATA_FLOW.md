# Telemetry, Data Flow, and Robot → Storage Pipeline

Telemetry is **not** just lightweight heartbeats. It includes **execution data** for collection and training: video, images, joint positions, logs. The **Telemetry service** is a **separate server** that ingests from robots, writes to your storage, and exposes **read APIs** so the **Gateway (BFF)** can query it and pass data to the frontend for display. The frontend never talks to the Telemetry service directly.

---

## 1. What handles telemetry today (v2)

**Only the Gateway** has telemetry logic (as a stopgap):

- **`services/gateway/src/routers/telemetry.py`**
  - **`POST /api/telemetry/heartbeat`**: accepts `{ robot_id, reachable, busy?, ts? }` from robots.
  - Stores the last heartbeat per robot in memory (`_heartbeats`).
  - No persistence, no joint/video, no forwarding to your own storage.

So today, “telemetry” = **heartbeats only**. Joint positions, video, and logs are **not** implemented. The target is a dedicated **Telemetry service** (another server) for ingest + storage + query; the Gateway will **query** that service and pass results to the frontend for display.

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
│  Central telemetry ingest (Gateway or dedicated service)                     │
│  - Receives streams per robot (and optionally tags with robot_id, plan_id,  │
│    task_id, session_id for later query).                                     │
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

**Displaying telemetry in the UI:** The **Telemetry service** is another **server**. It exposes **read APIs** (e.g. health summary, last N joints for a robot/task, video URL for a task). The **Gateway (BFF)** calls those APIs and passes the response to the frontend for display. So: **Browser → Gateway → Telemetry service (query) → Gateway → Browser**. The frontend never talks to the Telemetry service directly.

- **Who handles telemetry:** a dedicated **Telemetry service** (another server): ingest from robots, write to your storage, and expose **read APIs**. Today a minimal version lives in the Gateway (`services/gateway/src/routers/telemetry.py`); the target is to move ingest + storage + read API to a separate Telemetry service that the Gateway **queries** for display.
- **Flow of data (ingest):** robot (ROS/Rust collector) → Telemetry service → your storage. **Flow of data (display):** Frontend → Gateway → Telemetry service (query) → Gateway → Frontend. The frontend never talks to the Telemetry service or the robot for telemetry; the Gateway queries the Telemetry service and passes data (summaries, joint samples, video URLs) to the frontend for display.

Concretely:

1. **Robot side (your responsibility per robot)**
   - Keep the existing task server for **control** (`/do_task`, `/health`, heartbeat).
   - Add a **telemetry sender** (ROS node, Rust binary, or another process) that:
     - Subscribes to joint state, camera, etc. (ROS1/2 or hardware APIs).
     - Pushes to the central ingest:
       - **Low-rate / small payloads** (e.g. joint positions, battery): HTTP POST or WebSocket to something like `POST /api/telemetry/stream` or `WS /api/telemetry/stream` with `robot_id` and optional `plan_id`/`task_id`/`session_id`.
       - **High-rate or large** (e.g. video): either chunked HTTP uploads to the same ingest, or stream to a **media service** (e.g. RTSP/WebRTC to a recorder that writes to object storage). The ingest or media service then writes to **your specified storage** (config-driven: bucket, path, DB table).

2. **Central ingest (Gateway or dedicated service)**
   - **Already:** `POST /api/telemetry/heartbeat` (keep it).
   - **Add:** e.g. `POST /api/telemetry/stream` (and/or WebSocket) for structured telemetry:
     - Body or messages: `robot_id`, optional `plan_id`, `task_id`, `session_id`, `ts`, and payload (e.g. joint positions, log lines).
     - Validate/normalize (e.g. canonical schema for “joint_state”).
     - Forward to:
       - **Time-series DB** (you configure connection and retention).
       - **Object storage** (you configure bucket/prefix; e.g. `s3://your-bucket/telemetry/{robot_id}/{date}/{stream_id}.json` or `.bin`).
   - **Video:** either the same service accepts binary chunks and writes to object storage, or a separate **media/recording** component (e.g. RTSP/WebRTC sink) writes to your bucket; ingest only stores **metadata** (e.g. `robot_id`, `task_id`, `url`, `start_ts`, `end_ts`) in Postgres or your DB so you can query “all videos for this task”.

3. **Storage you specify**
   - Configure the ingest (or media service) with:
     - Time-series DB URL and schema (if used).
     - Object storage bucket + region/credentials (and optional path template).
     - Optionally, Postgres (or existing fleet DB) for metadata and indexes.
   - So “where it’s stored” is **fully under your control** via config; the code path is: robot → central ingest → your storage.

4. **Frontend**
   - For “talking to one robot” in the sense of **control**: unchanged (Send Task → robot; Execute Plan → Gateway → Fleet → robot).
   - For **telemetry**: no direct robot↔frontend telemetry. Frontend can:
     - Call Gateway (or a separate API) that **reads from your storage** (e.g. “last N joint samples for robot X”, “list of videos for task Y”) for dashboards and debugging.
     - Optionally, live view via a **stream** that the Gateway (or media service) fans out from the ingest or from the recorder, so the robot still sends once to the ingest.

This keeps a single path: **robot → central ingest → your specified storage**, and each robot only needs to know how to **send** to that ingest (and how to read from ROS/Rust on its side).

---

## 4. Summary

| Question | Answer |
|----------|--------|
| **What handles telemetry?** | **Gateway** today: only `routers/telemetry.py` (heartbeat). For full telemetry, the same component (or a dedicated ingest service) should receive streams and write to your storage. |
| **Frontend → one robot flow** | **Control:** (1) “Send Task” = frontend → robot `POST /do_task`; response = `TaskResult`. (2) Execute plan = frontend → Gateway → Fleet → robot `/do_task`; “data back” = task status/result via DB and Gateway APIs/WS. **Telemetry:** not implemented; target is robot → ingest → your storage; frontend queries storage/API, not the robot. |
| **Joint / video / storage** | Robot uses ROS1/2 or Rust to read data → sends to central ingest (Gateway or service) → ingest writes to **your configured** time-series DB and object storage; optional metadata in Postgres. Each robot implements only the reader/sender; storage location is configured in the central ingest. |

If you want, next step can be a short “Telemetry ingest API” spec (e.g. `POST /api/telemetry/stream` schema and a minimal storage adapter interface) and where in `robot_fleet` to add it (e.g. under `services/gateway/src/routers/telemetry.py` and a new `services/gateway/src/telemetry/` for storage wiring).
