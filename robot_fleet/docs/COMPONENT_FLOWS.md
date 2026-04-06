# Who Talks to Whom: Gateway, Fleet Server, Robots

Clear picture of the three main backend pieces and how data flows.

---

## The three components

| Component | What it is | Typical port |
|-----------|------------|--------------|
| **Fleet server** | Central control plane. Owns the database (plans, tasks, robots, goals). Runs plan execution (sends tasks to robots). Exposes **gRPC** only. | 50051 |
| **Gateway** | Single entry point for **all clients** (browser, mobile, CLI). Exposes **REST + WebSockets**. Proxies control to the fleet; receives heartbeats/telemetry from robots. | 8000 |
| **Robot** | One per physical/simulated robot. Runs a small HTTP server: `/do_task`, `/health`, and (optionally) sends heartbeats/telemetry **to the gateway**. | 8001, 8002, … |

---

## Rule of thumb

- **Clients (e.g. browser)** → talk **only** to the **Gateway**.
- **Gateway** → talks to the **Fleet server** (for all control and read-only data).
- **Fleet server** → talks to **Robots** (to send tasks: “do this”).
- **Robots** → talk to the **Gateway** (to push heartbeats/telemetry). They do **not** talk to the Fleet server for that.

So: **Gateway** is the only thing that talks to both the Fleet server and the Robots (in different directions).

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
         │  gRPC: CreatePlan, StartExecution, RegisterRobot, etc.
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
         │  gRPC (or bridge to DB): ListPlans, ListTasks, ListRobots, GetPlan, …
         ▼
  ┌──────────────┐
  │ Fleet Server │ ──────►  Database: read plans, tasks, robots, goals
  └──────────────┘
```

- **Request flow:** Client → Gateway → Fleet server (which reads from DB).
- **Data flow:** Fleet returns data to Gateway; Gateway returns it to the client. **Robots are not in this path** for “list plans / tasks / robots.” (Robot health can come from heartbeat data stored in the Gateway; see below.)

---

### 3) Robot → Gateway: heartbeats (and future telemetry)

```
  ┌──────────────┐
  │    Robot     │
  └──────────────┘
         │
         │  HTTP POST: /api/telemetry/heartbeat  { robot_id, reachable, busy }
         │  (every N seconds; no one polls the robot)
         ▼
  ┌──────────────┐
  │   Gateway    │  stores last heartbeat per robot; can expose via GET /api/robots/health/all
  └──────────────┘
```

- **Request flow:** Robot → Gateway only. Fleet server is **not** in this path.
- **Data flow:** Robot pushes “I’m alive, busy/idle” to Gateway. Gateway stores it (and can later forward a summary to Fleet if you want “reconciliation”).

So: **Robots talk to the Gateway** for heartbeats (and future telemetry like joints/video). They do **not** send heartbeats to the Fleet server.

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

| Who            | Talks to Fleet server?      | Talks to Gateway?           | Talks to Robot?        |
|----------------|-----------------------------|-----------------------------|------------------------|
| **Browser**    | No                          | Yes (all API + WS)         | Yes, only “Send Task”  |
| **Gateway**   | Yes (all control + reads)   | —                           | No (doesn’t call robot)|
| **Fleet server** | —                         | No                          | Yes (during execution) |
| **Robot**     | No                          | Yes (heartbeat/telemetry)  | —                      |

So:

- **Gateway** talks to **Fleet server** (and receives from **Robots**).
- **Fleet server** talks to **Robots** (and DB).
- **Robots** talk only to **Gateway** for push data (heartbeat/telemetry); they respond to **Fleet server** (and optionally browser) for `/do_task`.
