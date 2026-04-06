# Communication Architecture Review: Polling vs Event-Driven

Revisit after many days: what’s done, what’s wasteful, and what’s left for the frontend, gateway, fleet server, and robot servers.

---

## 1. What’s Been Done and Changed

### Frontend ↔ Gateway (single UI boundary)
- **Done:** UI talks only to the gateway via `/api/*` and `/ws/*` (no direct calls to fleet or robots).
- **Done:** `useRealtimeUpdates()` exists and is used on Plans, PlanDetails, Robots, Goals. It connects to `ws/global-updates` and invalidates React Query when it receives `type: "invalidate"` with `queries: ["robot-health", "plans", "robots", "robot-allocations"]`.
- **Done:** PlanDetails and Plans explicitly avoid `refetchInterval` and rely on WebSocket-driven invalidation (comments: “No refetchInterval - using WebSocket real-time updates”).

### Gateway WebSocket “real-time”
- **Current behavior:** The gateway does **not** push updates when something changes. It runs a loop that **polls** every 1 second and then sends an invalidation message.
  - `dashboard/backend/routers/websocket.py`:
    - `ws/global-updates`: `while True` → (optionally fetch robots/plans) → `send_json({ type: "invalidate", queries: [...] })` → `await asyncio.sleep(1)`.
    - `ws/execution/{plan_id}`: `while True` → `bridge.list_tasks(plan_ids=[plan_id])` → `send_json({ type: "tasks_update", ... })` → `await asyncio.sleep(1)`.
- So “real-time” is **polling on the server**, then pushing that poll result over WebSocket. It’s still wasteful and not event-driven.

### Where polling / refetch is still used (wasteful unless necessary)
- **`dashboard/frontend/src/main.tsx`**  
  - `refetchInterval: 5000` and `staleTime: 2000` on the **default** QueryClient.  
  - Effect: every query refetches at least every 5s unless a page overrides it. Redundant when WS-driven invalidation is supposed to drive refetches.

- **`dashboard/frontend/src/pages/Dashboard.tsx`**  
  - Robot health: `queryKey: ['robot-health']`, `refetchInterval: 5000` and direct `fetch('/api/robots/health/all')`.  
  - Dashboard does **not** use `useRealtimeUpdates()`, so it never gets invalidation for `robot-health`. So this page is **fully polling** for health every 5s.

- **`dashboard/frontend/src/pages/Execution.tsx`**  
  - Tasks: `refetchInterval: 1000` (every second).  
  - Robots: `refetchInterval: 2000`.  
  - This page does **not** use `useRealtimeUpdates()`, and even if it did, the global WS only sends generic invalidation keys like `"plans"` / `"robots"`, not `["tasks", planId]`. So Execution is **fully polling** and is the heaviest poller.

### Fleet server (central server)
- **Done:** Writes all plan/task/robot truth to Postgres via `instance_registry` (e.g. `update_task_status`, `update_task`, `update_plan`). Executor updates task status and plan execution status as execution runs.
- **Gap:** The fleet server **never notifies the gateway** when state changes. There is no event bus, no callback, no “push” from fleet → gateway. So the gateway cannot do true event-driven push; it can only poll the bridge/DB.

### Robot servers (Docker / fake)
- **Done:** Robots expose HTTP (e.g. `/do_task`, `/health`). Fleet executor calls robots via `RobotClient` (HTTP) to run tasks and gets success/failure and replan flag.
- **Gap:** Robots do **not** push anything to the central server or gateway:
  - No heartbeats.
  - No “task started / task completed” status pushes.
  - No telemetry.  
  So “robot status” and “robot health” are entirely determined by:
  - What the fleet server has in the DB (task assignments), and
  - What the gateway (or UI) **polls** via `/api/robots/health/all` (HTTP calls from gateway to each robot).

---

## 2. Summary: What Is Wasteful Today

| Layer | What’s happening | Why it’s wasteful |
|-------|------------------|-------------------|
| **Gateway** | `ws/global-updates` and `ws/execution/{plan_id}` both run `while True` + `asyncio.sleep(1)` and send messages every second | Constant work and network even when nothing changed. Not event-driven. |
| **Frontend default** | QueryClient `refetchInterval: 5000` | All queries refetch every 5s unless overridden; duplicates WS invalidation intent. |
| **Dashboard page** | `robot-health` with `refetchInterval: 5000` + no WS | Health is polled every 5s; no invalidation from WS. |
| **Execution page** | Tasks 1s, robots 2s `refetchInterval` + no WS | Heavy polling; no event-driven updates. |
| **Robot health** | Gateway (or UI) calls each robot’s `/health` on a schedule | Repeated HTTP probes to every robot; robots don’t push. |

---

## 3. What’s Remaining for the Frontend

- **Remove default polling:** In `main.tsx`, remove or greatly increase default `refetchInterval` (e.g. 60s or false) and rely on `useRealtimeUpdates()` + explicit invalidation for “live” data. Keep a short `staleTime` if you want refetch-on-focus.
- **Dashboard:** Either:
  - Use `useRealtimeUpdates()` and include `robot-health` in the invalidation list when gateway sends it, and **remove** `refetchInterval` for `robot-health`, or  
  - If health stays gateway-polled for now, at least document that this is the only intentional poll and consider a longer interval (e.g. 15–30s).
- **Execution page:**
  - Prefer **event-driven**: connect to `ws/execution/{planId}` and consume `tasks_update` (or future event types). On message, invalidate only `['tasks', planId]` (and optionally `['robots']`) instead of polling.
  - Remove `refetchInterval` for tasks and robots on this page once WS pushes real updates (or invalidation) on task/plan changes.
- **Consistency:** Any page that should feel “live” (Plans, PlanDetails, Execution, Dashboard, Robots, Goals) should either:
  - Use `useRealtimeUpdates()` and have the gateway send invalidation for the relevant query keys, and **not** use `refetchInterval`, or  
  - Have a clear reason for a single intentional poll (e.g. health every 30s) and document it.

---

## 4. What’s Remaining for Gateway ↔ Fleet (Event-Driven)

- **Fleet → Gateway events:** Today the fleet server does not push. To remove gateway polling:
  - Fleet must **emit events** when things change (e.g. task status, plan status, allocation changes). Options:
    - In-process: gateway subscribes to a fleet-side callback/queue (only works if gateway and fleet run in the same process or you add a channel).
    - Out-of-process: use a small event bus (e.g. Redis pub/sub, Redis Streams, or NATS). Fleet publishes “task.state_changed”, “plan.state_changed”, etc.; gateway subscribes and forwards to WebSocket clients.
  - Gateway then **replaces** the `while True` + `sleep(1)` loops with:
    - On event: send one message (or invalidation) to the right WS connections.
    - Optionally: send a snapshot on WS connect, then only deltas/events.
- **Schema:** Align with `DESIGN.md`: e.g. `task.state_changed`, `plan.state_changed`, and later `robot.heartbeat` / `robot.state_changed`. Keep payloads small; use references for large data.

---

## 5. What’s Remaining for Robot Servers → Central Server / Gateway

- **Today:** Robots only respond to fleet’s HTTP calls (`/do_task`, and `/health` when something polls them). They do not push.
- **Recommended (from DESIGN.md):**
  - **Heartbeat / status:** Robots should periodically push “I’m alive” (and optionally busy/idle, battery, faults) to the **central server** or a **telemetry endpoint** on the gateway. The fleet (or gateway) then updates “last seen” / effective status and can reconcile with DB state.
  - **Protocol:** Low-rate heartbeats can be HTTP POST or a single WebSocket connection from robot to gateway/fleet. High-rate telemetry (joint state, logs) can be a separate WS or stream.
  - **Who receives:** Either:
    - Fleet server: receives heartbeats, updates DB or in-memory “robot status”, and emits `robot.state_changed` for the gateway to fan out, or  
    - Gateway: exposes a small “robot telemetry” endpoint, receives heartbeats, and pushes to UI and/or forwards to fleet.  
  Design choice: fleet remains source of truth for “assigned tasks”; robot is source of truth for “reachable / busy / fault”. Reconciled view lives in fleet or gateway and is pushed to UI via events.
- **Implementation order:** Start with a simple heartbeat (e.g. POST every 10–30s) from each robot to fleet or gateway; then replace UI/gateway health **polling** with “last heartbeat” + optional on-demand health check only when needed.

---

## 6. Quick Reference: Current vs Desired

| Concern | Current | Desired |
|--------|--------|--------|
| **UI → Gateway** | Only `/api` and `/ws`; single boundary | Keep as is. |
| **Live data (plans, tasks, robots)** | Mix: some pages use WS invalidation, others poll. Gateway WS is a 1s poll loop. | Event-driven: fleet emits; gateway forwards; no gateway or frontend polling loops. |
| **Default refetch** | 5s global refetchInterval | No default polling; refetch on invalidation or on focus. |
| **Execution page** | 1s/2s polling | WS `execution/{planId}` with real updates or invalidation; no refetchInterval. |
| **Robot health** | Gateway/UI polls `/health` per robot | Robots push heartbeats; fleet/gateway expose “last seen”; optional rare on-demand check. |
| **Fleet → Gateway** | No push | Events (task/plan/robot) via queue or in-process subscription. |
| **Robots → Fleet/Gateway** | No push | Heartbeat (and later telemetry) from robot to central server or gateway. |

---

## 7. Files to Touch (for “no polling unless necessary”)

- **Frontend:**  
  `dashboard/frontend/src/main.tsx` (default refetchInterval),  
  `dashboard/frontend/src/pages/Dashboard.tsx` (robot-health),  
  `dashboard/frontend/src/pages/Execution.tsx` (tasks/robots refetchInterval; optionally use WS execution channel).
- **Gateway:**  
  `dashboard/backend/routers/websocket.py` (replace loop with event-driven send: either subscribe to fleet events or a queue, or keep a single “state changed” channel from fleet).
- **Fleet (later):**  
  Emit events on `update_task_status`, `update_task`, `update_plan` (and optionally on robot heartbeat receipt). No change to executor logic, only “after DB write, publish event”.
- **Robots (later):**  
  Add a small heartbeat (e.g. POST to fleet or gateway every N seconds) and optionally a “task completed” push so the fleet can update DB and then emit; reduces need for polling.

This keeps your architecture aligned with DESIGN.md and removes unnecessary polling while leaving room for the few places where a slow, intentional poll might still be acceptable (e.g. health fallback every 30s).
