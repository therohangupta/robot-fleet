# What’s Required to Stop Using Polling (Except Heartbeat)

Heartbeat is already push-based (robot → gateway). Everything below is about making the rest event-driven so no timer-based polling is needed.

---

## 1. Fleet server: emit events when state changes

**Today:** The fleet server (and its executor) update the DB only. Nothing notifies the gateway when data changes.

**Required:** After every mutation that the UI should reflect, the fleet must emit an event. Emit at least:

- **Task state change:** `task_id`, `plan_id`, `status` (and optionally `robot_id`, `result`) — after `update_task_status` / `update_task` in the executor and in gRPC handlers (CreateTask, UpdateTask, DeleteTask).
- **Plan state change:** `plan_id`, `status` (e.g. executing / completed / failed) — after executor sets plan execution_status, and after plan create/update/delete.
- **Robot/list change:** after register/unregister robot, so the UI can refresh robot list and allocations.

**Ways to do it (pick one):**

- **A) HTTP callback (no new infra)**  
  - Fleet is configured with a “gateway event URL” (e.g. `http://gateway:8000/internal/events`).  
  - After any of the mutations above, fleet does `POST <url>` with body e.g. `{ "type": "task.state_changed", "plan_id": 1, "task_id": 2, "status": "completed" }`.  
  - Gateway exposes that endpoint and pushes to WebSocket clients (see below).  
  - Pros: simple, no Redis/NATS. Cons: fleet must know gateway URL; gateway must be reachable from fleet.

- **B) Redis (or NATS) pub/sub**  
  - Fleet publishes to a channel (e.g. `fleet:events`) on each mutation.  
  - Gateway subscribes and, on message, pushes to WS clients.  
  - Pros: decoupled, scalable. Cons: need to run Redis (or NATS).

- **C) Shared DB / “last event” table**  
  - Fleet writes a row to an `events` table (or bumps a “version”) on each mutation.  
  - Gateway polls that table/version on an interval (or uses LISTEN/NOTIFY if Postgres).  
  - This is still a form of polling (DB poll), so it doesn’t fully remove polling; use only if you can’t do A or B.

**Recommendation for v2:** Start with **A (HTTP callback)**. Add B later if you need decoupling or multiple gateways.

---

## 2. Gateway: event-driven WebSocket, no sleep loops

**Today:**

- `ws/global-updates`: `while True` → (optionally fetch data) → `send_json({ type: "invalidate", queries: [...] })` → `asyncio.sleep(1)`.
- `ws/execution/{plan_id}`: `while True` → `bridge.list_tasks(plan_ids=[plan_id])` → `send_json({ type: "tasks_update", ... })` → `asyncio.sleep(1)`.

**Required:**

1. **In-memory event bus (or callback handler)**  
   - When the gateway receives an event (from fleet via HTTP callback or from Redis), it pushes to the right WS connections:
     - For `task.state_changed` / `plan.state_changed`: send to every client subscribed to `ws/global-updates` (invalidation or small payload) and to every client subscribed to `ws/execution/{plan_id}` for that `plan_id` (e.g. send updated task list or a “tasks_changed” event so the client refetches).
     - For `robot.*`: send to `ws/global-updates` so the UI invalidates robot/list and robot-health.

2. **Replace the two `while True` loops**  
   - **Global updates:**  
     - On connect: send `{ type: "connected" }` once.  
     - Then **do not** run a loop. Instead, when the gateway receives an event from the fleet (or Redis), call a broadcast that sends one message to all clients connected to `ws/global-updates`, e.g. `{ type: "invalidate", queries: ["plans", "robots", "robot-allocations"] }` (and optionally `"robot-health"` when you use heartbeat for health).  
   - **Execution per plan:**  
     - On connect: send one initial `tasks_update` (fetch tasks once via bridge).  
     - Then **do not** run a loop. When the gateway receives a `task.state_changed` or `plan.state_changed` for that `plan_id`, fetch tasks for that plan once and send one `tasks_update` to all clients connected to `ws/execution/{plan_id}`.

3. **Internal HTTP endpoint (if using callback A)**  
   - e.g. `POST /internal/events` (or `/api/internal/events`), auth’d (e.g. shared secret or network-only), body: `{ "type": "task.state_changed", "plan_id": 1, "task_id": 2, "status": "completed" }`.  
   - Handler parses the event and calls the same broadcast logic as above (so one code path for “something changed”).

Result: no `asyncio.sleep(1)` in the gateway; messages are sent only when something actually changes.

---

## 3. Frontend: no refetch intervals; rely on WS invalidation

**Today:**

- `main.tsx`: default `refetchInterval: 5000` for all queries.
- `Dashboard.tsx`: `robot-health` query with `refetchInterval: 5000`.
- `Execution.tsx`: tasks with `refetchInterval: 1000`, robots with `refetchInterval: 2000`.
- Plans / PlanDetails / Goals / Robots: use `useRealtimeUpdates()` and no refetchInterval (good).

**Required:**

1. **main.tsx**  
   - Remove default `refetchInterval` (or set to `false`). Keep a short `staleTime` if you want refetch-on-window-focus only.

2. **Dashboard**  
   - Use `useRealtimeUpdates()` so it receives invalidation when gateway sends it.  
   - Remove `refetchInterval` from the `robot-health` query.  
   - When gateway sends `invalidate` including `"robot-health"`, the dashboard will refetch once. (To avoid *any* polling for health, gateway should derive robot health from heartbeat and send invalidation when heartbeat data changes; see below.)

3. **Execution page**  
   - Use `useRealtimeUpdates()` and, when the Execution page is mounted, ensure the app is subscribed to `ws/execution/{planId}` (or that global invalidation includes the tasks query for that plan).  
   - Remove `refetchInterval` for tasks and robots.  
   - When the gateway sends a `tasks_update` (or an invalidation for that plan’s tasks), React Query refetches and the view updates.

4. **useRealtimeUpdates**  
   - When it receives `invalidate` with `queries: ["tasks", ...]`, it should invalidate the right keys. For execution, the query key is typically `['tasks', planId]`. So either:  
     - Gateway sends something like `invalidate: { "queries": ["tasks"] }` and the frontend invalidates all queries whose key starts with `tasks`, or  
     - Gateway sends `invalidate: { "queries": ["tasks"], "plan_id": 5 }` and the frontend invalidates `['tasks', 5]`.  
   - Same idea for `robot-health`, `plans`, `robots`, `robot-allocations`: ensure the list of query keys the frontend invalidates matches what the gateway sends and what the pages use.

Result: no timer-based refetch; updates only when the gateway pushes an invalidation (or a payload) after a real event.

---

## 4. Robot health: use heartbeat so gateway doesn’t poll robots

**Today:** Gateway (or UI) still calls each robot’s `/health` on a schedule for “robot-health”.

**Required:**

- Gateway **stops** polling robots for health.  
- Gateway uses the **heartbeat** data it already stores from `POST /api/telemetry/heartbeat` to derive “last seen” / “reachable” per robot.  
- Expose that to the UI (e.g. `GET /api/robots/health/all` returns data from the heartbeat store, not from live HTTP calls to each robot).  
- When a new heartbeat arrives (or when you detect “last seen” is stale), gateway sends an invalidation for `"robot-health"` (or pushes a small payload) so the dashboard updates once.

Result: no polling of robot `/health`; only heartbeat push from robot → gateway, then event-driven update to the UI.

---

## 5. Summary checklist

| Layer        | Change |
|-------------|--------|
| **Fleet**   | Emit event (HTTP callback or Redis) after task/plan/robot mutations (executor + gRPC handlers). |
| **Gateway** | Add event receiver (e.g. `POST /internal/events`). Replace both WS loops with “on event → broadcast to relevant WS clients”. Optionally: derive robot health from heartbeat and expose it; stop calling robot `/health`. |
| **Frontend**| Remove default and per-query `refetchInterval`. Use `useRealtimeUpdates()` on Dashboard and Execution; ensure invalidation keys match (e.g. `tasks`, `robot-health`, `plans`, `robots`). |
| **Robot**   | No change for “no polling” (heartbeat already push). |

After this, the only “repeating” work is the robot’s own heartbeat send (which is push, not polling). Everything else is event-driven: fleet → event → gateway → WS → frontend refetch/invalidate.
