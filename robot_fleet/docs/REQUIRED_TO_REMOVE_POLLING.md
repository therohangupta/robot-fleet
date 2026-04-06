# What’s Required to Stop Using Polling (Except Heartbeat)

Heartbeat remains push-based (robot → telemetry; gateway reads aggregated health from telemetry). This doc tracks **what is already event-driven** vs **what still uses timers**.

---

## Status overview

| Area | Status |
|------|--------|
| Fleet → gateway events (HTTP) | **[DONE]** |
| Gateway WebSockets (global + execution) | **[DONE]** (no `sleep(1)` loops; see §2) |
| Telemetry service + health fan-out | **[DONE]** |
| Gateway reads health from telemetry | **[DONE]** |
| Dashboard / most pages | **[DONE]** — `useRealtimeUpdates()`, no default `refetchInterval` |
| Execution page — plan query | **TODO** — conditional `refetchInterval` while executing |

---

## 1. Fleet server: emit events when state changes — **[DONE]**

**Implemented:** After mutations, the fleet server POSTs to the gateway via **`GATEWAY_EVENT_URL`** (see `packages/config.py`). Emitter: `services/fleet_server/src/events.py` (`emit_task_changed`, `emit_plan_changed`, `emit_robot_changed`).

**Wiring (non-exhaustive):**

- `services/fleet_server/src/executor/executor.py` — task and plan status transitions.
- `services/fleet_server/src/service.py` — gRPC paths for robots, tasks, plans.

**Original options (A/B/C):** **A (HTTP callback)** is what shipped. B/C remain optional if you need a bus or stronger decoupling later.

---

## 2. Gateway: event-driven WebSocket — **[DONE]**

**Implemented:**

- **`POST /internal/events`** — accepts fleet (and telemetry) events; maps types → React Query key lists; notifies subscribers.  
  Code: `services/gateway/src/routers/websocket.py`
- **Per-subscriber `asyncio.Queue`s** — no broadcast loss when many clients wake together; not a global `sleep` poll loop.
- **`/ws/global-updates`** — on connect sends `connected`; then waits on the queue. **Keepalive:** `asyncio.wait_for(..., timeout=30)` sends a `ping` if idle — that is connection liveness, not data polling.
- **`/ws/execution/{plan_id}`** — initial connect + task list updates when events arrive (bridge `list_tasks` runs **on event**, not on a 1s timer). Same queue + 30s ping pattern as global.

**Note:** Handlers still use `while True` around **`queue.get()`** — that is normal for long-lived WebSockets, not the old “wake every second and push” pattern.

**Optional hardening (not required to drop polling):** filter execution WS refreshes by `plan_id` when the event carries it, so unrelated plans do less work.

---

## 3. Frontend: refetch intervals and WebSocket invalidation

### **[DONE]** Global query defaults

- **`services/dashboard-web/src/main.tsx`** — no default `refetchInterval` (only `staleTime: 2000` on the `QueryClient`).

### **[DONE]** Dashboard

- **`services/dashboard-web/src/pages/Dashboard.tsx`** — uses `useRealtimeUpdates()`; `robot-health` query has **no** `refetchInterval` (invalidation via gateway events, including telemetry-driven `telemetry.health_changed` → `robot-health`).

### **[DONE]** Other listing pages

Plans, Plan details, Goals, Robots, etc. use `useRealtimeUpdates()` and avoid timer refetch where documented in code (e.g. comments in `Plans.tsx`, `PlanDetails.tsx`, `Robots.tsx`).

**Invalidation hook:** `services/dashboard-web/src/lib/api.ts` — `useRealtimeUpdates()` invalidates query keys that match gateway `invalidate.queries` (e.g. `tasks`, `plans`, `robots`, `robot-health`).

### **TODO** — Execution page

**File:** `services/dashboard-web/src/pages/Execution.tsx`

- **[DONE]** Uses `useRealtimeUpdates()` and **`GatewayRealtimeClient`** for `ws/execution/{planId}` (`tasks_update` path).
- **[DONE]** `tasks` / `robots` queries use no `refetchInterval` in the current code.
- **TODO:** The **`plan`** query still uses a **conditional `refetchInterval`** (~3s) while `execution_status === 'executing'`. To remove timer-based polling entirely, rely on WebSocket invalidation / `tasks_update` (and plan invalidation when the gateway includes `plans` on relevant events) and drop this interval.

---

## 4. Robot health: telemetry + gateway — **[DONE]**

**Implemented:**

- **Telemetry** is a separate service under **`services/telemetry/src/`** (heartbeats, health summary, `health_changed` fan-out to the gateway — see `services/telemetry/src/events.py`, publishing/ingest routers).
- **Gateway** reads robot health from telemetry via **`services/gateway/src/services/telemetry_client.py`** (`TELEMETRY_URL`), not by polling each robot’s `/health` in the default path. Robot routes default to `source=telemetry` where applicable (`services/gateway/src/routers/robots.py`).
- Gateway WebSocket mapping includes **`telemetry.health_changed` → `robot-health`** (`services/gateway/src/routers/websocket.py`).

**Note:** Direct per-robot HTTP health checks may still exist as **fallback** (`source=direct` / `services/gateway/src/services/robot_health.py`) — not the primary “no polling” architecture.

---

## 5. Summary checklist

| Layer | Status | Notes |
|-------|--------|--------|
| **Fleet** | **[DONE]** | HTTP events via `GATEWAY_EVENT_URL`; `services/fleet_server/src/events.py` + call sites in executor / gRPC service. |
| **Gateway** | **[DONE]** | `services/gateway/src/routers/websocket.py` — `/internal/events`, event bus, WS endpoints. Telemetry client: `services/gateway/src/services/telemetry_client.py`. |
| **Telemetry** | **[DONE]** | `services/telemetry/src/` — heartbeats, health API, events to gateway. |
| **Frontend** | **Mostly [DONE]** | Remove remaining **Execution** plan `refetchInterval` in `services/dashboard-web/src/pages/Execution.tsx`. |
| **Robot** | **[DONE]** (for push path) | Heartbeats to telemetry; no change required for “stop polling” story. |

After the Execution plan-interval removal, the only repeating client work should be **WebSocket keepalive pings** and the robot’s **heartbeat cadence** (push, not UI polling the fleet for fresh task/plan/health state on a timer).
