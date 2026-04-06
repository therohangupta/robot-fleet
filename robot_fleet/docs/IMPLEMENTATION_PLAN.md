# RobotFleet v2: Step-by-Step Implementation Plan

This document orders the main architectural improvements we analyzed and how to test each step. It reflects **both** the original phased intent and **current implementation status** in the repo (Fleet, **separate** Gateway and Telemetry, Dashboard, fake robots).

---

## Status overview

| Phase | Topic | Status |
|-------|--------|--------|
| **1** | Event-driven updates (remove polling) | **Largely done** — Fleet → Gateway HTTP callback, Gateway event-driven WebSockets with per-subscriber queues; some frontend timers may remain (see Phase 1). |
| **2** | Robot health from heartbeat (no per-robot `/health` polling) | **Done** — Health for the UI is derived from heartbeat data via the **Telemetry service**; Gateway reads `GET /health/summary` (and per-robot health) from Telemetry. |
| **3** | Telemetry as a separate service | **Done** — `services/telemetry/src/` on port **9000**; robots POST `TELEMETRY_URL/ingest/heartbeat`; Gateway uses `TELEMETRY_URL` for health. |
| **4** | Telemetry read API for display (joints, artifacts, …) | **Partially done** — Health read API is implemented end-to-end; **joints / video / artifact list APIs are not implemented yet** (no Telemetry routes + no Gateway proxies for those). |
| **5** | Full telemetry ingest (streams, object storage, training path) | **Not done** — Future work. |

---

## Summary of components (target vs today)

| Component | Role | Target state | Today |
|-----------|------|--------------|--------|
| **Fleet server** | Control plane: DB, execution, gRPC. | Emit events on mutations so Gateway can push to UI without polling. | **Implemented:** `events.py` + `GATEWAY_EVENT_URL` → `POST .../internal/events`. |
| **Gateway (BFF)** | Client-facing: proxy to Fleet, WebSockets. | Event-driven WS; query Telemetry for display. | **Implemented:** `POST /internal/events`, WS invalidation + execution updates; **telemetry_client** reads health from Telemetry. Legacy `POST /api/telemetry/heartbeat` still in tree for compatibility. |
| **Telemetry service** | Ingest + read API. | Heartbeat → later joints/video; storage; read API for Gateway. | **Heartbeat ingest + health read API + optional Gateway event publish** (see `services/telemetry/src/`). No joints/video ingest or read routes yet. |
| **Frontend** | Single client → Gateway only. | Rely on WS invalidation; minimal polling. | **Mostly** WS-driven; **Execution** may still use a short `refetchInterval` on the plan query while executing (see Phase 1). |
| **Robots** | Tasks + telemetry push. | Push heartbeats to Telemetry. | **Fake robots** POST to `TELEMETRY_URL/ingest/heartbeat`. |

---

## Phase 1: Event-driven updates (remove polling)

**Status: Largely done**

**Goal:** Avoid timer-driven churn. Fleet notifies Gateway on change; Gateway pushes to WebSocket clients; frontend refetches when invalidated (not on a fixed 1s loop).

### Implemented (reference)

- **Fleet:** `GATEWAY_EVENT_URL` (e.g. `http://gateway:8000/internal/events` in Compose); fire-and-forget POST after relevant mutations (`services/fleet_server/src/events.py` and call sites).
- **Gateway:** `POST /internal/events` accepts fleet (and telemetry) events; `EventBus` with **per-subscriber `asyncio.Queue`s**; `/ws/global-updates` and `/ws/execution/{plan_id}` react to events (with periodic **ping** on idle timeout, not 1s data polling). See `services/gateway/src/routers/websocket.py`.
- **Telemetry → Gateway:** Telemetry can POST health-change notifications to the same internal endpoint so the UI refreshes robot health without polling Telemetry on a timer (`services/telemetry/src/publishing.py`).

### Remaining / verify

- **Frontend:** Confirm remaining `refetchInterval` usage is intentional (e.g. `Execution.tsx` plan query while `execution_status === 'executing'`). Tighten or remove if you want **zero** polling for that view.
- **Execution WebSocket:** Today, subscribers receive task refreshes when **any** invalidation-relevant event is processed; fine-tuning per-`plan_id` filtering is optional.

**Phase 1 complete when (strict):** No production reliance on short-interval refetch for data that is already covered by WS invalidation; Gateway WS paths do not use `asyncio.sleep(1)`-style polling loops for fresh data. *(Current code meets the Gateway side; frontend may still have narrow intervals.)*

---

## Phase 2: Robot health from heartbeat only

**Status: Done** (via Telemetry as the heartbeat sink; Gateway does not poll each robot’s HTTP `/health` for the dashboard path.)

**Goal:** UI “online” state comes from heartbeat-derived data, not from the Gateway hammering each robot’s `/health`.

### As implemented

- **Telemetry** stores last-seen heartbeats and serves `GET /health/summary` and `GET /health/{robot_id}`.
- **Gateway** `GET /api/robots/health/...` uses **`telemetry_client`** to call Telemetry and shape responses for the frontend (`services/gateway/src/routers/robots.py`).

### Original steps (historical)

The steps below described migrating **from** an in-gateway heartbeat store **to** Telemetry; that migration is done. Skim them only if you are comparing to older branches.

<details>
<summary>Original Phase 2 checklist (collapsed)</summary>

### Step 2.1 — Gateway: health from heartbeat store

- **Was:** Read from in-gateway store / then from Telemetry.
- **Now:** Read from Telemetry over HTTP.

**Phase 2 done when:** Robot health in the UI is driven by heartbeat data; no scheduled HTTP calls from the Gateway to each robot’s `/health` for that UI path.

</details>

---

## Phase 3: Telemetry service (separate server) — heartbeat + read API

**Status: Done**

**Goal:** Telemetry ingest and read API in a **separate process**; Gateway **queries** Telemetry for health; robots push to Telemetry.

### As implemented

- **Telemetry app:** `services/telemetry/src/` — FastAPI, `POST /ingest/heartbeat`, `GET /health/summary`, `GET /health/{robot_id}`, `GET /healthz`.
- **Robots:** `TELEMETRY_URL` + `POST .../ingest/heartbeat` (see fake robots under `robot_fleet/robots/fake/`).
- **Gateway:** `TELEMETRY_URL` env (e.g. `http://telemetry:9000` in Compose) and `telemetry_client.py`.
- **Compose:** `telemetry` service on **9000**, `gateway` depends on `telemetry`.

**Phase 3 done when:** Telemetry runs as its own service; robots send heartbeats to it; Gateway serves health only by reading Telemetry; UI still uses Gateway only. **— Achieved.**

---

## Phase 4: Telemetry read API for display (Gateway → Telemetry → UI)

**Status: Partially done**

**Goal:** Gateway exposes “display telemetry” endpoints (e.g. last N joint samples, artifact/video metadata) by proxying Telemetry.

### Done today

- **Health summary and per-robot health** — Telemetry read API + Gateway integration (this satisfies part of “read API for display”).

### Not done yet

- **Telemetry service:** `GET /telemetry/joints?...`, `GET /telemetry/artifacts?...` (or equivalent) — **not present**; no stub routes in `services/telemetry/src/routers/` for these.
- **Gateway:** Proxies such as `GET /api/telemetry/joints` — **not present** alongside the legacy ingest router.

### Next steps (when you pick this up)

- Add minimal read endpoints on Telemetry (empty list / placeholder responses acceptable at first).
- Add Gateway routes that forward query params and responses.
- Optional: small UI surfaces that show “No data” until Phase 5 fills storage.

**Phase 4 complete when:** Any **non-health** telemetry the UI needs flows Browser → Gateway → Telemetry → Gateway → Browser, with Telemetry as the single source of truth for those reads.

---

## Phase 5: Full telemetry ingest (joints, video, storage) — later

**Status: Not done (future)**

**Goal:** Robots send joint streams and video to Telemetry; Telemetry writes to durable storage (time-series, object store); read APIs return real data for display and training/export.

### Step 5.1 — Telemetry service: ingest streams and storage

- **Implement:** Stream or batch ingest for joints/logs; chunked upload for video/images; config for TSDB / object storage.
- **Test:** Ingest sample payloads; assert storage and read-back via Phase 4 APIs.

### Step 5.2 — Robot-side senders

- **Implement:** Per-robot senders tagging `robot_id`, `task_id` / `plan_id` / `session_id`.
- **Test:** Run task, verify ingest + UI or export.

**Phase 5** can be split (joints first, then video) once Phase 4 stubs exist.

---

## Implementation order and testing summary

| Phase | Status | What to implement / verify |
|-------|--------|----------------------------|
| **1** | Largely done | Fleet POST + Gateway WS queues + frontend invalidation; trim any leftover refetch intervals you care about. |
| **2** | Done | Health from Telemetry-derived heartbeats; no robot `/health` polling for that path. |
| **3** | Done | Separate Telemetry container; `TELEMETRY_URL` on Gateway and robots. |
| **4** | Partial | Joints/artifacts read + Gateway proxy **still to build**; health path **done**. |
| **5** | Not done | Full ingest + storage + robot senders. |

Dependencies: Phase 4 (non-health reads) builds on Phase 3. Phase 5 builds on Phase 4’s API shapes and storage choices.

This remains the ordered roadmap; phases **1–3** and the **health** portion of **4** match the current codebase; **4** (joints/video/artifacts) and **5** are the main forward work.
