# Telemetry Store and Event-Driven Updates to the Gateway

**Status**

- **Implemented:** Robots POST heartbeats to the **Telemetry service**; Telemetry owns the **in-memory** heartbeat store; on health transitions Telemetry **POSTs** `telemetry.health_changed` to the Gateway’s **`POST /internal/events`**; the Gateway **maps** that to WebSocket invalidation (`robot-health`). The Gateway **reads** current health from Telemetry via **`GET {TELEMETRY_URL}/health/summary`** (and related) when API handlers run — **on-demand HTTP**, not a separate background poller.
- **Future / design only:** Redis, time-series DB, object storage as backing stores; additional ingest types (joints, video).

This doc clarifies: (1) the flow **robot → Telemetry → Gateway → UI**, (2) why Telemetry owns the store, and (3) patterns for separating telemetry storage so data survives restarts and scales later.

---

## 1. Flow: Robot → Telemetry → Gateway → UI

- **Robots** publish heartbeats to the **Telemetry service** (`POST {TELEMETRY_URL}/ingest/heartbeat`). They should **not** rely on the Gateway’s legacy **`POST /api/telemetry/heartbeat`** when Telemetry is deployed (see `TELEMETRY_AND_DATA_FLOW.md`).
- **Telemetry service** receives ingest, updates **`heartbeat_store.py`** (last **N** heartbeats per key, ring buffer), then — when **effective** reachability changes — sends an update to the Gateway via **`POST /internal/events`** with **`type: "telemetry.health_changed"`** and **`robot_ids`** (keys are **`host:port`** strings). A **timeout scanner** in `app.py` also publishes when heartbeats stop and reachability flips to false.
- **Gateway** receives that event on **`routers/websocket.py`** → **`receive_fleet_event`**, derives React Query keys (`telemetry.health_changed` → **`["robot-health"]`**), and notifies WebSocket clients so the UI refetches.
- **Gateway** does **not** run a loop that polls Telemetry only to simulate real time; **push** path = Telemetry → Gateway events → WS. When the UI (or any client) calls Gateway robot health endpoints, the Gateway **fetches fresh state** from Telemetry with **`services/gateway/src/services/telemetry_client.py`** (`GET /health/summary`, etc.) and merges with the Fleet registry (**`routers/robots.py`**, join on **`host:port`**).

So:

- **Source of truth for “robot online” (heartbeat-derived)** = Telemetry service + its store.
- **Gateway** = consumer of events from Telemetry + **BFF** that queries Telemetry read APIs on demand; optional small caches could be added later for resilience (not required for the current design).

---

## 2. Telemetry Service Owns the Stores

The Telemetry service **hosts** telemetry storage for the heartbeat path:

- **Implemented (minimal):** In-memory store — last **N** heartbeats per robot key (`MAX_HEARTBEATS_PER_ROBOT` in `packages/config.py`), effective reachability using **`HEARTBEAT_REACHABLE_THRESHOLD_SECS`**, and **`check_timeouts()`** for stale robots. Enough to derive “last seen”, “reachable”, “busy”, and to push **`telemetry.health_changed`**.
- **Later:** Larger-scale stores for time-series (joint positions, metrics) and blobs (video, images). Same idea: Telemetry service remains the writer; it exposes read APIs and pushes events when relevant.

The Gateway does **not** own the canonical heartbeat store for production. It:

- Receives **events** from Telemetry and broadcasts invalidations to WebSocket clients, and
- **Queries** Telemetry’s read API when serving **`GET /api/robots/health/all`** and related endpoints.

---

## 3. Separating Telemetry Storage (Containers and Stores)

Like the Fleet server and Postgres are separate (Fleet = process, Postgres = durable store), Telemetry can be split into **Telemetry process** and **store(s)**. Common patterns:

### Pattern A: In-process / same container (minimal) — **current implementation**

- **Store:** In-memory structure in **`heartbeat_store.py`** (ring buffer per robot key).
- **Pros:** No extra containers; simple for dev and heartbeat-only use.
- **Cons:** Data lost on Telemetry restart; not suitable for high volume or multi-instance Telemetry without external coordination.

This matches the **shipped** Telemetry service today.

---

### Pattern B: Telemetry + Redis (separate container) — **future**

- **Store:** Redis. Telemetry service reads/writes Redis (e.g. last N heartbeats per robot, or “current health” key per robot). Redis can persist (RDB/AOF) so data survives Telemetry restarts.
- **Compose:** `telemetry`, `redis` (and optionally `db` for fleet; telemetry doesn’t need Postgres for heartbeat).
- **Pros:** Durable, fast, supports TTL and “last N” patterns; same pattern scales to rate limiting, presence, etc.
- **Cons:** One more container; Redis is not a time-series DB (fine for last-seen and small windows).

---

### Pattern C: Telemetry + time-series DB (for joints, metrics) — **future**

- **Store:** TimescaleDB, InfluxDB, or similar. Telemetry service writes time-series (joint positions, sensor streams); read API queries “last N samples” or “range” for display.
- **Compose:** `telemetry`, `timescaledb` (or `influxdb`).
- **Pros:** Right tool for high-volume, queryable time-series; retention and downsampling built in.
- **Cons:** Heavier than Redis; only needed once you have real joint/sensor streams.

---

### Pattern D: Telemetry + object storage (for video, images) — **future**

- **Store:** S3, MinIO, or similar. Telemetry service writes blobs; read API returns URLs (or signed URLs) for the Gateway/frontend.
- **Compose:** `telemetry`, `minio` (or external S3).
- **Pros:** Standard for video/images; scales and survives restarts.
- **Cons:** Only needed when you add video/image ingest.

---

### Pattern E: Hybrid (what you grow into) — **future**

- **Heartbeat / “current health”:** Redis (or in-memory for v1). Telemetry writes here and pushes “health changed” to the Gateway.
- **Time-series (joints, metrics):** TimescaleDB or InfluxDB. Telemetry writes streams; read API queries for UI.
- **Blobs (video, images):** MinIO or S3. Telemetry writes; read API returns artifact URLs.

All stores are **separate containers** (or managed services); the **Telemetry service** is the only component that should write to them and expose read APIs. The Gateway should talk to Telemetry (and receive events), not to Redis/TimescaleDB/MinIO directly.

---

## 4. Event-Driven Update: Telemetry → Gateway — **implemented**

So the UI can refresh without polling Telemetry on a timer:

1. **Telemetry** receives a heartbeat (or the timeout scanner fires), updates the store, and — when effective health changes — **`POST`s to `GATEWAY_EVENT_URL`** (same path Fleet uses): **`POST /internal/events`** with JSON matching **`HealthChangedEvent`** in `events.py` (e.g. **`type`**: **`telemetry.health_changed`**, **`robot_ids`**: list of **`host:port`** keys). Implemented in **`publishing.py`**.
2. **Gateway** (`routers/websocket.py`) maps **`telemetry.health_changed`** → **`robot-health`** invalidation and pushes to WebSocket subscribers on **`/ws/global-updates`**.
3. **Frontend** receives the push and refetches health (e.g. React Query **`robot-health`**).

The Gateway’s **`/internal/events`** pipeline is shared with Fleet mutations; Telemetry reuses it — no duplicate polling layer is required for “something changed”.

---

## 5. Summary

| Question | Answer |
|----------|--------|
| Where do robots send heartbeats? | **`POST {TELEMETRY_URL}/ingest/heartbeat`** on the **Telemetry service**. Prefer this over the Gateway’s legacy **`/api/telemetry/heartbeat`**. |
| Who owns the heartbeat store? | **Telemetry** (`heartbeat_store.py`). Gateway queries read APIs and handles events. |
| How does the UI get timely “online” updates? | **Telemetry** pushes **`telemetry.health_changed`** → Gateway **`/internal/events`** → WebSocket **`robot-health`** invalidation; UI refetches. Gateway reads Telemetry **on demand** when those API calls run. |
| Separate store/container for telemetry? | **Yes** for production hardening: same idea as Fleet + Postgres. **Today:** in-process only. **Next:** Redis (or similar) for durability; then time-series and object-store containers when you add joints and video. |
| Common patterns? | **Implemented:** Pattern A (in-memory). **Future:** B → C → D, or hybrid E. |

**Future enhancements (not a blocker for current behavior)**

- Swap **`HeartbeatStore`** backing from memory to Redis while keeping ingest and read API contracts.
- Add stream ingest and pluggable storage adapters for training-scale data.
- Optional Gateway-side short-TTL cache if you want UI resilience when Telemetry is briefly unavailable.
