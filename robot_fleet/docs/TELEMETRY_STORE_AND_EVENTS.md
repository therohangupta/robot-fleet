# Telemetry Store and Event-Driven Updates to the Gateway

This doc clarifies: (1) the flow where robots publish to the **Telemetry service** first and the Gateway gets **event-driven** updates, (2) why the Telemetry service owns the store, and (3) common patterns for separating telemetry storage (containers, stores) so data survives restarts and scales later.

---

## 1. Flow: Robot → Telemetry → Gateway → UI

- **Robots** publish heartbeats (and later joints, video, etc.) to the **Telemetry service** only. They do not post directly to the Gateway for ingest.
- **Telemetry service** receives ingest, writes to **its store(s)**, and then sends an **event-driven update** to the Gateway (e.g. “health changed”).
- **Gateway** receives that event (e.g. `POST /internal/events`), invalidates relevant client state, and pushes over WebSocket to the frontend. The Gateway does **not** poll the Telemetry service for health; it reacts to pushes.
- **Gateway** can keep a small **fallback cache** (e.g. last-known health per robot) so if the Telemetry service is temporarily down, the UI still shows something reasonable instead of blank.

So:

- **Source of truth for “robot online” and telemetry data** = Telemetry service + its stores.
- **Gateway** = consumer of events from Telemetry (+ optional fallback cache), and BFF for the frontend (control via Fleet, display via Telemetry read API or cached data).

---

## 2. Telemetry Service Owns the Stores

The Telemetry service **hosts** (or fronts) all telemetry storage:

- **Now (minimal):** A small cache per robot, e.g. last 10–20 heartbeats in memory or in a dedicated store. Enough to derive “last seen” and “reachable” and to push “health changed” to the Gateway.
- **Later:** Larger-scale stores for time-series (joint positions, metrics) and blobs (video, images). Same idea: Telemetry service is the only writer; it exposes read APIs and pushes events when relevant.

The Gateway does **not** own the canonical heartbeat store. It either:
- Gets **events** from the Telemetry service and optionally caches them for resilience, or
- **Queries** the Telemetry service’s read API when it needs to (e.g. on demand or after an event).

---

## 3. Separating Telemetry Storage (Containers and Stores)

Like the Fleet server and Postgres are separate (Fleet = process, Postgres = durable store), Telemetry can be split into **Telemetry process** and **store(s)**. Common patterns:

### Pattern A: In-process / same container (minimal)

- **Store:** In-memory structure (e.g. last N heartbeats per robot, or a small dict).
- **Pros:** No extra containers; simple for dev and minimal heartbeat-only use.
- **Cons:** Data lost on restart; not suitable for high volume or multi-instance Telemetry.

Good for: first version of the Telemetry service with heartbeat-only and “last 10–20 heartbeats” as you described.

---

### Pattern B: Telemetry + Redis (separate container)

- **Store:** Redis. Telemetry service (its own container) reads/writes Redis (e.g. last N heartbeats per robot, or “current health” key per robot). Redis can persist (RDB/AOF) so data survives Telemetry restarts.
- **Compose:** `telemetry`, `redis` (and optionally `db` for fleet; telemetry doesn’t need Postgres for heartbeat).
- **Pros:** Durable, fast, supports TTL and “last N” patterns; same pattern scales to rate limiting, presence, etc.
- **Cons:** One more container; Redis is not a time-series DB (fine for last-seen and small windows).

Good for: heartbeat + “last 10–20 heartbeats” with persistence and a clear separation between Telemetry process and store.

---

### Pattern C: Telemetry + time-series DB (for joints, metrics)

- **Store:** TimescaleDB, InfluxDB, or similar. Telemetry service writes time-series (joint positions, sensor streams); read API queries “last N samples” or “range” for display.
- **Compose:** `telemetry`, `timescaledb` (or `influxdb`).
- **Pros:** Right tool for high-volume, queryable time-series; retention and downsampling built in.
- **Cons:** Heavier than Redis; only needed once you have real joint/sensor streams.

Good for: later phase when you add joint positions and metrics.

---

### Pattern D: Telemetry + object storage (for video, images)

- **Store:** S3, MinIO, or similar. Telemetry service writes blobs; read API returns URLs (or signed URLs) for the Gateway/frontend.
- **Compose:** `telemetry`, `minio` (or external S3).
- **Pros:** Standard for video/images; scales and survives restarts.
- **Cons:** Only needed when you add video/image ingest.

Good for: later phase when you add execution video/images.

---

### Pattern E: Hybrid (what you grow into)

- **Heartbeat / “current health”:** Redis (or in-memory for v1). Telemetry writes here and pushes “health changed” to the Gateway.
- **Time-series (joints, metrics):** TimescaleDB or InfluxDB. Telemetry writes streams; read API queries for UI.
- **Blobs (video, images):** MinIO or S3. Telemetry writes; read API returns artifact URLs.

All stores are **separate containers** (or managed services); the **Telemetry service** is the only component that writes to them and exposes read APIs. The Gateway never talks to Redis/TimescaleDB/MinIO directly; it talks only to the Telemetry service (and receives events from it).

---

## 4. Event-Driven Update: Telemetry → Gateway

So that the UI updates without polling:

1. **Telemetry** receives heartbeat (or other relevant ingest), updates its store, then **POSTs to the Gateway** (e.g. `POST /internal/events` with payload `{ "type": "telemetry.health_changed", "robot_ids": ["r1", "r2"] }` or similar).
2. **Gateway** treats this like fleet events: invalidates “robot health” (and any related) client state and pushes a message over WebSocket (e.g. `robot-health` channel).
3. **Frontend** receives the push and refetches or updates UI (e.g. “online” from heartbeats).

The Gateway’s existing event pipeline (used for fleet mutations) can be extended to accept events from the Telemetry service; no need for the Gateway to poll Telemetry.

---

## 5. Summary

| Question | Answer |
|----------|--------|
| Where do robots send heartbeats? | **Telemetry service** only. Not the Gateway (for ingest). |
| Who owns the heartbeat store? | **Telemetry service** (and its store: in-memory, Redis, etc.). Gateway can keep a small fallback cache. |
| How does the UI get “online” without polling? | **Telemetry** pushes an event to the Gateway after updating the store; Gateway pushes to the frontend over WebSocket. |
| Separate store/container for telemetry? | **Yes**, same idea as Fleet + Postgres. Start with in-process or Redis; add time-series and object-store containers when you add joints and video. |
| Common patterns? | In-process (v1) → Redis (durable heartbeat/cache) → TimescaleDB/Influx (time-series) → S3/MinIO (blobs). Telemetry service is the single writer and read-API owner; Gateway only consumes events and optionally caches. |

Next implementation steps that fit this design:

1. **Minimal Telemetry service:** One container, in-memory store (e.g. last 10–20 heartbeats per robot), `POST /ingest/heartbeat`, `GET /health/summary`, and `POST` to Gateway `/internal/events` on health change.
2. **Gateway:** Extend `/internal/events` to accept events from Telemetry; push “robot-health” (or similar) to WebSocket clients; optionally keep a small in-memory fallback cache.
3. **Later:** Add Redis (or similar) as a named volume or container for Telemetry so heartbeat data survives restarts; then add time-series and object-store containers when you add joints and video.
