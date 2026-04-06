# TODO

Items to address later. Moving on to other features for now.

---

## Telemetry ingest: store/publish edge cases

**Context:** Telemetry ingest uses latest-wins coalescing and a pluggable publisher. If `store.record_heartbeat()` raises after we've popped from the pending map, that heartbeat is dropped (no retry).

**When would the store raise?**
- **Current in-memory store:** Effectively never in normal operation. The only realistic case is a **malformed payload** (e.g. `ts` or other fields with wrong types that break the Heartbeat dataclass). If that happens, the fix belongs on the **robot server side** (send valid payloads), not in the telemetry service's heartbeat store.
- **Future store (Redis, DB, etc.):** Network errors, timeouts, serialization errors could raise. Then consider: re-insert `to_apply` into `_pending` on exception (with a retry limit), or another recovery strategy.

**Deferred:** Address when swapping the store for a remote/persistent backend or if malformed payloads become an issue in practice.

---

## Telemetry service: future integrations (orchestration layer)

The telemetry service is designed as a thin orchestration layer over off-the-shelf integrations, similar to the fleet server using Postgres instead of a custom DB.

**Possible integrations to add later (no need now):**

- **Redis (or other remote store)** for heartbeat/health state when:
  - Running multiple telemetry instances that must share the same view, or
  - You want persistence across restarts, or
  - Scale makes in-memory pressure a concern.
  - Today: in-memory store is fine for hundreds/thousands of robots (last N heartbeats per robot).

- **Kafka (or Redis Streams, NATS, etc.)** for fan-out when:
  - Multiple consumers need the same heartbeat/health_changed stream (gateway, analytics, ML, alerting), or
  - You want replay/backfill.
  - Today: pluggable publisher is in place; default is HTTP to gateway. Add a `KafkaHealthChangedPublisher` (or composite) in `publishing.py` and wire in `app.py` when needed.

- **Time-series DB** (for joint positions, sensor streams, numeric telemetry over time):
  - **InfluxDB** – popular, good for metrics/sensors, has a clear time-series model and retention; open-source and cloud.
  - **TimescaleDB** – Postgres extension; SQL + time-series (hypertables, compression). Fits if you already use Postgres.
  - **QuestDB** – fast ingestion, SQL; good for high-frequency robot/sensor data.
  - **Prometheus** – if you only need metrics (counters/gauges) and PromQL, not full arbitrary series; often used for infra, can be used for robot metrics.
  - Use when: you want to query “joint positions for robot X in the last 5 minutes” or “sensor Y over time” with retention and aggregation.

- **Object storage** (for video, images, large blobs; telemetry stores metadata/URLs, not the bytes):
  - **MinIO** – S3-compatible, self-hosted, open-source; easy to run in Docker.
  - **AWS S3 / GCS / Azure Blob** – managed, scalable; use when you’re already in the cloud.
  - **Local filesystem** – directory per robot or per session; simplest for dev/single-node.
  - Use when: robots push video clips or images; telemetry records `robot_id`, `ts`, `url` or `path`, and optionally a small thumbnail or metadata.

- **Video / streams** (if you need more than “store a file and link it”):
  - **Object storage + metadata in Postgres/Redis** – store files in MinIO/S3, store (robot_id, ts, object_key, duration, etc.) in DB; telemetry API returns URLs or keys.
  - **Redis Streams or Kafka** – stream frame metadata or small thumbnails; heavier, for real-time pipelines or multiple consumers.
  - **Dedicated video (e.g. Livepeer, Wowza, or just HLS/S3)** – only if you need live streaming or heavy transcoding; otherwise “upload clip → object storage → link in telemetry” is enough.

- **Joint positions (and similar numeric telemetry)**:
  - Same as time-series: use **InfluxDB**, **TimescaleDB**, or **QuestDB**; robot sends `{ robot_id, ts, joints: [ ... ] }` or similar; telemetry writes to the TSDB and optionally notifies (e.g. “new joint snapshot” event) for UI or other consumers.
  - Lightweight option: **Redis** with key `robot:{id}:joints` and TTL, or append to a list and trim; good for “latest only” or last N samples per robot without full TSDB.

Same idea everywhere: telemetry orchestrates (ingest API, validation, maybe publish events); the integration does storage and query.

**Free / open-source options (self-hosted, no vendor lock-in):**
- **Redis** – BSD; free. Redis Streams, Kafka (Apache), NATS – all have free OSS versions.
- **Time-series:** InfluxDB OSS, TimescaleDB, QuestDB, Prometheus – all have free open-source editions. Cloud/managed versions are paid.
- **Object storage:** MinIO (AGPL) – free. Local filesystem – free. AWS S3 / GCS / Azure have free tiers (limited GB and requests per month); beyond that, paid.
- **Video:** MinIO + Postgres/Redis for metadata = free (self-hosted). Kafka/Redis Streams = free OSS. Dedicated video (Wowza, etc.) usually has a free tier or trial, then paid.
