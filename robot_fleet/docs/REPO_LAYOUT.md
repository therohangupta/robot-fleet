# Repository layout

This document describes the **current** `robot_fleet/` directory layout: deployable services, shared packages, robots, and tooling. Earlier drafts proposed a different shape (`fleet-core`, nested `gateway/` packages, `tools/docker/`); that material is superseded by what is on disk now.

### Goals of this layout

- **Directory = deployable component OR shared package**, not dashboard-centric.
- **UI talks only to Gateway** (single HTTP/WebSocket API boundary for the browser).
- **Fleet Server owns orchestration and persistence** (Postgres, plan/task state, gRPC control plane).
- **Telemetry service** handles robot heartbeat ingest and health summaries; Gateway reads from it (see `docs/GATEWAY_VS_TELEMETRY_SPLIT.md`, `docs/TELEMETRY_AND_DATA_FLOW.md`).
- **Robots are grouped under `robots/`** (`fake/` vs `real/`) with YAML, servers, and helpers colocated per robot.
- Shared Python config and cross-cutting utilities live in `packages/` so services do not import across product boundaries.

---

## Top-level tree (current)

```text
robot_fleet/
├── README.md
├── __init__.py
├── pyproject.toml
├── docker-compose.yml
├── docker-compose.dev.yml
│
├── docs/                         # Architecture, runbooks, contracts
│   ├── COMPONENT_FLOWS.md
│   ├── COMMUNICATION_REVIEW.md
│   ├── DESIGN.md
│   ├── GATEWAY_VS_TELEMETRY_SPLIT.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── OPENAPI_CONTRACT.md
│   ├── REPO_LAYOUT.md            # (this file)
│   ├── REQUIRED_TO_REMOVE_POLLING.md
│   ├── RUN.md
│   ├── TELEMETRY_AND_DATA_FLOW.md
│   ├── TELEMETRY_STORE_AND_EVENTS.md
│   └── TODO.md
│
├── cli/                          # Operator CLI (robotctl)
│   ├── __init__.py
│   ├── robotctl.py
│   └── printer.py
│
├── packages/
│   ├── __init__.py
│   ├── config.py                 # Central settings (env-backed)
│   ├── metrics.py
│   ├── proto/                    # fleet_manager.proto + generated *_pb2*.py
│   ├── fleet_sdk/                # DB models, instance registry, fleet gRPC helpers
│   │   └── src/
│   │       ├── models.py
│   │       ├── instance_registry.py
│   │       └── grpc_client.py
│   ├── robot_sdk/                # Robot task client, server base, schema
│   │   └── src/
│   │       ├── models.py
│   │       ├── client/robot_client.py
│   │       ├── server/server_base.py
│   │       └── schema/           # schema.yaml, yaml_validator.py
│   └── client_sdk/               # Typed HTTP + WS clients for Gateway
│       ├── README.md
│       ├── contract/             # e.g. events.schema.json
│       ├── python/src/...
│       └── typescript/src/...
│
├── services/
│   ├── __init__.py
│   ├── cli/                      # Reserved; empty placeholder today
│   │
│   ├── fleet_server/             # gRPC orchestration + planners + allocators + executor
│   │   ├── Dockerfile
│   │   └── src/
│   │       ├── __main__.py       # Process entry
│   │       ├── service.py        # gRPC FleetManager service
│   │       ├── events.py         # Gateway notification helpers
│   │       ├── README.md
│   │       ├── executor/
│   │       │   └── executor.py
│   │       ├── planners/
│   │       │   ├── base.py
│   │       │   └── types/<dag|big_dag|monolithic|replanner>/
│   │       │       ├── planner.py
│   │       │       ├── summary.yaml
│   │       │       ├── system.prompt
│   │       │       └── user.prompt
│   │       ├── allocators/
│   │       │   ├── base.py
│   │       │   └── types/<lp|llm|cost_based>/
│   │       │       ├── allocator.py
│   │       │       ├── summary.yaml
│   │       │       └── *.prompt (as applicable)
│   │       └── formats/
│   │           └── formats.py
│   │
│   ├── gateway/                  # FastAPI: REST + WebSocket; bridges to fleet (gRPC)
│   │   ├── Dockerfile
│   │   └── src/
│   │       ├── main.py
│   │       ├── app.py
│   │       ├── config.py
│   │       ├── grpc_bridge.py    # Fleet gRPC calls from HTTP handlers
│   │       ├── dependencies.py
│   │       ├── routers/
│   │       │   ├── plans.py
│   │       │   ├── robots.py
│   │       │   ├── tasks.py
│   │       │   ├── goals.py
│   │       │   ├── strategies.py
│   │       │   ├── methods.py
│   │       │   ├── world.py
│   │       │   ├── websocket.py  # WS + POST /internal/events (fleet → gateway)
│   │       │   ├── telemetry.py  # Proxies health queries to telemetry service
│   │       │   ├── prompts.py
│   │       │   ├── embodiments.py
│   │       │   ├── metrics.py
│   │       │   └── __init__.py
│   │       ├── models/
│   │       │   ├── requests.py
│   │       │   └── responses.py
│   │       └── services/         # e.g. telemetry_client, robot_health, yaml_scanner
│   │
│   ├── telemetry/                # Heartbeat ingest + health API for Gateway
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   └── src/
│   │       ├── __main__.py
│   │       ├── main.py
│   │       ├── app.py
│   │       ├── config.py
│   │       ├── dependencies.py
│   │       ├── heartbeat_store.py
│   │       ├── publishing.py
│   │       ├── events.py
│   │       └── routers/
│   │           ├── ingest.py
│   │           └── health.py
│   │
│   └── dashboard-web/            # Vite + React dashboard
│       ├── package.json
│       ├── vite.config.ts
│       └── src/
│           ├── main.tsx
│           ├── App.tsx
│           ├── types.ts
│           ├── lib/              # api.ts, utils.ts
│           ├── pages/
│           │   ├── Plans.tsx
│           │   ├── PlanDetails.tsx
│           │   ├── Execution.tsx
│           │   ├── Robots.tsx
│           │   ├── Goals.tsx
│           │   ├── World.tsx
│           │   ├── Dashboard.tsx
│           │   ├── Planners.tsx
│           │   └── Allocators.tsx
│           ├── components/
│           └── ...
│
├── robots/
│   ├── fake/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── moma/
│   │   ├── nav/
│   │   └── pick_place/
│   └── real/
│       ├── hsr/
│       └── locobot/
│
└── scripts/
    ├── grpc_gen.sh
    ├── backup_db.sh
    ├── database_mgmt/            # e.g. backup_db.sh, restore_db.sh
    ├── examples/
    │   └── populate_fake.sh
    └── fake_robots/
        ├── rebuild_examples_docker.sh
        ├── run_examples_docker.sh
        └── run_examples_docker_dev.sh
```

**Not listed above:** generated artifacts (e.g. `__pycache__/`, `node_modules/`, `dist/`), local `db_backups/`, and other editor-specific files.

---

## What goes where (responsibilities)

### `services/fleet_server/` (control plane)

Runs the gRPC fleet manager, planners, allocators, and the execution loop. Persists plan/task state via `packages/fleet_sdk`. Dispatches work to robots using `packages/robot_sdk` (`RobotClient`). Emits change notifications toward the Gateway (see `events.py` and `GATEWAY_EVENT_URL` in `services/fleet_server/src/README.md`).

### `services/gateway/` (client-facing edge / BFF)

Single HTTP and WebSocket surface for browsers and other clients. Validates and maps REST requests to gRPC via `grpc_bridge.py`. WebSocket invalidation and `POST /internal/events` live in `routers/websocket.py`. Telemetry *queries* are routed through `routers/telemetry.py` and `services/telemetry_client.py` to the Telemetry service.

### `services/telemetry/` (robot health)

Accepts robot heartbeats (ingest), keeps a latest-wins store, and exposes health summaries for the Gateway. See `services/telemetry/README.md`.

### `services/dashboard-web/` (browser UI)

React SPA; talks to the Gateway only (relative `/api/...` in dev, proxied to Gateway). Shared API helpers in `src/lib/api.ts`; some pages also call `fetch` directly.

### `packages/*` (shared libraries)

- **`config.py`**: Environment-driven configuration shared across processes.
- **`proto/`**: Canonical protobuf + Python stubs for the fleet gRPC API.
- **`fleet_sdk/`**: SQLAlchemy models, `RobotInstanceRegistry`, and fleet-side gRPC client helpers.
- **`robot_sdk/`**: Robot-facing task protocol (client + server base + YAML schema).
- **`client_sdk/`**: Optional typed clients (Python/TypeScript) generated or maintained alongside `docs/OPENAPI_CONTRACT.md`.

### `cli/`

Command-line entry (`robotctl`) for operators. Distinct from the empty `services/cli/` placeholder.

### `robots/<fake|real>/<robot>/`

Per-robot YAML, `server.py`, Dockerfiles (where used), and helpers. Layout varies slightly by robot; goal is to keep one robot’s artifacts discoverable in one subtree.

### `scripts/`

Tooling: protobuf generation, DB backup/restore, fake-fleet population, and Docker-based fake robot runs.

---

## Layout notes (historical context)

The repository already follows the service/package split described in older design notes. In particular:

- Python modules use **`packages/fleet_sdk/`** (not a separate `fleet-core` package name on disk).
- Gateway code lives under **`services/gateway/src/`** with a flat module layout (`app.py`, `grpc_bridge.py`, `routers/`), not `src/gateway/api/...`.
- **Telemetry** is a **separate service**, not only a subdirectory inside Gateway.
- **Protobuf generation** is driven from **`scripts/grpc_gen.sh`** with outputs beside **`packages/proto/`**.

---

## End-to-end flows (what calls what)

This section answers: “If the UI does X, which components and files run?”

### Conventions

- **Dashboard UI**: `services/dashboard-web/src/pages/*.tsx` and `services/dashboard-web/src/lib/api.ts`
- **Gateway REST**: `services/gateway/src/routers/*.py`
- **Gateway ⇄ Fleet (gRPC)**: `services/gateway/src/grpc_bridge.py`
- **Fleet gRPC service**: `services/fleet_server/src/service.py`
- **Fleet process entry**: `services/fleet_server/src/__main__.py`
- **Execution loop**: `services/fleet_server/src/executor/executor.py` (uses `packages/robot_sdk/src/client/robot_client.py`)
- **Planners / allocators**:
  - `services/fleet_server/src/planners/types/<name>/planner.py`
  - `services/fleet_server/src/allocators/types/<name>/allocator.py`
- **DB access (models + registry)**: `packages/fleet_sdk/src/instance_registry.py`, `packages/fleet_sdk/src/models.py`
- **Realtime**: `services/gateway/src/routers/websocket.py` (WebSocket subscribers + `POST /internal/events` from Fleet)
- **Robot heartbeats**: `services/telemetry/src/routers/ingest.py` → `heartbeat_store.py`; Gateway reads via `services/gateway/src/routers/telemetry.py` and `services/gateway/src/services/telemetry_client.py`

### A) Robot registration + removal

#### Register robot

1. UI → Gateway: e.g. `POST /api/robots/register` (see `routers/robots.py`).
2. Gateway → Fleet: `grpc_bridge.py` invokes the matching gRPC RPC on `service.py`.
3. Fleet persists via `RobotInstanceRegistry` / models in `packages/fleet_sdk/src/`.
4. Updates propagate: Fleet notifies Gateway (`events.py` / HTTP POST to Gateway); Gateway fans out over WebSocket (`websocket.py`).

#### Unregister robot

Same shape via `DELETE` or equivalent route in `routers/robots.py` → gRPC → registry updates → events.

### B) Create a plan (planning + optional allocation)

1. UI → Gateway: `POST /api/plans` (`routers/plans.py`).
2. Gateway → Fleet: `grpc_bridge.py` → `CreatePlan` (or equivalent) handled in `service.py`.
3. Fleet selects a planner under `planners/types/<name>/`, writes tasks/plan through `fleet_sdk`, optionally runs an allocator under `allocators/types/<name>/`.
4. Observe: Fleet → Gateway internal event path → WebSocket clients.

### C) Allocate an existing plan

1. UI → Gateway: allocate endpoint in `routers/plans.py`.
2. Gateway → Fleet via `grpc_bridge.py`; allocator runs from `allocators/types/<name>/allocator.py`.
3. Registry updates persist allocation; events notify subscribers.

### D) Start execution / monitor execution

#### Start execution

1. UI → Gateway: start endpoint in `routers/plans.py`.
2. Gateway → Fleet: `grpc_bridge.py` → `service.py` starts or resumes work.
3. `executor/executor.py` drives tasks, calls `RobotClient`, updates DB through `fleet_sdk`, emits task/plan change events.

#### Monitor execution

- UI opens a WebSocket (see `websocket.py`); Gateway pushes invalidation/query keys when Fleet posts to `/internal/events`.
- UI may also poll or fetch snapshots via plan/task routes in `routers/plans.py` / `routers/tasks.py`.

### E) Modify an existing plan (tasks CRUD)

Task create/update/delete routes in `routers/tasks.py` map to Fleet gRPC and registry methods in `fleet_sdk`; `service.py` coordinates strategy flags and event emission. Shapes match the OpenAPI contract in `docs/OPENAPI_CONTRACT.md`.

### F) Plan metadata (name, description, copy)

Plan update and copy routes in `routers/plans.py` → Fleet → `fleet_sdk` persistence → events.

### G) Robot health + heartbeat (current split)

- **Robots → Telemetry**: HTTP ingest (`services/telemetry/src/routers/ingest.py`), stored in `heartbeat_store.py`, optional fan-out via `publishing.py`.
- **Dashboard / Gateway → Telemetry**: Health summary endpoints consumed through `routers/telemetry.py` (and `telemetry_client.py`).
- **Fleet** remains authoritative for registration and task state; **effective** health in the UI is the combination of telemetry signals and fleet data. For deeper detail, see `docs/TELEMETRY_AND_DATA_FLOW.md` and `docs/TELEMETRY_STORE_AND_EVENTS.md`.

---

## Why this structure scales

- **Deployable boundaries** are explicit: `fleet_server`, `gateway`, `telemetry`, and `dashboard-web` each have their own Dockerfile and runtime concerns.
- **Clients** depend on a stable Gateway contract, not on Fleet gRPC directly.
- **Realtime** is event-driven (Fleet pushes to Gateway; WS subscribers refresh), aligned with `docs/REQUIRED_TO_REMOVE_POLLING.md`.
- **Robots** stay grouped and shippable per folder under `robots/`.
- **Shared code** is centralized under `packages/` to avoid services importing each other’s internals.
