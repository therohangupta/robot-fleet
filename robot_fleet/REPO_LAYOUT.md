# REPO_LAYOUT.md

### Goals of this layout
- **Directory = deployable component OR shared package**, not “dashboard-centric”.
- **UI talks only to Gateway** (single API boundary).
- **Fleet Server owns truth + execution** (Postgres, task/plan state machine).
- **Robots are packaged per-robot**: YAML + Dockerfile + server + utilities live together.
- Easy to grow into production: events, telemetry, media, scaling.

---

## Top-level tree (Option A: services + packages)

```text
robot_fleet/
  README.md
  REPO_LAYOUT.md
  DESIGN.md                      # architecture decisions (copy/adapt from current root DESIGN.md)
  docs/
    architecture.md
    events.md
    api-contracts.md
    ops.md

  services/
    fleet-server/
      README.md
      Dockerfile
      pyproject.toml             # optional: if you want each service independently packaged
      src/
        fleet_server/
          cli.py                 # entrypoint: `fleet-server ...`
          app.py                 # service wiring (gRPC server startup, config)
          config.py
          logging.py
          server/
            grpc_service.py      # your current robot_fleet/server/service.py responsibilities
          execution/
            plan_executor.py     # your current robot_fleet/server/executor/executor.py responsibilities
            events.py            # emits ExecutionEvents on state changes
          planners/
            base.py
            types/
              dag/
                README.md
                summary.yaml
                prompts/
                  system.txt
                  user.txt
                planner.py
              big_dag/
                README.md
                summary.yaml
                prompts/
                  system.txt
                  user.txt
                planner.py
              monolithic/
                README.md
                summary.yaml
                prompts/
                  system.txt
                  user.txt
                planner.py
              replanner/
                README.md
                summary.yaml
                prompts/
                  system.txt
                  user.txt
                planner.py
          allocators/
            base.py
            types/
              lp/
                README.md
                summary.yaml
                prompts/
                  system.txt
                  user.txt
                allocator.py
              llm/
                README.md
                summary.yaml
                prompts/
                  system.txt
                  user.txt
                allocator.py
              cost_based/
                README.md
                summary.yaml
                prompts/
                  system.txt
                  user.txt
                allocator.py
          robot_control/
            robot_task_client.py # command/control client to robot task servers (RPC/HTTP)
          storage/
            registry.py          # DB facade (wraps SQLAlchemy models)
          proto/
            fleet_manager_pb2.py
            fleet_manager_pb2_grpc.py

    gateway/
      README.md
      Dockerfile
      pyproject.toml
      src/
        gateway/
          cli.py                 # entrypoint: `gateway ...`
          app.py                 # FastAPI app factory
          config.py
          logging.py
          auth/
            dependencies.py
            models.py
          clients/
            fleet_client.py      # talks to fleet-server (gRPC/HTTP)
          api/                   # client-facing REST API (stable contract)
            routers/
              robots.py
              plans.py
              tasks.py
              goals.py
              strategies.py
              methods.py
              world.py
            models/
              requests.py
              responses.py
          realtime/
            ws.py                # websocket transport only
            subscriptions.py     # topics, filters, per-plan streams
          events/
            schema.py            # event types (TaskStateChanged, RobotHeartbeat, ...)
            broker.py            # in-proc broker now; swap to NATS later
            fleet_subscription.py# subscribe to fleet-server event stream (no polling loops)
          telemetry/
            ingest.py            # robot -> gateway telemetry ingest endpoints
            normalize.py         # convert robot payloads to canonical schema
          media/
            signaling.py         # WebRTC signaling endpoints (future)
            streams.py           # stream registry + access control (future)

    dashboard-web/
      README.md
      Dockerfile                 # optional (prod build)
      package.json
      vite.config.ts
      src/
        lib/api.ts               # calls gateway only
        pages/
        components/

    cli/                         # optional: if you want an official “client” product
      README.md
      pyproject.toml
      src/
        robot_fleet_cli/
          main.py

  packages/
    proto/
      README.md
      fleet_manager.proto
      grpc_gen.sh
      generated/
        python/
          fleet_manager_pb2.py
          fleet_manager_pb2_grpc.py
        typescript/              # optional if you later generate TS clients
          ...

    fleet-core/
      README.md
      pyproject.toml
      src/
        fleet_core/
          config.py
          models.py              # SQLAlchemy models (PlanModel/TaskModel/RobotModel/GoalModel)
          registry.py            # RobotInstanceRegistry (DB operations)
          conversions.py         # model <-> proto/dict conversions
          errors.py

    robot-sdk/
      README.md
      pyproject.toml
      src/
        models.py               # TaskRequest/TaskResult data models
        client/
          robot_client.py       # RobotClient (used by fleet-server)
        server/
          server_base.py        # RobotServerBase
        schema/
          schema.yaml
          yaml_validator.py

    client-sdk/
      README.md
      pyproject.toml
      src/
        client_sdk/
          gateway_client.py      # typed REST client for gateway (python)
          models.py

  robots/
    fake/                        # “examples” class robots (simulated)
      moma/
        README.md
        moma.yaml
        Dockerfile
        server.py                # fake server implementation
        _tools.py
      nav/
        README.md
        nav.yaml
        Dockerfile
        server.py
        _tools.py
      pick_place/
        README.md
        pick_place.yaml
        Dockerfile
        server.py
        _tools.py

    real/                        # “real” robots (lab hardware + demos)
      hsr/
        README.md
        hsr.yaml
        Dockerfile
        server.py
        _tools.py
      locobot/
        README.md
        locobot.yaml
        Dockerfile
        server.py
        _tools.py

  tools/
    scripts/
      run_examples_dev.sh        # equivalent to your run_examples_docker_dev.sh
      run_examples.sh
      rebuild_examples.sh
      dev_up.sh                  # starts postgres + fleet-server + gateway + web
    docker/
      compose.yaml               # local orchestration (postgres + services + optional robots)
      env.example

```

---

## What goes where (high-level responsibilities)

### `services/fleet-server/` (Control Plane)
- Executes plans, owns DB truth, coordinates robots, publishes execution events.
- Should not contain any browser-specific logic.

### `services/gateway/` (Client-facing edge / BFF)
- One stable API for *any* client (web/mobile/CLI).
- Hosts realtime fanout (WS/SSE) and ingests telemetry/media *as needed*.
- Auth/permissions live here (or a dedicated auth service later).

### `services/dashboard-web/` (Browser UI)
- Pure client; calls gateway only.

### `packages/*` (Shared libraries)
- Shared schema/proto/registry/sdk code used by services and robots.
- Prevents circular “import from dashboard into robot_fleet” patterns.

### `robots/<fake|real>/<robot_name>/` (Robot bundles)
Each robot directory contains **everything needed for that robot**:
- YAML config
- Dockerfile
- server implementation (`server.py`)
- helper files (`_tools.py`, assets, calibration, etc.)

This directly matches your “don’t make me search 3 dirs for one robot” requirement.

---

## How your current repo maps into this layout (examples)

### Fleet server
- `robot_fleet/server/service.py` → `services/fleet-server/src/fleet_server/server/grpc_service.py`
- `robot_fleet/server/executor/executor.py` → `services/fleet-server/src/fleet_server/execution/plan_executor.py`
- `robot_fleet/server/__main__.py` → `services/fleet-server/src/fleet_server/cli.py`

### Gateway
- `dashboard/backend/app.py` → `services/gateway/src/gateway/app.py`
- `dashboard/backend/grpc_bridge.py` → `services/gateway/src/gateway/clients/fleet_client.py`
- `dashboard/backend/routers/*` → `services/gateway/src/gateway/api/routers/*`
- `dashboard/backend/routers/websocket.py` → split into:
  - `services/gateway/src/gateway/realtime/ws.py` (transport)
  - `services/gateway/src/gateway/events/*` (event subscription/broker)

### Frontend
- `dashboard/frontend/*` → `services/dashboard-web/*`

### Robots
- `robot_fleet/robots/examples/moma/*` + `fake_robot_server.py` usage → `robots/fake/moma/*`
- `robot_fleet/robots/demo/*` → `robots/real/*` (or `robots/real/demo/*` if you prefer)

### Shared packages
- `robot_fleet/proto/*` → `packages/proto/*`
- `robot_fleet/robots/registry/*` → `packages/fleet-core/*`
- `robot_fleet/robots/client/robot_client.py` + server base/schema → `packages/robot-sdk/*`

---

## End-to-end flows (what calls what)

This section answers: “If the UI does X, which components/files run?”

### Conventions used below
- **Client UI**: `services/dashboard-web/src/...` (or any other client later)
- **Gateway/BFF API** (client-facing REST): `services/gateway/src/gateway/api/routers/*.py`
- **Gateway ⇄ Fleet client**: `services/gateway/src/gateway/clients/fleet_client.py`
- **Fleet gRPC service**: `services/fleet-server/src/fleet_server/server/grpc_service.py`
- **Fleet execution loop**: `services/fleet-server/src/fleet_server/execution/plan_executor.py`
- **Fleet planners/allocators**:
  - `services/fleet-server/src/fleet_server/planners/types/<name>/planner.py`
  - `services/fleet-server/src/fleet_server/allocators/types/<name>/allocator.py`
- **DB (truth) access**: `packages/fleet-core/src/fleet_core/registry.py`

### A) Robot registration + removal

#### Register robot
1) UI → Gateway:
   - `services/dashboard-web/src/lib/api.ts` calls `POST /api/robots/register`
2) Gateway router:
   - `services/gateway/src/gateway/api/routers/robots.py` validates request and forwards to fleet
3) Gateway → Fleet:
   - `services/gateway/src/gateway/clients/fleet_client.py` calls fleet `RegisterRobot` (gRPC)
4) Fleet server:
   - `services/fleet-server/src/fleet_server/server/grpc_service.py` validates and stores robot in DB via registry
5) DB write:
   - `packages/fleet-core/src/fleet_core/registry.py` persists robot + task server address
6) Observe updates:
   - Fleet emits `robot.state_changed` / `robot.registered` event → Gateway → UI realtime.

#### Unregister robot
Same call chain shape, via:
- UI calls `DELETE /api/robots/{robot_id}`
- Gateway `robots.py` → Fleet `UnregisterRobot`
- Fleet updates DB; emits `robot.unregistered` event.

---

### B) Create a plan (auto-planning + optional auto-allocation)

1) UI → Gateway:
   - `services/dashboard-web/src/lib/api.ts` calls `POST /api/plans`
2) Gateway router:
   - `services/gateway/src/gateway/api/routers/plans.py` forwards request to fleet
3) Gateway → Fleet:
   - `services/gateway/src/gateway/clients/fleet_client.py` calls `CreatePlan` (gRPC)
4) Fleet server (control-plane implementation):
   - `services/fleet-server/src/fleet_server/server/grpc_service.py`:
     - selects planner: `fleet_server/planners/base.py` → `types/<planner>/planner.py`
     - planner generates DAG + saves tasks/plan to DB via `fleet_core.registry`
     - if allocation strategy != NONE:
       - selects allocator: `fleet_server/allocators/base.py` → `types/<allocator>/allocator.py`
       - allocator assigns robots to tasks + stores allocation artifacts/prompts
5) DB write:
   - `packages/fleet-core/src/fleet_core/registry.py` creates plan + tasks and links them
6) Observe updates:
   - Fleet emits `plan.created`, `task.created*`, and (if allocated) `plan.allocated` events → Gateway → UI.

---

### C) Allocate an existing plan

1) UI → Gateway:
   - `POST /api/plans/{plan_id}/allocate`
2) Gateway:
   - `services/gateway/src/gateway/api/routers/plans.py` → fleet client
3) Fleet:
   - `grpc_service.py` runs allocator from `allocators/types/<name>/allocator.py`
4) DB:
   - `fleet_core.registry.update_plan(...)` stores allocation strategy + artifacts/prompts
   - task assignments stored via task updates (robot_id/robot_type)
5) Observe:
   - Fleet emits `plan.allocated`, `task.assigned*` events.

---

### D) Start execution / monitor execution

#### Start execution (control)
1) UI → Gateway:
   - `POST /api/plans/{plan_id}/start`
2) Gateway:
   - `services/gateway/src/gateway/api/routers/plans.py` → fleet client
3) Fleet:
   - `grpc_service.py` validates “fully allocated” then starts executor
4) Executor:
   - `services/fleet-server/src/fleet_server/execution/plan_executor.py`:
     - dispatches tasks to robots using `fleet_server/robot_control/robot_task_client.py`
     - updates task status/result in DB via `fleet_core.registry`
     - emits `task.state_changed` events (started/completed/failed)
5) Observe:
   - Gateway realtime pushes task/plan state changes to UI.

#### Monitor execution (observe)
- UI opens a realtime subscription:
  - `services/gateway/src/gateway/realtime/ws.py` (transport)
  - backed by `services/gateway/src/gateway/events/broker.py` and `fleet_subscription.py`
- UI also fetches snapshots as needed:
  - `GET /api/plans/{plan_id}` and/or `GET /api/tasks?plan_id=...`

---

### E) Modify an existing plan (tasks CRUD inside a plan)
These changes are “control-plane mutations” but are usually initiated from the UI.

#### Add a task to a plan
1) UI → Gateway: `POST /api/tasks` with `plan_id`
2) Gateway tasks router → fleet client → Fleet `CreateTask`
3) Fleet writes task to DB via `fleet_core.registry.create_task(...)`
4) Fleet recomputes/marks plan strategy:
   - planning = MANUAL_PLAN
   - allocation = MANUAL_ALLOCATION if any task has robot_id else NONE
5) Fleet emits `task.created` + `plan.strategy_changed` events

#### Edit a task (description/goal/dependencies/robot assignment)
1) UI → Gateway: `PATCH /api/tasks/{task_id}`
2) Gateway → Fleet `UpdateTask`
3) Fleet updates DB (task fields) and reconciles robot_type if robot_id changed
4) Fleet marks plan manual strategies + emits events:
   - `task.updated` / `task.state_changed` (if relevant)
   - `plan.strategy_changed`

#### Delete a task (unlink dependencies, do not delete dependents)
1) UI → Gateway: `DELETE /api/tasks/{task_id}`
2) Gateway → Fleet `DeleteTask`
3) Fleet runs “delete + unlink dependents” in DB via registry
4) Fleet marks plan manual strategies + emits:
   - `task.deleted` (includes updated_task_ids)
   - `task.updated` for each dependent that had dependency removed

---

### F) Plan metadata changes (name/description/copy)
- Update plan name/description:
  - UI → Gateway: `PUT /api/plans/{plan_id}`
  - Gateway → Fleet (or direct DB update through fleet) → DB → emits `plan.updated`
- Copy plan:
  - UI → Gateway: `POST /api/plans/{plan_id}/copy`
  - Fleet copies tasks + resets execution status → emits `plan.copied`

---

### G) Robot health + heartbeat (recommended production-shaped path)

#### Robot → fleet (control-relevant summary)
- Robots send heartbeat summary (reachable/faults/busy/last_seen)
- Fleet stores the latest heartbeat and computes **effective** status by reconciling:
  - heartbeat (observed) + DB/task state (desired)
- Fleet emits `robot.state_changed` events to Gateway.

#### Robot → gateway (high-rate telemetry/media)
- Joint states, logs, camera streams go to Gateway telemetry/media modules:
  - `services/gateway/src/gateway/telemetry/*`
  - `services/gateway/src/gateway/media/*`
- Gateway fans out to clients (WS/WebRTC) and optionally stores summaries.

---

## Why this is “production-grade”
- Clean deployable boundaries: **fleet-server** vs **gateway** vs **clients**.
- No browser → fleet direct dependency; the gateway contract can stabilize.
- Realtime becomes event-driven (no polling loops baked into the design).
- Robots are packaged as cohesive bundles.
- Shared code is centralized in packages so services don’t import across product boundaries.

