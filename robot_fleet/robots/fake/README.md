# Fake robots (v2)

Simulated robot servers for testing. Each robot runs a FastAPI server with `/do_task` and `/health`, and **pushes heartbeats** to the Telemetry service; the gateway reads health from Telemetry so the UI shows robots as online/offline without polling each robot.

## No-rebuild dev workflow

1. **Build images once** (or when Dockerfile / `requirements.txt` change), from repo root:
   ```bash
   ./robot_fleet/scripts/fake_robots/rebuild_examples_docker.sh
   ```

2. **Run with bind mount** so code changes are visible in the container:
   ```bash
   ./robot_fleet/scripts/fake_robots/run_examples_docker_dev.sh
   ```
   The repo is mounted at `/app`; edits to `server.py` (or any file) are picked up **without rebuilding**. Restart the container to reload Python: `docker restart <container_name>`.

3. **Telemetry URL** (for heartbeat push): Robots POST to `TELEMETRY_URL/ingest/heartbeat` every 15s (override with `HEARTBEAT_INTERVAL`). The gateway reads health from the Telemetry service, so heartbeats must go to Telemetry. Default from inside Docker is `http://host.docker.internal:9000`. Override when needed:
   ```bash
   TELEMETRY_URL=http://host.docker.internal:9000 ./robot_fleet/scripts/fake_robots/run_examples_docker_dev.sh
   ```

## Running without bind mount

Use `run_examples_docker.sh` after building. Any code change requires rebuilding the image.

## Populating demo data

After starting the robots, register them and set up world state/goals:

```bash
./robot_fleet/scripts/examples/populate_fake.sh
```

## Robot ports

- moma (first): 8001  
- nav: 8002  
- pick_place: 8003  
- moma (second, robot_id=moma_2): 8004  
