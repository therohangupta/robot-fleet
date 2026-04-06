"""
Fake Pick-place robot server (v2). Use bind-mount + restart to pick up code changes without rebuild.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

from packages.robot_sdk.src.models import TaskRequest, TaskResult
from packages.robot_sdk.src.server.server_base import RobotServerBase
from packages.robot_sdk.src.schema.yaml_validator import YAMLValidator
from packages.config import TELEMETRY_URL as _TELEMETRY_URL_DEFAULT, ROBOT_HEARTBEAT_INTERVAL_SECS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TELEMETRY_URL = os.environ.get("TELEMETRY_URL", _TELEMETRY_URL_DEFAULT).rstrip("/")
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL", str(ROBOT_HEARTBEAT_INTERVAL_SECS)))
TASK_DURATION = float(os.environ.get("TASK_DURATION", "8"))
_busy = False
_heartbeat_task: asyncio.Task | None = None


class FakeRobotServer(RobotServerBase):
    def __init__(self, robot_id: str, port: int):
        super().__init__(robot_id, port)
        logger.info("FakeRobotServer '%s' initialized on port %s.", robot_id, port)

    async def _execute_task(self, task_request: TaskRequest) -> TaskResult:
        global _busy
        logger.info("Received task: %s", task_request.task_description)
        desc = task_request.task_description
        if "DO THE FOLLOWING TASK:" in desc:
            desc = desc.split("DO THE FOLLOWING TASK:")[1].strip()
        else:
            desc = desc.strip()
        _busy = True
        try:
            await asyncio.sleep(TASK_DURATION)
            return TaskResult(
                success=True,
                message=f"Succeeded task!\nTask Given by Planner: '{desc}'\nTask Result Status by Robot: 'Completed: {desc}'",
                replan=False,
            )
        finally:
            _busy = False


_instance: FakeRobotServer | None = None


async def _send_heartbeat(host: Optional[str], port: Optional[int]):
    """Send heartbeat; identity is host:port (no robot_id)."""
    payload: dict = {"reachable": True, "busy": _busy}
    if host is not None and port is not None:
        payload["host"] = host
        payload["port"] = port
    while True:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{TELEMETRY_URL}/ingest/heartbeat",
                    json=payload,
                )
        except Exception as e:
            logger.debug("Heartbeat failed: %s", e)
        await asyncio.sleep(HEARTBEAT_INTERVAL)


# Set in main() from YAML taskServer + TASK_SERVER_HOST/TASK_SERVER_PORT env overrides (single source of truth)
_task_server_host: Optional[str] = None
_task_server_port: Optional[int] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _heartbeat_task
    if _instance is not None:
        _heartbeat_task = asyncio.create_task(_send_heartbeat(_task_server_host, _task_server_port))
        logger.info("Heartbeat task started -> %s/ingest/heartbeat every %ss", TELEMETRY_URL, HEARTBEAT_INTERVAL)
    yield
    if _heartbeat_task:
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Self-identity is host:port (same as heartbeat); fleet robot_id lives only in the fleet."""
    return {"status": "healthy", "robot_id": _instance.robot_id if _instance else None}


@app.post("/do_task", response_model=TaskResult)
async def do_task(request: TaskRequest):
    if _instance is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    try:
        return await _instance._execute_task(request)
    except Exception as e:
        logger.exception("Error processing task")
        raise HTTPException(status_code=500, detail=str(e)) from e


def main():
    global _instance, _task_server_host, _task_server_port
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to robot YAML config")
    args = parser.parse_args()
    validator = YAMLValidator()
    config = validator.validate_file(args.config)
    port = int(config["taskServer"]["port"])
    # Single source: YAML taskServer, overridable by env. Self-identity = host:port (no fleet robot_id).
    _task_server_host = os.environ.get("TASK_SERVER_HOST") or config["taskServer"].get("host", "localhost")
    _task_server_port = int(os.environ.get("TASK_SERVER_PORT", str(port)))
    _instance = FakeRobotServer(robot_id=f"{_task_server_host}:{_task_server_port}", port=port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
