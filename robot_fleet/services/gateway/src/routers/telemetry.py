"""
Telemetry ingest: robot heartbeats and status pushed from robots.

Robots POST here instead of being polled. Gateway stores last_seen and can
fan out to UI / use for health without polling each robot.
"""

import time
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel


class HeartbeatPayload(BaseModel):
    robot_id: str
    reachable: bool = True
    busy: Optional[bool] = None
    ts: Optional[float] = None


# In-memory last heartbeat per robot (ts, reachable, busy).
# In production this could be Redis or fleet-server owned.
_heartbeats: dict[str, dict] = {}


def get_last_heartbeat(robot_id: str) -> Optional[dict]:
    return _heartbeats.get(robot_id)


def get_all_heartbeats() -> dict[str, dict]:
    return dict(_heartbeats)


router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.post("/heartbeat")
async def ingest_heartbeat(payload: HeartbeatPayload):
    """Accept heartbeat from a robot. Replaces polling that robot for /health."""
    ts = payload.ts if payload.ts is not None else time.time()
    _heartbeats[payload.robot_id] = {
        "ts": ts,
        "reachable": payload.reachable,
        "busy": payload.busy,
    }
    return {"ok": True, "robot_id": payload.robot_id}
