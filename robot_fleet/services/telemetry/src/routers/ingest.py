"""
Ingest endpoints for robot telemetry.

Robots POST heartbeats here. Latest-wins coalescing: multiple pending heartbeats
for the same robot are collapsed to one (the latest) before updating the store.
Fan-out is pluggable via HealthChangedPublisher (HTTP to gateway today; Kafka etc. later).
"""

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..dependencies import get_publisher
from ..events import HealthChangedEvent, HealthChangedPublisher
from ..heartbeat_store import get_heartbeat_store

router = APIRouter(prefix="/ingest", tags=["Ingest"])

# Pending heartbeats per robot_id (latest wins). Protected by _pending_lock.
_pending: dict[str, "HeartbeatPayload"] = {}
_pending_lock = asyncio.Lock()


class HeartbeatPayload(BaseModel):
    host: str  # Task server host; identity is host:port
    port: int  # Task server port; identity is host:port
    reachable: bool = True
    busy: Optional[bool] = None
    ts: Optional[float] = None


@router.post("/heartbeat")
async def ingest_heartbeat(
    payload: HeartbeatPayload,
    publisher: HealthChangedPublisher = Depends(get_publisher),
):
    """
    Accept a heartbeat from a robot. Identity is host:port only (no robot_id).
    Coalescing: multiple pending heartbeats for the same host:port are latest-wins.
    """
    store = get_heartbeat_store()
    key = f"{payload.host}:{payload.port}"

    # 1. Enqueue (latest wins per key)
    async with _pending_lock:
        _pending[key] = payload

    # 2. Pop and apply
    async with _pending_lock:
        to_apply = _pending.pop(key, None)
        if to_apply is None:
            return {"ok": True}

        changed = store.record_heartbeat(
            robot_id=key,
            reachable=to_apply.reachable,
            busy=to_apply.busy,
            ts=to_apply.ts,
            host=to_apply.host,
            port=to_apply.port,
        )
        if changed:
            await publisher.publish(
                HealthChangedEvent(
                    robot_ids=[key],
                    ts=time.time(),
                )
            )

    return {"ok": True}
