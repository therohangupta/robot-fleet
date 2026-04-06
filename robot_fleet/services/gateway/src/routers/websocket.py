"""
Event-driven WebSocket endpoints for real-time updates.

The fleet server POSTs events to /internal/events on every mutation.
Connected WebSocket clients are notified immediately — no polling.
"""

import asyncio
import logging
import time
from typing import Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event bus: lightweight in-process pub-sub with per-subscriber queues
# ---------------------------------------------------------------------------

class EventBus:
    """
    Broadcasts invalidation signals to all connected WebSocket clients.

    Each subscriber gets its own asyncio.Queue so events are never lost
    even when multiple clients wake concurrently.
    """

    def __init__(self):
        self._subscribers: Set[asyncio.Queue] = set()

    def notify(self, query_keys: List[str]) -> None:
        """Push query keys to every subscriber's queue."""
        for q in self._subscribers:
            q.put_nowait(query_keys)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)


event_bus = EventBus()


# ---------------------------------------------------------------------------
# Map fleet events → React Query keys
# ---------------------------------------------------------------------------

_EVENT_TO_QUERIES: Dict[str, List[str]] = {
    "task.state_changed": ["tasks", "plans"],
    "plan.state_changed": ["plans", "tasks"],
    "robot.changed":      ["robots", "robot-health", "robot-allocations"],
    "telemetry.health_changed": ["robot-health"],
}


def _derive_query_keys(event_type: str) -> List[str]:
    return _EVENT_TO_QUERIES.get(event_type, ["plans", "tasks", "robots"])


# ---------------------------------------------------------------------------
# POST /internal/events — called by fleet server (fire-and-forget)
# ---------------------------------------------------------------------------

class FleetEvent(BaseModel):
    type: str
    task_id: int | None = None
    plan_id: int | None = None
    robot_id: str | None = None
    robot_ids: list[str] | None = None  # for telemetry.health_changed
    status: str | None = None
    action: str | None = None


@router.post("/internal/events", tags=["Internal"])
async def receive_fleet_event(event: FleetEvent):
    """
    Accept an event from the fleet server and broadcast to WS clients.
    This replaces the old sleep-loop polling.
    """
    keys = _derive_query_keys(event.type)
    event_bus.notify(keys)
    return {"ok": True}


# ---------------------------------------------------------------------------
# WebSocket /ws/global-updates — pushes invalidation to the frontend
# ---------------------------------------------------------------------------

@router.websocket("/ws/global-updates")
async def websocket_global_updates(websocket: WebSocket):
    """
    Event-driven WebSocket.  Sends an invalidation message only when the
    fleet server has reported a mutation — zero wasted traffic at idle.
    """
    await websocket.accept()

    await websocket.send_json({
        "type": "connected",
        "message": "Real-time updates enabled (event-driven)"
    })

    queue = event_bus.subscribe()

    try:
        while True:
            try:
                keys = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            if keys == ["__shutdown__"]:
                break

            await websocket.send_json({
                "type": "invalidate",
                "queries": keys,
                "timestamp": time.time(),
            })
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket global-updates handler failed")
    finally:
        event_bus.unsubscribe(queue)


# ---------------------------------------------------------------------------
# WebSocket /ws/execution/{plan_id} — plan-specific task updates
# ---------------------------------------------------------------------------

class ExecutionConnectionManager:
    def __init__(self):
        self.active: Dict[int, List[WebSocket]] = {}

    async def connect(self, ws: WebSocket, plan_id: int):
        await ws.accept()
        self.active.setdefault(plan_id, []).append(ws)

    def disconnect(self, ws: WebSocket, plan_id: int):
        conns = self.active.get(plan_id, [])
        if ws in conns:
            conns.remove(ws)
            if not conns:
                del self.active[plan_id]

    async def broadcast(self, plan_id: int, message: dict):
        for ws in self.active.get(plan_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                logger.exception("Failed to send execution update to WebSocket client for plan %s", plan_id)


execution_manager = ExecutionConnectionManager()


@router.websocket("/ws/execution/{plan_id}")
async def websocket_execution(websocket: WebSocket, plan_id: int):
    """
    Plan-scoped WS.  Re-fetches task list only when an event arrives
    that is relevant to this plan.
    """
    from ..dependencies import get_bridge

    bridge = get_bridge()
    await execution_manager.connect(websocket, plan_id)
    queue = event_bus.subscribe()

    try:
        while True:
            try:
                keys = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            if keys == ["__shutdown__"]:
                break

            tasks = bridge.list_tasks(plan_ids=[plan_id])
            await websocket.send_json({
                "type": "tasks_update",
                "plan_id": plan_id,
                "tasks": [t if isinstance(t, dict) else t.dict() for t in tasks],
            })
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket execution handler failed for plan %s", plan_id)
    finally:
        event_bus.unsubscribe(queue)
        execution_manager.disconnect(websocket, plan_id)
