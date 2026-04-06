"""
Fire-and-forget event emitter for the fleet server.

After any mutation (task/plan/robot), call emit() to notify the Gateway.
The Gateway uses these to push WS invalidation to connected clients.
"""

import asyncio
import logging
from typing import Optional

import httpx

from packages.config import GATEWAY_EVENT_URL

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=3.0)
    return _client


async def _post_event(payload: dict) -> None:
    try:
        resp = await _get_client().post(GATEWAY_EVENT_URL, json=payload)
        logger.debug("Event sent (%s): %s", resp.status_code, payload.get("type"))
    except Exception as exc:
        logger.debug("Event delivery failed (non-fatal): %s", exc)


def emit(event_type: str, **kwargs) -> None:
    """Fire-and-forget: schedule an event POST without blocking the caller."""
    payload = {"type": event_type, **kwargs}
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_post_event(payload))
    except RuntimeError:
        pass  # No running event loop (expected in sync contexts)
    except Exception:
        logger.exception("Unexpected error scheduling event emission")


def emit_task_changed(task_id: int, plan_id: Optional[int] = None, status: Optional[str] = None) -> None:
    emit("task.state_changed", task_id=task_id, plan_id=plan_id, status=status)


def emit_plan_changed(plan_id: int, status: Optional[str] = None) -> None:
    emit("plan.state_changed", plan_id=plan_id, status=status)


def emit_robot_changed(robot_id: str, action: str = "updated") -> None:
    emit("robot.changed", robot_id=robot_id, action=action)
