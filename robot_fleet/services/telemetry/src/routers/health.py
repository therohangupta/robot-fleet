"""
Health read API for the Telemetry service.

Gateway (and other consumers) query these endpoints to get robot health derived from heartbeats.
"""

from fastapi import APIRouter, HTTPException

from ..heartbeat_store import get_heartbeat_store

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/summary")
async def get_health_summary():
    """
    Get health summary for all known robots.

    Returns a dict keyed by robot_id with last_seen, reachable, and busy.
    """
    store = get_heartbeat_store()
    return store.get_all_health()


@router.get("/{robot_id}")
async def get_robot_health(robot_id: str):
    """
    Get health summary for a single robot.

    Returns 404 if the robot is unknown (has never sent a heartbeat).
    """
    store = get_heartbeat_store()
    health = store.get_health(robot_id)
    if health is None:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found in telemetry store")
    return health
