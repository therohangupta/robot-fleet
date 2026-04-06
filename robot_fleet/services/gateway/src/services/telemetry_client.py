"""
Client for querying the Telemetry service.

Gateway uses this to get robot health derived from heartbeats instead of polling each robot.
"""

import logging
from typing import Dict, Any, Optional

import httpx

from ..config import TELEMETRY_URL

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=5.0)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def get_health_summary() -> Dict[str, Dict[str, Any]]:
    """
    Fetch health summary for all robots from the Telemetry service.

    Returns a dict keyed by robot_id with {last_seen, reachable, busy}.
    Returns empty dict on error (so UI degrades gracefully).
    """
    try:
        client = await get_client()
        resp = await client.get(f"{TELEMETRY_URL}/health/summary")
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Telemetry /health/summary returned %d", resp.status_code)
    except Exception as e:
        logger.warning("Failed to fetch health summary from Telemetry: %s", e)
    return {}


async def get_robot_health(robot_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch health for a single robot from the Telemetry service.

    Returns None if robot is unknown or on error.
    """
    try:
        client = await get_client()
        resp = await client.get(f"{TELEMETRY_URL}/health/{robot_id}")
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        logger.warning("Telemetry /health/%s returned %d", robot_id, resp.status_code)
    except Exception as e:
        logger.warning("Failed to fetch health for %s from Telemetry: %s", robot_id, e)
    return None
