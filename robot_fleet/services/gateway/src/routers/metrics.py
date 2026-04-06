"""
Metrics query endpoint for observability.

Exposes stored MetricEvent rows from the database
with optional filtering by service, event_type, and time window.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_bridge

router = APIRouter(prefix="/metrics", tags=["Metrics"])
logger = logging.getLogger(__name__)


@router.get("")
async def list_metrics(
    service: Optional[str] = Query(None, description="Filter by service name"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    since: Optional[int] = Query(None, description="Only events from the last N seconds"),
    limit: int = Query(200, ge=1, le=1000, description="Max rows to return"),
):
    """Return stored metric events with optional filters."""
    bridge = get_bridge()
    try:
        rows = await bridge.registry.list_metrics(
            service=service,
            event_type=event_type,
            since_seconds=since,
            limit=limit,
        )
        return {"metrics": rows, "count": len(rows)}
    except Exception:
        logger.exception("Failed to query metrics")
        return {"metrics": [], "count": 0, "error": "Failed to query metrics"}
