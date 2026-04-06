"""
Lightweight observability: async context manager for timing operations
and recording metrics to the database.

Usage:
    async with track_operation(registry, "fleet_server", "plan_execution", str(plan_id)) as meta:
        meta["tasks"] = total_tasks
        await execute_plan(...)
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)


@asynccontextmanager
async def track_operation(
    registry,
    service: str,
    event_type: str,
    entity_id: Optional[str] = None,
):
    """
    Async context manager that times the wrapped block and records a metric.

    The yielded dict can be mutated to attach extra metadata to the metric.
    Recording is fire-and-forget — failures are logged but never raised.
    """
    start = time.monotonic()
    success = True
    meta: dict = {}
    try:
        yield meta
    except Exception:
        success = False
        raise
    finally:
        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            asyncio.get_running_loop().create_task(
                registry.record_metric(
                    service=service,
                    event_type=event_type,
                    entity_id=entity_id,
                    duration_ms=duration_ms,
                    success=success,
                    metadata=meta if meta else None,
                )
            )
        except RuntimeError:
            pass  # No running event loop
        except Exception:
            logger.debug("Failed to schedule metric recording", exc_info=True)
