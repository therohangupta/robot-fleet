"""
Telemetry service FastAPI application.

Provides:
- /ingest/heartbeat — robots POST here (latest-wins coalescing, pluggable fan-out)
- /health/summary, /health/{robot_id} — gateway queries here
- /healthz — liveness probe
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, HEARTBEAT_SCANNER_INTERVAL_SECS
from .events import HealthChangedEvent
from .heartbeat_store import get_heartbeat_store
from .publishing import GatewayHealthChangedPublisher
from .routers import ingest, health

logger = logging.getLogger(__name__)


async def _timeout_scanner(publisher: GatewayHealthChangedPublisher) -> None:
    """Periodically check for robots that stopped heartbeating and push events."""
    store = get_heartbeat_store()
    while True:
        await asyncio.sleep(HEARTBEAT_SCANNER_INTERVAL_SECS)
        try:
            timed_out = store.check_timeouts()
            if timed_out:
                logger.info("Timeout detected for %s", timed_out)
                await publisher.publish(
                    HealthChangedEvent(robot_ids=timed_out, ts=time.time())
                )
        except Exception:
            logger.exception("Error in timeout scanner")


@asynccontextmanager
async def lifespan(app: FastAPI):
    publisher = GatewayHealthChangedPublisher()
    app.state.health_changed_publisher = publisher
    scanner = asyncio.create_task(_timeout_scanner(publisher))
    yield
    scanner.cancel()
    try:
        await scanner
    except asyncio.CancelledError:
        pass
    await publisher.close()


app = FastAPI(
    title="Telemetry Service",
    description="Heartbeat ingest and health read API for the robot fleet",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(health.router)


@app.get("/healthz", tags=["Meta"])
async def healthz():
    """Liveness probe for orchestrators."""
    return {"status": "ok"}
