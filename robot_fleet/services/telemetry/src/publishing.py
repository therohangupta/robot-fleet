"""
Publisher implementations for telemetry events.

Default: HTTP POST to gateway. Later: Kafka or composite (HTTP + Kafka).
"""

import logging

import httpx

from .config import GATEWAY_EVENT_URL
from .events import HealthChangedEvent

logger = logging.getLogger(__name__)


class GatewayHealthChangedPublisher:
    """Publish health_changed events to the gateway via HTTP (current behavior)."""

    def __init__(self, gateway_event_url: str | None = None):
        self._url = gateway_event_url or GATEWAY_EVENT_URL
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=2.0)
        return self._client

    async def publish(self, event: HealthChangedEvent) -> None:
        if not event.robot_ids:
            return
        try:
            client = await self._get_client()
            resp = await client.post(self._url, json=event.model_dump())
            if resp.status_code >= 400:
                logger.warning(
                    "Gateway event POST returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as e:
            logger.warning("Failed to publish health_changed to gateway: %s", e)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
