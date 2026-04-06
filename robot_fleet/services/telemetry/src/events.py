"""
Event models and publisher protocol for telemetry fan-out.

Stable event shapes so the same payload can be sent via HTTP today or Kafka later.
Pluggable publisher: ingest calls the protocol; implementation can be gateway HTTP or a message bus.
"""

from typing import Protocol

from pydantic import BaseModel


class HealthChangedEvent(BaseModel):
    """Event emitted when effective reachable status changes for one or more robots."""

    type: str = "telemetry.health_changed"
    robot_ids: list[str]
    ts: float | None = None


class HealthChangedPublisher(Protocol):
    """Pluggable fan-out: implement with HTTP to gateway, Kafka, or both."""

    async def publish(self, event: HealthChangedEvent) -> None:
        """Publish a health_changed event. Fire-and-forget; log and ignore errors."""
        ...
