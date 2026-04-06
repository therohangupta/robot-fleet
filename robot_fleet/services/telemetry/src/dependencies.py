"""FastAPI dependencies for the telemetry service."""

from fastapi import Request

from .events import HealthChangedPublisher


def get_publisher(request: Request) -> HealthChangedPublisher:
    """Return the pluggable health_changed publisher (set in app lifespan)."""
    return request.app.state.health_changed_publisher
