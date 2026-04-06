"""
Telemetry service configuration.

Re-exports from the shared packages.config for backward compatibility.
All actual config values live in packages/config.py (single source of truth).
"""

from packages.config import (
    CORS_ORIGINS,
    GATEWAY_EVENT_URL,
    TELEMETRY_PORT,
    MAX_HEARTBEATS_PER_ROBOT,
    HEARTBEAT_REACHABLE_THRESHOLD_SECS,
    HEARTBEAT_SCANNER_INTERVAL_SECS,
)
