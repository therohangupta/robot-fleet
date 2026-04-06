"""
Shared configuration for all robot_fleet services.

████  SINGLE SOURCE OF TRUTH  ████

Every port, hostname, and URL used by any Python service MUST be defined
here, then imported.  Only truly sensitive data (API keys) lives in .env.

At import time, each value falls back to a sensible *native-host* default
(localhost).  Docker Compose / scripts override via environment variables.
"""

import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent.resolve()

EMBODIMENTS_DIR = REPO_ROOT / "robots" / "fake"

PLANNER_TYPES_DIR = REPO_ROOT / "services" / "fleet_server" / "src" / "planners" / "types"
ALLOCATOR_TYPES_DIR = REPO_ROOT / "services" / "fleet_server" / "src" / "allocators" / "types"

# =============================================================================
# Database
# =============================================================================

DB_USER = os.environ.get("DB_USER", "robot_user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "secret")
DB_NAME = os.environ.get("DB_NAME", "robot_fleet")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

# =============================================================================
# Service Ports  (one place — grep "PORT" here, not across 30 files)
# =============================================================================

GRPC_SERVER_PORT = int(os.environ.get("GRPC_PORT", "50051"))
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8000"))
TELEMETRY_PORT = int(os.environ.get("TELEMETRY_PORT", "9000"))
FRONTEND_DEV_PORT = 5173

# =============================================================================
# Service Hosts
# =============================================================================

GRPC_SERVER_HOST = os.environ.get("GRPC_SERVER_HOST", "localhost")
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "localhost")
TELEMETRY_HOST = os.environ.get("TELEMETRY_HOST", "localhost")

# Default host robots register with.  On Mac Docker Desktop
# host.docker.internal resolves to 127.0.0.1 from both host and containers.
DEFAULT_ROBOT_HOST = os.environ.get("DEFAULT_ROBOT_HOST", "localhost")

# =============================================================================
# Derived URLs  (built from host + port above)
# =============================================================================

GRPC_SERVER_ADDRESS = os.environ.get(
    "GRPC_SERVER_ADDRESS",
    f"{GRPC_SERVER_HOST}:{GRPC_SERVER_PORT}",
)

GATEWAY_URL = os.environ.get(
    "GATEWAY_URL",
    f"http://{GATEWAY_HOST}:{GATEWAY_PORT}",
)

GATEWAY_EVENT_URL = os.environ.get(
    "GATEWAY_EVENT_URL",
    f"{GATEWAY_URL}/internal/events",
)

TELEMETRY_URL = os.environ.get(
    "TELEMETRY_URL",
    f"http://{TELEMETRY_HOST}:{TELEMETRY_PORT}",
)

# =============================================================================
# Robot Defaults
# =============================================================================

DEFAULT_ROBOT_BASE_PORT = int(os.environ.get("DEFAULT_ROBOT_BASE_PORT", "8001"))

# =============================================================================
# Health / Heartbeat Tuning
# =============================================================================

HEALTH_CHECK_TIMEOUT_SECS = 2.0

ROBOT_HEARTBEAT_INTERVAL_SECS = float(
    os.environ.get("HEARTBEAT_INTERVAL", "0.5")
)

HEARTBEAT_MISS_TOLERANCE = 3

HEARTBEAT_REACHABLE_THRESHOLD_SECS = (
    ROBOT_HEARTBEAT_INTERVAL_SECS * HEARTBEAT_MISS_TOLERANCE + 0.5
)

HEARTBEAT_SCANNER_INTERVAL_SECS = 0.5

MAX_HEARTBEATS_PER_ROBOT = 20

# =============================================================================
# CORS Configuration
# =============================================================================

CORS_ORIGINS = [
    f"http://localhost:{FRONTEND_DEV_PORT}",
    "http://localhost:3000",
]
