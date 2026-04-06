"""
Gateway configuration.

Re-exports from packages.config so gateway code can do `from ..config import X`.
"""

from packages.config import (
    REPO_ROOT,
    EMBODIMENTS_DIR,
    PLANNER_TYPES_DIR,
    ALLOCATOR_TYPES_DIR,
    GRPC_SERVER_ADDRESS,
    GRPC_SERVER_HOST,
    GRPC_SERVER_PORT,
    DATABASE_URL,
    TELEMETRY_URL,
    DEFAULT_ROBOT_BASE_PORT,
    DEFAULT_ROBOT_HOST,
    HEALTH_CHECK_TIMEOUT_SECS,
    CORS_ORIGINS,
)
