"""
Configuration constants for the Robot Fleet Dashboard backend.

This module centralizes all configuration values, paths, and constants
used throughout the application. Import from here rather than hardcoding
values in individual modules.
"""

from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

# Project root directory (robot-fleet/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Directory containing robot embodiment YAML configurations
EMBODIMENTS_DIR = PROJECT_ROOT / "robot_fleet" / "robots" / "examples"

# Directory containing planner types (with prompts co-located)
PLANNER_TYPES_DIR = PROJECT_ROOT / "robot_fleet" / "server" / "planners" / "types"

# Directory containing allocator types (with prompts co-located)
ALLOCATOR_TYPES_DIR = PROJECT_ROOT / "robot_fleet" / "server" / "allocators" / "types"

# =============================================================================
# Server Configuration
# =============================================================================

# Fleet Manager gRPC server address
GRPC_SERVER_ADDRESS = "localhost:50051"

# Default port range for local robot instances
DEFAULT_BASE_PORT = 8001

# Health check timeout in seconds
HEALTH_CHECK_TIMEOUT = 2.0

# =============================================================================
# CORS Configuration
# =============================================================================

# Allowed origins for CORS
CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative dev server
]
