"""
Shared dependencies for FastAPI dependency injection.

This module provides singleton instances and dependency functions
that can be injected into route handlers using FastAPI's Depends().
"""

from typing import Optional
from .grpc_bridge import GRPCBridge

# =============================================================================
# Global Instances
# =============================================================================

# Singleton gRPC bridge instance - initialized on app startup
_bridge: Optional[GRPCBridge] = None


def get_bridge() -> GRPCBridge:
    """
    Get the global GRPCBridge instance.
    
    Use this as a FastAPI dependency:
        @app.get("/api/robots")
        async def list_robots(bridge: GRPCBridge = Depends(get_bridge)):
            return bridge.list_robots()
    
    Raises:
        RuntimeError: If bridge hasn't been initialized (app not started)
    """
    if _bridge is None:
        raise RuntimeError("GRPCBridge not initialized. App may not have started properly.")
    return _bridge


def init_bridge() -> GRPCBridge:
    """
    Initialize the global GRPCBridge instance.
    Called during app startup.
    """
    global _bridge
    _bridge = GRPCBridge()
    return _bridge


def close_bridge() -> None:
    """
    Close the global GRPCBridge instance.
    Called during app shutdown.
    """
    global _bridge
    if _bridge:
        _bridge.close()
        _bridge = None
