"""
Robot Fleet Dashboard - FastAPI Application Factory

This module creates and configures the FastAPI application.
All routes are organized into routers in the routers/ directory.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS
from .dependencies import init_bridge, close_bridge
from .routers import api_router
from .routers.websocket import router as websocket_router
from .routers.embodiments import ports_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Initializes the gRPC bridge on startup and closes it on shutdown.
    """
    init_bridge()
    yield
    close_bridge()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Robot Fleet Dashboard API",
        description="""
REST API for managing robot fleets, goals, plans, and execution.

## Features

- **Robots**: Register, monitor, and manage robot instances
- **Goals**: Define high-level objectives for the fleet
- **Plans**: Generate task DAGs using LLM planners
- **Execution**: Monitor real-time plan execution via WebSocket
- **World State**: Manage environmental facts for planning context
        """,
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include all API routers
    app.include_router(api_router)
    
    # Include ports router at /api/ports for backwards compatibility
    app.include_router(ports_router, prefix="/api", tags=["Ports"])
    
    # Include WebSocket router (no /api prefix)
    app.include_router(websocket_router)
    
    # Health check endpoint (root level)
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        return {"status": "healthy", "service": "robot-fleet-dashboard"}
    
    return app


# Create the application instance
app = create_app()
