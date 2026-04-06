"""
Embodiment (robot type template) endpoints.

Handles discovery and querying of robot embodiments - the YAML
configurations that define robot types and their capabilities.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from ..services import scan_embodiments, get_used_ports, suggest_next_port
from ..models.responses import EmbodimentResponse
from ..config import DEFAULT_ROBOT_BASE_PORT

router = APIRouter(prefix="/embodiments")


@router.get("", response_model=List[EmbodimentResponse])
async def list_embodiments():
    """
    List all available robot embodiments.
    
    Scans the embodiments directory for YAML configurations
    that define robot types and their capabilities.
    """
    return scan_embodiments()


@router.get("/{name}")
async def get_embodiment(name: str):
    """Get a specific embodiment by name."""
    embodiments = scan_embodiments()
    for emb in embodiments:
        if emb["name"].lower() == name.lower():
            return emb
    raise HTTPException(status_code=404, detail=f"Embodiment {name} not found")


# =============================================================================
# Port Management (for local robot instances)
# =============================================================================

@router.get("/ports/suggest", include_in_schema=False)
async def suggest_port_endpoint(base_port: int = DEFAULT_ROBOT_BASE_PORT):
    """Suggest the next available port for a new robot instance."""
    return {"suggested_port": suggest_next_port(base_port)}


@router.get("/ports/used", include_in_schema=False)
async def get_used_ports_endpoint():
    """Get list of ports currently in use."""
    return {"used_ports": list(get_used_ports())}


# Also expose at /api/ports/* for backwards compatibility
from fastapi import APIRouter as _APIRouter
ports_router = _APIRouter(prefix="/ports")


@ports_router.get("/suggest")
async def suggest_port_compat(base_port: int = DEFAULT_ROBOT_BASE_PORT):
    """Suggest the next available port for a new robot instance."""
    return {"suggested_port": suggest_next_port(base_port)}


@ports_router.get("/used")
async def get_used_ports_compat():
    """Get list of ports currently in use."""
    return {"used_ports": list(get_used_ports())}
