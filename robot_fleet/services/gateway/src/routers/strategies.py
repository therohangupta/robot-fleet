"""
Strategy options endpoint.

Returns the available planning and allocation strategies
dynamically from each method's summary.yaml (single source of truth).
"""

from fastapi import APIRouter
from packages.proto import fleet_manager_pb2

from ..services.yaml_scanner import scan_planner_types, scan_allocator_types

router = APIRouter(prefix="/strategies")


@router.get("")
async def get_strategies():
    """
    Get available planning and allocation strategies.

    Built dynamically from each planner/allocator's summary.yaml so there
    is exactly one place where strategy metadata lives.
    """
    planners = scan_planner_types()
    allocators = scan_allocator_types()

    planning = [
        {
            "value": p["type"],
            "id": p["id"],
            "label": p["name"],
            "description": p["description"],
        }
        for p in sorted(planners, key=lambda x: x.get("id", 0))
        if p["type"] != "replanner"
    ]

    allocation = [
        {
            "value": a["type"],
            "id": a["id"],
            "label": a["name"],
            "description": a["description"],
        }
        for a in sorted(allocators, key=lambda x: x.get("id", 0))
    ]

    allocation.append({
        "value": "none",
        "id": int(fleet_manager_pb2.AllocationStrategy.NONE),
        "label": "None (Unallocated)",
        "description": "Skip allocation — assign robots later",
    })

    return {"planning": planning, "allocation": allocation}
