"""
World state endpoints.

Handles world statements - facts about the environment that
the planner uses to generate appropriate task sequences.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Depends

from ..dependencies import get_bridge, GRPCBridge
from ..models.requests import WorldStatementCreate
from ..models.responses import WorldStatementResponse

router = APIRouter(prefix="/world")


@router.get("", response_model=List[WorldStatementResponse])
async def list_world_statements(bridge: GRPCBridge = Depends(get_bridge)):
    """
    List all world statements.
    
    World statements describe facts about the environment that
    the planner considers when generating task sequences.
    """
    return bridge.list_world_statements()


@router.post("", response_model=WorldStatementResponse)
async def add_world_statement(
    statement: WorldStatementCreate,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Add a new world statement.
    
    Examples:
        - "The kitchen is on the first floor"
        - "Object A is currently on Table B"
        - "Robot charging station is in Room C"
    """
    result = bridge.add_world_statement(statement.statement)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to add world statement")
    return result


@router.delete("/{statement_id}")
async def delete_world_statement(
    statement_id: str,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Delete a world statement."""
    result = bridge.delete_world_statement(statement_id)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to delete world statement")
    return {"success": True, "message": f"World statement {statement_id} deleted"}
