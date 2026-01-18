"""
Goal management endpoints.

Handles CRUD operations for goals - the high-level objectives
that plans are created to achieve.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Depends

from ..dependencies import get_bridge, GRPCBridge
from ..models.requests import GoalCreate
from ..models.responses import GoalResponse

router = APIRouter(prefix="/goals")


@router.get("", response_model=List[GoalResponse])
async def list_goals(bridge: GRPCBridge = Depends(get_bridge)):
    """List all goals in the system."""
    return bridge.list_goals()


@router.get("/{goal_id}", response_model=GoalResponse)
async def get_goal(
    goal_id: int,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Get a specific goal by ID."""
    goal = bridge.get_goal(goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail=f"Goal {goal_id} not found")
    return goal


@router.post("", response_model=GoalResponse)
async def create_goal(
    goal: GoalCreate,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Create a new goal.
    
    Goals are high-level objectives that the robot fleet should accomplish.
    Once created, goals can be included in plans for execution.
    """
    result = bridge.create_goal(goal.description)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create goal")
    return result


@router.delete("/{goal_id}")
async def delete_goal(
    goal_id: int,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Delete a goal."""
    result = bridge.delete_goal(goal_id)
    if not result.get("success"):
        raise HTTPException(
            status_code=400, 
            detail=result.get("message", "Deletion failed")
        )
    return {"success": True, "message": f"Goal {goal_id} deleted"}
