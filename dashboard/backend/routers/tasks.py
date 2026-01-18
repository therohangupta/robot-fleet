"""
Task management endpoints.

Handles task queries and creation. Tasks are the atomic units of work
that make up a plan.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends

from ..dependencies import get_bridge, GRPCBridge
from ..models.requests import TaskCreate
from ..models.responses import TaskResponse

router = APIRouter(prefix="/tasks")


@router.get("", response_model=List[TaskResponse])
async def list_tasks(
    plan_id: Optional[int] = None,
    goal_id: Optional[int] = None,
    robot_id: Optional[str] = None,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    List tasks with optional filtering.
    
    Args:
        plan_id: Filter by plan
        goal_id: Filter by goal
        robot_id: Filter by assigned robot
    """
    return bridge.list_tasks(
        plan_ids=[plan_id] if plan_id else None,
        goal_ids=[goal_id] if goal_id else None,
        robot_ids=[robot_id] if robot_id else None
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: int,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Get a specific task by ID."""
    task = bridge.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.post("", response_model=TaskResponse)
async def create_task(
    task: TaskCreate,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Create a new task.
    
    Tasks are typically created as part of a plan, but this endpoint
    allows creating individual tasks for testing or manual plan building.
    """
    result = bridge.create_task(
        description=task.description,
        goal_id=task.goal_id,
        plan_id=task.plan_id,
        robot_id=task.robot_id,
        robot_type=task.robot_type,
        dependency_task_ids=task.dependency_task_ids
    )
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create task")
    return result
