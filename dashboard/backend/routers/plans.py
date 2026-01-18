"""
Plan management endpoints.

Handles plan creation (auto-generated and manual), allocation,
execution, and status queries.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..dependencies import get_bridge, GRPCBridge
from ..models.requests import PlanCreate, ManualPlanCreate
from ..models.responses import PlanResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plans")


# =============================================================================
# Request Models (local to this router)
# =============================================================================

class AllocatePlanRequest(BaseModel):
    """Request body for plan allocation."""
    allocation_strategy: str  # lp, llm, cost_based


# =============================================================================
# Plan CRUD
# =============================================================================

@router.get("", response_model=List[PlanResponse])
async def list_plans(bridge: GRPCBridge = Depends(get_bridge)):
    """List all plans in the system."""
    return await bridge.list_plans()


@router.get("/{plan_id}", response_model=PlanResponse)
async def get_plan(
    plan_id: int,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Get a specific plan by ID with its tasks."""
    plan = await bridge.get_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
    return plan


@router.post("", response_model=PlanResponse)
async def create_plan(
    plan: PlanCreate,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Create a new plan using automated planning and allocation.
    
    Uses the specified planning strategy (monolithic, dag, big_dag)
    and allocation strategy (lp, llm, cost_based, none) to generate
    a task DAG for the given goals.
    """
    result = bridge.create_plan(
        planning_strategy=plan.planning_strategy,
        allocation_strategy=plan.allocation_strategy,
        goal_ids=plan.goal_ids
    )
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create plan")
    return result


@router.delete("/{plan_id}")
async def delete_plan(
    plan_id: int,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Delete a plan and its associated tasks."""
    result = bridge.delete_plan(plan_id)
    if not result.get("success"):
        raise HTTPException(
            status_code=400, 
            detail=result.get("message", "Deletion failed")
        )
    return {"success": True, "message": f"Plan {plan_id} deleted"}


# =============================================================================
# Manual Plan Creation
# =============================================================================

@router.post("/manual", response_model=PlanResponse)
async def create_manual_plan(
    plan_data: ManualPlanCreate,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Create a plan manually by specifying tasks and dependencies.
    
    Bypasses the automated planner to let you define your own task DAG.
    Tasks reference each other using temporary IDs (temp_id) which are
    mapped to real task IDs after creation.
    
    Each task MUST have a goal_id assigned. The plan's goal_ids are
    derived from the tasks automatically.
    """
    if not plan_data.tasks:
        raise HTTPException(status_code=400, detail="At least one task is required")
    
    # Validate all tasks have goal_id
    tasks_without_goals = [t.temp_id for t in plan_data.tasks if not t.goal_id]
    if tasks_without_goals:
        raise HTTPException(
            status_code=400, 
            detail=f"All tasks must have a goal_id. Missing: {', '.join(tasks_without_goals)}"
        )
    
    # Validate dependencies reference valid temp_ids
    temp_ids = {t.temp_id for t in plan_data.tasks}
    for task in plan_data.tasks:
        for dep in task.depends_on:
            if dep not in temp_ids:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Task '{task.temp_id}' depends on unknown task '{dep}'"
                )
    
    # Check for circular dependencies
    if _has_cycle(plan_data.tasks):
        raise HTTPException(
            status_code=400, 
            detail="Circular dependency detected in tasks"
        )
    
    # Derive goal_ids from tasks
    derived_goal_ids = list(set(t.goal_id for t in plan_data.tasks if t.goal_id))
    
    # Create the plan shell (manual strategy)
    result = bridge.create_manual_plan(goal_ids=derived_goal_ids)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create plan")
    
    plan_id = result["plan_id"]
    
    # Create tasks in dependency order
    sorted_tasks = _topo_sort(plan_data.tasks)
    temp_to_real: dict[str, int] = {}
    
    for task_def in sorted_tasks:
        # Convert temp dependency IDs to real task IDs
        real_deps = [temp_to_real[dep] for dep in task_def.depends_on]
        
        task_result = bridge.create_task(
            description=task_def.description,
            goal_id=task_def.goal_id,
            plan_id=plan_id,
            robot_id=task_def.robot_id,
            robot_type=task_def.robot_type,
            dependency_task_ids=real_deps
        )
        
        if task_result:
            temp_to_real[task_def.temp_id] = task_result["task_id"]
    
    # Return the complete plan with tasks
    return await bridge.get_plan(plan_id)


def _has_cycle(tasks) -> bool:
    """Check for circular dependencies using DFS."""
    visited = set()
    rec_stack = set()
    adj = {t.temp_id: t.depends_on for t in tasks}
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False
    
    for t in tasks:
        if t.temp_id not in visited:
            if dfs(t.temp_id):
                return True
    return False


def _topo_sort(tasks) -> list:
    """Topological sort for creating tasks in dependency order."""
    in_degree = {t.temp_id: 0 for t in tasks}
    adj = {t.temp_id: [] for t in tasks}
    task_map = {t.temp_id: t for t in tasks}
    
    for t in tasks:
        for dep in t.depends_on:
            adj[dep].append(t.temp_id)
            in_degree[t.temp_id] += 1
    
    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    order = []
    
    while queue:
        node = queue.pop(0)
        order.append(task_map[node])
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return order


# =============================================================================
# Plan Allocation
# =============================================================================

@router.post("/{plan_id}/allocate")
async def allocate_plan(
    plan_id: int,
    request: AllocatePlanRequest,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Allocate robots to tasks in an existing plan.
    
    Use this to run allocation on an unallocated or partially allocated plan.
    Available strategies: lp, llm, cost_based
    """
    valid_strategies = ["lp", "llm", "cost_based"]
    if request.allocation_strategy not in valid_strategies:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid allocation strategy. Must be one of: {valid_strategies}"
        )
    
    result = bridge.allocate_plan(plan_id, request.allocation_strategy)
    if not result.get("success"):
        raise HTTPException(
            status_code=400, 
            detail=result.get("message", "Allocation failed")
        )
    return result.get("plan")


@router.get("/{plan_id}/status")
async def get_plan_allocation_status(
    plan_id: int,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Get the allocation status of a plan.
    
    Returns:
        - status: 'empty' | 'unallocated' | 'partially_allocated' | 'fully_allocated'
        - total_tasks: Number of tasks in the plan
        - allocated_tasks: Number of tasks with robot assignments
        - unallocated_task_ids: List of task IDs without assignments
        - is_executable: True only if fully_allocated with tasks
    """
    result = await bridge.get_plan_allocation_status(plan_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# =============================================================================
# Plan Execution
# =============================================================================

@router.post("/{plan_id}/start")
async def start_plan(
    plan_id: int,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Start executing a plan.

    The plan must be fully allocated before execution can begin.
    """
    result = bridge.start_plan(plan_id)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"success": True, "message": f"Plan {plan_id} started"}


@router.post("/{plan_id}/copy")
async def copy_plan(
    plan_id: int,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Copy a plan for re-execution. Creates a new plan with the same goals, tasks,
    and allocation results, but resets execution status to not_executed.
    """
    try:
        # Get the original plan with full details
        original_plan = await bridge.get_plan(plan_id)
        if not original_plan:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")

        # Create the new plan with the same parameters as the original
        result = bridge.create_plan(
            planning_strategy=original_plan.get("planning_strategy", "big_dag"),
            allocation_strategy=original_plan.get("allocation_strategy", "llm"),
            goal_ids=original_plan.get("goal_ids", [])
        )
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        new_plan = result.get("plan")
        if not new_plan:
            raise HTTPException(status_code=500, detail="Failed to create plan copy")

        new_plan_id = new_plan.get("plan_id")
        if not new_plan_id:
            raise HTTPException(status_code=500, detail="New plan missing plan_id")

        # Copy allocation artifacts from original plan to new plan
        # This preserves the allocation results without re-running allocation
        if original_plan.get("allocation_artifacts"):
            try:
                # Update the new plan with copied allocation artifacts and reset execution status
                update_result = await bridge.registry.update_plan(
                    plan_id=new_plan_id,
                    allocation_artifacts=original_plan["allocation_artifacts"],
                    planning_artifacts=original_plan.get("planning_artifacts"),
                    allocation_prompts=original_plan.get("allocation_prompts"),
                    planning_prompts=original_plan.get("planning_prompts"),
                    server_logs=original_plan.get("server_logs", []),
                    execution_status=0  # Reset to not_executed
                )

                if not update_result:
                    logger.warning(f"Failed to update copied plan {new_plan_id} with artifacts")

            except Exception as e:
                logger.error(f"Failed to copy artifacts to new plan {new_plan_id}: {e}")
                # Continue anyway - the plan was created successfully

        return new_plan

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to copy plan {plan_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to copy plan: {str(e)}")
