"""
Plan management endpoints.

Handles plan creation (auto-generated and manual), allocation,
execution, and status queries.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..dependencies import get_bridge, GRPCBridge
from ..models.requests import PlanCreate, ManualPlanCreate
from ..models.responses import PlanResponse
from ..services.yaml_scanner import get_allocation_strategy_id, scan_allocator_types

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plans")


# =============================================================================
# Request Models (local to this router)
# =============================================================================

class AllocatePlanRequest(BaseModel):
    """Request body for plan allocation."""
    allocation_strategy: str  # lp, llm, cost_based


class PlanCopyRequest(BaseModel):
    """Request body for copying a plan with required name/description updates."""
    name: str = Field(..., description="New name for the copied plan")
    description: str = Field(..., description="New description for the copied plan")


class PlanUpdateRequest(BaseModel):
    """Request body for updating plan name and description."""
    name: str = Field(..., description="Updated name for the plan")
    description: str = Field(..., description="Updated description for the plan")


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
    logger.debug("Creating plan with data: %s", plan.dict())
    try:
        result = bridge.create_plan(
            planning_strategy=plan.planning_strategy,
            allocation_strategy=plan.allocation_strategy,
            goal_ids=plan.goal_ids,
            name=plan.name,
            description=plan.description
        )
        logger.debug("Bridge result: %s", result)
        if not result:
            logger.debug("Bridge returned None")
            raise HTTPException(status_code=400, detail="Failed to create plan")
        logger.debug("Returning plan: %s", result)
        return result
    except Exception as e:
        logger.error("Exception in create_plan: %s", e, exc_info=True)
        raise


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
    logger.debug("Creating manual plan with data: %s", plan_data.dict())
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
    result = bridge.create_manual_plan(goal_ids=derived_goal_ids, name=plan_data.name, description=plan_data.description)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create plan")
    
    plan_id = result["plan_id"]
    
    # Create tasks in dependency order
    sorted_tasks = _topo_sort(plan_data.tasks)
    temp_to_real: dict[str, int] = {}
    robot_type_cache: dict[str, str] = {}
    
    for task_def in sorted_tasks:
        # Convert temp dependency IDs to real task IDs
        real_deps = [temp_to_real[dep] for dep in task_def.depends_on]

        # If a robot is assigned but robot_type is missing, infer it from robot_id via GetRobot
        inferred_robot_type = task_def.robot_type
        if task_def.robot_id and not inferred_robot_type:
            if task_def.robot_id in robot_type_cache:
                inferred_robot_type = robot_type_cache[task_def.robot_id]
            else:
                robot = bridge.get_robot(task_def.robot_id)
                if not robot:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Robot '{task_def.robot_id}' not found (required to infer robot_type)"
                    )
                inferred_robot_type = robot.get("robot_type")
                if inferred_robot_type:
                    robot_type_cache[task_def.robot_id] = inferred_robot_type
        
        task_result = bridge.create_task(
            description=task_def.description,
            goal_id=task_def.goal_id,
            plan_id=plan_id,
            robot_id=task_def.robot_id,
            robot_type=inferred_robot_type,
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
    
    Strategy names are resolved from each allocator's summary.yaml (single source of truth).
    """
    strategy_int = get_allocation_strategy_id(request.allocation_strategy)
    if strategy_int is None:
        available = [m["type"] for m in scan_allocator_types()]
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid allocation strategy '{request.allocation_strategy}'. Must be one of: {available}"
        )
    
    result = bridge.allocate_plan(plan_id, strategy_int)
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
    request: PlanCopyRequest,
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

        # Validate that we have all required plan configuration
        planning_strategy = original_plan.get("planning_strategy")
        allocation_strategy = original_plan.get("allocation_strategy")
        goal_ids = original_plan.get("goal_ids", [])

        if not planning_strategy:
            raise HTTPException(status_code=400, detail=f"Original plan {plan_id} missing planning_strategy")
        if not allocation_strategy:
            raise HTTPException(status_code=400, detail=f"Original plan {plan_id} missing allocation_strategy")

        # Use the provided name and description
        copy_name = request.name
        copy_description = request.description

        # Create the new plan directly in the database (bypass planner)
        try:
            new_plan_proto = await bridge.registry.create_plan(
                planning_strategy=planning_strategy,
                allocation_strategy=allocation_strategy,
                goal_ids=goal_ids,
                task_ids=[],  # We'll add tasks separately
                planning_prompts=original_plan.get("planning_prompts"),
                allocation_prompts=original_plan.get("allocation_prompts"),
                planning_artifacts=original_plan.get("planning_artifacts"),
                allocation_artifacts=original_plan.get("allocation_artifacts"),
                server_logs=original_plan.get("server_logs") or None,
                name=copy_name,
                description=copy_description
            )

            if not new_plan_proto:
                raise HTTPException(status_code=500, detail="Failed to create plan copy")

            new_plan_id = new_plan_proto.plan_id

            # Copy all tasks from the original plan to the new plan
            if original_plan.get("tasks"):
                await bridge.registry.copy_plan_tasks(original_plan["plan_id"], new_plan_id)

            # Reset execution status to not_executed
            await bridge.registry.update_plan(
                plan_id=new_plan_id,
                execution_status=0
            )

            # Get the final plan with copied tasks
            copied_plan = await bridge.get_plan(new_plan_id)
            return copied_plan

        except Exception as e:
            logger.error(f"Failed to copy plan {plan_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to copy plan: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to copy plan {plan_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to copy plan: {str(e)}")


@router.put("/{plan_id}")
async def update_plan(
    plan_id: int,
    request: PlanUpdateRequest,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Update a plan's name and description.

    Only name and description can be updated. Other plan properties
    (strategies, tasks, etc.) remain unchanged.
    """
    try:
        # Get the current plan to ensure it exists
        current_plan = await bridge.get_plan(plan_id)
        if not current_plan:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")

        # Update the plan name and description
        updated_plan_proto = await bridge.registry.update_plan(
            plan_id=plan_id,
            name=request.name,
            description=request.description
        )

        if not updated_plan_proto:
            raise HTTPException(status_code=500, detail="Failed to update plan")

        # Get the updated plan data
        updated_plan = await bridge.get_plan(plan_id)
        return updated_plan

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update plan {plan_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update plan: {str(e)}")
