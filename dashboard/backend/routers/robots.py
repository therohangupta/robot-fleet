"""
Robot management endpoints.

Handles robot registration, unregistration, health checks,
and YAML-based configuration refresh.
"""

import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..config import PROJECT_ROOT, EMBODIMENTS_DIR
from ..dependencies import get_bridge, GRPCBridge
from ..services import (
    check_robot_health,
    check_all_robots_health,
    get_used_ports,
    is_localhost,
    find_yaml_for_robot,
)
from ..models.requests import RobotInstanceCreate
from ..models.responses import RobotResponse

router = APIRouter(prefix="/robots")


# =============================================================================
# Request Models (local to this router)
# =============================================================================

class RefreshRequest(BaseModel):
    """Request body for robot YAML refresh."""
    config_path: Optional[str] = None


# =============================================================================
# List and Get Robots
# =============================================================================

@router.get("", response_model=List[RobotResponse])
async def list_robots(
    filter: str = "all",
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    List all registered robots.
    
    Args:
        filter: Filter option - 'all', 'deployed', or 'registered'
    """
    return bridge.list_robots(filter)


@router.get("/{robot_id}", response_model=RobotResponse)
async def get_robot(
    robot_id: str,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Get a specific robot by ID."""
    robot = bridge.get_robot_status(robot_id)
    if not robot:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")
    return robot


# =============================================================================
# Robot Registration
# =============================================================================

@router.post("/register")
async def register_robot(
    instance: RobotInstanceCreate,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Register a new robot instance.
    
    Registers a robot from a YAML configuration file with custom
    host and port settings. For localhost robots, validates that
    the port isn't already in use.
    """
    # Resolve config path (support both absolute and relative paths)
    config_path = instance.config_path
    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)
    
    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Config not found: {instance.config_path}"
        )
    
    # For localhost, check if port is already in use
    if is_localhost(instance.host):
        used_ports = get_used_ports()
        if instance.port in used_ports:
            raise HTTPException(
                status_code=400, 
                detail=f"Port {instance.port} is already in use on localhost"
            )
    
    # Register with host and port override
    result = bridge.register_robot_with_host_port(
        config_path=config_path,
        robot_id=instance.robot_id,
        host=instance.host,
        port=instance.port
    )
    
    if not result.get("success"):
        raise HTTPException(
            status_code=400, 
            detail=result.get("message", "Registration failed")
        )
    
    return result.get("robot")


@router.delete("/{robot_id}")
async def unregister_robot(
    robot_id: str,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Unregister a robot from the fleet."""
    result = bridge.unregister_robot(robot_id)
    if not result.get("success"):
        raise HTTPException(
            status_code=400, 
            detail=result.get("message", "Unregistration failed")
        )
    return {"success": True, "message": f"Robot {robot_id} unregistered"}


# =============================================================================
# Health Checks
# =============================================================================

@router.get("/{robot_id}/health")
async def check_robot_health_endpoint(
    robot_id: str,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Check if a specific robot is reachable.
    
    Pings the robot's HTTP endpoint to verify connectivity.
    """
    robot = bridge.get_robot_status(robot_id)
    if not robot:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")
    
    host = robot.get("task_server_info", {}).get("host", "localhost")
    port = robot.get("task_server_info", {}).get("port", 8000)
    
    health = await check_robot_health(host, port)
    return {"robot_id": robot_id, "host": host, "port": port, **health}


@router.get("/health/all")
async def check_all_robots_health_endpoint(
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Check health of all registered robots."""
    robots = bridge.list_robots("all")
    results = await check_all_robots_health(robots)
    return {"robots": results}


# =============================================================================
# YAML Configuration
# =============================================================================

@router.get("/{robot_id}/yaml")
async def get_robot_yaml_details(
    robot_id: str,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Get the original YAML configuration for a robot.
    
    Attempts to locate and parse the YAML file based on the robot's type.
    """
    import yaml
    
    robot = bridge.get_robot_status(robot_id)
    if not robot:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")
    
    # Try to find the YAML file
    yaml_path = find_yaml_for_robot(robot)
    
    if yaml_path and os.path.exists(yaml_path):
        try:
            with open(yaml_path) as f:
                yaml_content = yaml.safe_load(f)
            return {
                "robot": robot,
                "yaml_path": str(Path(yaml_path).relative_to(PROJECT_ROOT)),
                "yaml_content": yaml_content
            }
        except Exception:
            pass
    
    # Return robot info without YAML if not found
    return {
        "robot": robot,
        "yaml_path": None,
        "yaml_content": None
    }


def _refresh_single_robot(
    bridge: GRPCBridge,
    robot_id: str, 
    config_path: Optional[str] = None
) -> dict:
    """
    Refresh a single robot from its YAML configuration.
    
    Re-registers the robot to pick up any capability changes
    from the YAML file while preserving host/port settings.
    """
    robot = bridge.get_robot_status(robot_id)
    if not robot:
        return {"success": False, "robot_id": robot_id, "error": f"Robot {robot_id} not found"}
    
    # Find the YAML path
    yaml_path = None
    if config_path:
        yaml_path = config_path if os.path.isabs(config_path) else str(PROJECT_ROOT / config_path)
    else:
        yaml_path = find_yaml_for_robot(robot)
    
    if not yaml_path or not os.path.exists(yaml_path):
        return {"success": False, "robot_id": robot_id, "error": "Could not find YAML config file"}
    
    # Preserve current host/port
    host = robot.get("task_server_info", {}).get("host", "localhost")
    port = robot.get("task_server_info", {}).get("port", 8000)
    
    # Unregister and re-register with fresh YAML
    bridge.unregister_robot(robot_id)
    result = bridge.register_robot_with_host_port(
        config_path=yaml_path,
        robot_id=robot_id,
        host=host,
        port=port
    )
    
    if not result.get("success"):
        return {"success": False, "robot_id": robot_id, "error": result.get("message", "Failed to refresh")}
    
    return {
        "success": True,
        "robot_id": robot_id,
        "message": f"Robot {robot_id} refreshed from YAML",
        "robot": result.get("robot")
    }


@router.post("/{robot_id}/refresh")
async def refresh_robot_from_yaml(
    robot_id: str,
    request: RefreshRequest = RefreshRequest(),
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Refresh a robot's capabilities from its YAML file.
    
    Use this after updating a robot's YAML to pick up new capabilities
    without needing to manually unregister and re-register.
    """
    result = _refresh_single_robot(bridge, robot_id, request.config_path)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to refresh"))
    return result


@router.post("/refresh/all")
async def refresh_all_robots_from_yaml(
    bridge: GRPCBridge = Depends(get_bridge)
):
    """Refresh all robots from their YAML files."""
    robots = bridge.list_robots("all")
    results = []
    
    for robot in robots:
        result = _refresh_single_robot(bridge, robot["robot_id"])
        results.append(result)
    
    success_count = sum(1 for r in results if r.get("success"))
    failed_count = len(results) - success_count
    
    return {
        "total": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "results": results
    }


# =============================================================================
# Robot Allocations
# =============================================================================

@router.get("/{robot_id}/allocations")
async def get_robot_allocations(
    robot_id: str,
    bridge: GRPCBridge = Depends(get_bridge)
):
    """
    Get allocation details for a specific robot.

    Returns counts of plans, goals, and tasks allocated to this robot.
    """
    try:
        # Query tasks allocated to this robot
        from robot_fleet.robots.registry.models import TaskModel, PlanModel, GoalModel
        from sqlalchemy import select, func, distinct

        async with bridge.registry.async_session_factory() as session:
            # Get all tasks allocated to this robot
            task_query = select(
                TaskModel.task_id,
                TaskModel.plan_id,
                TaskModel.goal_id,
                PlanModel.execution_status.label('plan_status')
            ).join(
                PlanModel, TaskModel.plan_id == PlanModel.plan_id
            ).where(
                TaskModel.robot_id == robot_id,
                TaskModel.robot_id.isnot(None)  # Only allocated tasks
            )

            result = await session.execute(task_query)
            tasks_data = result.fetchall()

            if not tasks_data:
                return {
                    "robot_id": robot_id,
                    "plans_count": 0,
                    "goals_count": 0,
                    "tasks_count": 0,
                    "plans": [],
                    "goals": []
                }

            # Extract unique plan IDs from tasks
            plan_ids = set()
            plan_statuses = {}

            for task in tasks_data:
                if task.plan_id:
                    plan_ids.add(task.plan_id)
                    plan_statuses[task.plan_id] = task.plan_status

            # Get plan details for the summary and collect goal IDs
            plans_summary = []
            goal_ids = set()

            if plan_ids:
                plan_details_query = select(
                    PlanModel.plan_id,
                    PlanModel.goal_ids,
                    PlanModel.execution_status,
                    PlanModel.name,
                    PlanModel.description
                ).where(PlanModel.plan_id.in_(plan_ids))

                plan_result = await session.execute(plan_details_query)
                plans_data = plan_result.fetchall()

                for plan in plans_data:
                    # Count tasks for this robot in this plan
                    robot_tasks_in_plan = sum(1 for task in tasks_data if task.plan_id == plan.plan_id)
                    plans_summary.append({
                        "plan_id": plan.plan_id,
                        "goal_ids": plan.goal_ids or [],
                        "task_count": robot_tasks_in_plan,
                        "status": ["not_executed", "executing", "completed", "failed"][plan.execution_status or 0],
                        "name": plan.name,
                        "description": plan.description
                    })

                    # Collect goal IDs from this plan
                    if plan.goal_ids:
                        goal_ids.update(plan.goal_ids)

            return {
                "robot_id": robot_id,
                "plans_count": len(plan_ids),
                "goals_count": len(goal_ids),
                "tasks_count": len(tasks_data),
                "plans": plans_summary,
                "goals": list(goal_ids)
            }

    except Exception as e:
        logger.error(f"Error getting allocations for robot {robot_id}: {e}")
        import traceback
        traceback.print_exc()
        # Return empty data on error
        return {
            "robot_id": robot_id,
            "plans_count": 0,
            "goals_count": 0,
            "tasks_count": 0,
            "plans": [],
            "goals": []
        }


# Import Path for relative path handling
from pathlib import Path
