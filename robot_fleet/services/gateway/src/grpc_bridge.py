"""
Bridge between the Dashboard API and the Fleet Manager gRPC service.
Uses in-repo packages (proto, fleet_sdk, robot_sdk). Requires the repo to be
installed in the environment: from repo root run `pip install -e .`
"""
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

from packages.proto import fleet_manager_pb2
from packages.fleet_sdk.src.grpc_client import FleetManagerClient
from packages.fleet_sdk.src.instance_registry import RobotInstanceRegistry
from packages.fleet_sdk.src.models import PlanModel, TaskModel, GoalModel
from packages.config import DATABASE_URL, GRPC_SERVER_HOST, GRPC_SERVER_PORT
from packages.robot_sdk.src.schema.yaml_validator import YAMLValidator
from google.protobuf.json_format import MessageToDict
import asyncio


TASK_STATUS_MAP = {
    0: "unknown",
    1: "pending",
    2: "in_progress",
    3: "completed",
    4: "cancelled",
    5: "failed",
}

ROBOT_STATE_MAP = {
    0: "unknown",
    1: "registered",
    2: "deploying",
    3: "running",
    4: "error",
    5: "stopped",
}


class GRPCBridge:
    """Bridge to the Fleet Manager gRPC service."""
    
    def __init__(self, host: str = GRPC_SERVER_HOST, port: int = GRPC_SERVER_PORT, db_url: Optional[str] = None):
        server_address = f"{host}:{port}"
        self.client = FleetManagerClient(server_address=server_address)
        self.validator = YAMLValidator()
        self.db_url = db_url or DATABASE_URL
        self.registry = RobotInstanceRegistry(self.db_url)
    
    def close(self):
        """Close the gRPC client connection."""
        if self.client:
            self.client.close()
    
    def _robot_to_dict(self, robot) -> Dict[str, Any]:
        """Convert a Robot protobuf to a dictionary."""
        return {
            "robot_id": robot.robot_id,
            "robot_type": robot.robot_type,
            "description": robot.description,
            "capabilities": list(robot.capabilities),
            "status": ROBOT_STATE_MAP.get(robot.status.state, "unknown") if robot.status else "unknown",
            "task_server_info": {
                "host": robot.task_server_info.host,
                "port": robot.task_server_info.port,
            } if robot.HasField("task_server_info") else None,
            "container": {
                "container_id": robot.container.container_id,
                "image": robot.container.image,
                "host": robot.container.host,
                "port": robot.container.port,
            } if robot.HasField("container") else None,
            "task_ids": list(robot.task_ids),
        }
    
    def _goal_to_dict(self, goal) -> Dict[str, Any]:
        """Convert a Goal protobuf to a dictionary."""
        return {
            "goal_id": goal.goal_id,
            "description": goal.description,
            "status": "pending",  # Default status for new goals
            "task_ids": list(goal.task_ids),
            "created_at": None,  # Could be added if protobuf has timestamp
        }
    
    def _task_to_dict(self, task) -> Dict[str, Any]:
        """Convert a Task protobuf to a dictionary."""
        return {
            "task_id": task.task_id,
            "description": task.description,
            "goal_id": task.goal_id if task.goal_id else None,
            "plan_id": task.plan_id if task.plan_id else None,
            "robot_id": task.robot_id if task.robot_id else None,
            "robot_type": task.robot_type if task.HasField("robot_type") else None,
            "dependency_task_ids": list(task.dependency_task_ids),
            "status": TASK_STATUS_MAP.get(task.status, "unknown"),
            "result": task.result if task.HasField("result") and task.result else None,
        }
    
    def _plan_to_dict(self, plan, include_tasks: bool = False) -> Dict[str, Any]:
        """Convert a Plan protobuf to a dictionary."""
        result = {
            "plan_id": plan.plan_id,
            "name": plan.name,
            "description": plan.description,
            "planning_strategy": plan.planning_strategy,
            "allocation_strategy": plan.allocation_strategy,
            "task_ids": list(plan.task_ids),
            "goal_ids": list(plan.goal_ids),
        }
        if include_tasks:
            tasks = self.list_tasks(plan_ids=[plan.plan_id])
            result["tasks"] = tasks

        # Note: Additional plan data (prompts, artifacts, logs) is fetched from database
        # in the calling methods (list_plans/get_plan) rather than from protobuf
        # since the protobuf files haven't been regenerated with the new fields

        return result

    async def _get_plan_details_from_db(self, plan_id: int) -> Optional[Dict[str, Any]]:
        """Fetch additional plan details from the database."""
        try:
            # Query the database directly for the additional plan data
            from sqlalchemy import select

            async with self.registry.async_session_factory() as session:
                result = await session.execute(
                    select(PlanModel).where(PlanModel.plan_id == plan_id)
                )
                plan_model = result.scalar_one_or_none()

                if not plan_model:
                    logger.warning(f"No plan found with ID {plan_id}")
                    return None

                # Extract the additional fields we added
                details = {}
                if plan_model.planning_prompts:
                    details['planning_prompts'] = plan_model.planning_prompts
                if plan_model.allocation_prompts:
                    details['allocation_prompts'] = plan_model.allocation_prompts
                if plan_model.planning_artifacts:
                    details['planning_artifacts'] = plan_model.planning_artifacts
                if plan_model.allocation_artifacts:
                    details['allocation_artifacts'] = plan_model.allocation_artifacts
                if plan_model.server_logs:
                    details['server_logs'] = plan_model.server_logs
                if plan_model.created_at:
                    details['created_at'] = plan_model.created_at.isoformat()

                # Add execution status
                execution_status_map = {0: 'not_executed', 1: 'executing', 2: 'completed', 3: 'failed'}
                details['execution_status'] = execution_status_map.get(plan_model.execution_status, 'not_executed')

                return details
        except Exception as e:
            logger.error("Error fetching plan details from DB: %s", e, exc_info=True)
            return None

    def _world_statement_to_dict(self, ws) -> Dict[str, Any]:
        """Convert a WorldStatement protobuf to a dictionary."""
        return {
            "id": ws.id,
            "statement": ws.statement,
            "created_at": ws.created_at.ToDatetime().isoformat() if ws.HasField("created_at") else None,
        }
    
    # =========================================================================
    # Robot Operations
    # =========================================================================
    
    def list_robots(self, filter_type: str = "all") -> List[Dict[str, Any]]:
        """List all robots."""
        response = self.client.list_robots(filter_type)
        return [self._robot_to_dict(r) for r in response.robots]
    
    def get_robot_status(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific robot."""
        try:
            robots = self.list_robots()
            for robot in robots:
                if robot["robot_id"] == robot_id:
                    return robot
            return None
        except Exception as e:
            logger.error("Error getting robot status for %s: %s", robot_id, e)
            return None

    def get_robot(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific robot by ID."""
        try:
            response = self.client.get_robot(robot_id)
            if hasattr(response, "error") and response.error:
                return None
            if hasattr(response, "HasField") and response.HasField("robot"):
                return self._robot_to_dict(response.robot)
            return None
        except Exception as e:
            logger.error("Error getting robot %s: %s", robot_id, e)
            return None
    
    def register_robot_from_yaml(self, config_path: str, robot_id: Optional[str] = None) -> Dict[str, Any]:
        """Register a robot from a YAML config file."""
        try:
            config = self.validator.validate_file(config_path)
            robot_type = config['metadata']['name']
            rid = robot_id if robot_id else f"{robot_type}-1"
            
            response = self.client.register_robot(
                robot_id=rid,
                robot_type=robot_type,
                description=config['metadata'].get('description', ''),
                capabilities=config.get('capabilities', []),
                task_server_host=config['taskServer']['host'],
                task_server_port=config['taskServer']['port'],
                docker_host=config['deployment']['docker_host'],
                docker_port=config['deployment']['docker_port'],
                container_image=config['container']['image'],
                container_env=config['container'].get('environment', {})
            )
            
            return {
                "success": response.success,
                "message": response.message,
                "robot": self._robot_to_dict(response.robot) if response.robot else None,
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def register_robot_with_port(self, config_path: str, robot_id: str, port: int) -> Dict[str, Any]:
        """Register a robot from a YAML config file with a custom port override."""
        try:
            config = self.validator.validate_file(config_path)
            robot_type = config['metadata']['name']
            
            response = self.client.register_robot(
                robot_id=robot_id,
                robot_type=robot_type,
                description=config['metadata'].get('description', ''),
                capabilities=config.get('capabilities', []),
                task_server_host=config['taskServer']['host'],
                task_server_port=port,  # Use the provided port instead of YAML default
                docker_host=config['deployment']['docker_host'],
                docker_port=port,  # Also update docker port
                container_image=config['container']['image'],
                container_env=config['container'].get('environment', {})
            )
            
            return {
                "success": response.success,
                "message": response.message,
                "robot": self._robot_to_dict(response.robot) if response.robot else None,
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def register_robot_with_host_port(self, config_path: str, robot_id: str, host: str, port: int) -> Dict[str, Any]:
        """Register a robot from a YAML config file with custom host and port."""
        try:
            config = self.validator.validate_file(config_path)
            robot_type = config['metadata']['name']
            
            response = self.client.register_robot(
                robot_id=robot_id,
                robot_type=robot_type,
                description=config['metadata'].get('description', ''),
                capabilities=config.get('capabilities', []),
                task_server_host=host,  # Use the provided host
                task_server_port=port,  # Use the provided port
                docker_host=config['deployment']['docker_host'],
                docker_port=port,
                container_image=config['container']['image'],
                container_env=config['container'].get('environment', {})
            )
            
            return {
                "success": response.success,
                "message": response.message,
                "robot": self._robot_to_dict(response.robot) if response.robot else None,
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def unregister_robot(self, robot_id: str) -> Dict[str, Any]:
        """Unregister a robot."""
        try:
            response = self.client.unregister_robot(robot_id)
            return {"success": response.success, "message": response.message}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    # =========================================================================
    # Goal Operations
    # =========================================================================
    
    def list_goals(self) -> List[Dict[str, Any]]:
        """List all goals."""
        response = self.client.list_goals()
        return [self._goal_to_dict(g) for g in response.goals]
    
    def get_goal(self, goal_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific goal."""
        try:
            response = self.client.get_goal(goal_id)
            if response.goal and response.goal.goal_id:
                return self._goal_to_dict(response.goal)
            return None
        except Exception as e:
            logger.error("Error getting goal %s: %s", goal_id, e)
            return None
    
    def create_goal(self, description: str) -> Optional[Dict[str, Any]]:
        """Create a new goal."""
        try:
            response = self.client.create_goal(description)
            if response.goal:
                return self._goal_to_dict(response.goal)
            return None
        except Exception as e:
            logger.error("Error creating goal: %s", e)
            return None
    
    def delete_goal(self, goal_id: int) -> Dict[str, Any]:
        """Delete a goal."""
        try:
            response = self.client.delete_goal(goal_id)
            return {"success": bool(response.goal), "message": response.error or "Deleted"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    # =========================================================================
    # Plan Operations
    # =========================================================================
    
    async def list_plans(self) -> List[Dict[str, Any]]:
        """List all plans."""
        response = self.client.list_plans()
        result = []
        for p in response.plans:
            plan_dict = self._plan_to_dict(p)
            # Fetch additional data from database
            try:
                details = await self._get_plan_details_from_db(p.plan_id)
                if details:
                    plan_dict.update(details)
            except Exception as e:
                logger.error(f"Failed to fetch plan details for {p.plan_id}: {e}")
            result.append(plan_dict)
        return result
    
    async def get_plan(self, plan_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific plan with tasks."""
        try:
            response = self.client.get_plan(plan_id)
            if response.plan and response.plan.plan_id:
                plan_dict = self._plan_to_dict(response.plan, include_tasks=True)
                # Fetch additional data from database
                try:
                    details = await self._get_plan_details_from_db(plan_id)
                    if details:
                        plan_dict.update(details)
                except Exception as e:
                    logger.error("Failed to fetch plan details for %s: %s", plan_id, e, exc_info=True)
                return plan_dict
            return None
        except Exception as e:
            logger.error("Error getting plan %s: %s", plan_id, e)
            return None

    async def list_plans_async(self) -> List[Dict[str, Any]]:
        """Async version of list_plans for WebSocket usage."""
        return self.list_plans()

    async def _fetch_robot_health(self) -> Dict[str, Dict[str, bool]]:
        """Fetch current robot health status."""
        try:
            # Use the existing robot health endpoint
            response = self.client.list_robots()
            health_map = {}

            # For each robot, we would need to check health
            # Since we don't have direct access to the health checking logic here,
            # we'll return a basic structure that the WebSocket can use
            for robot in response.robots:
                # This is a placeholder - in a real implementation you'd check actual health
                health_map[robot.robot_id] = {"reachable": True}  # Assume online for now

            return health_map
        except Exception as e:
            logger.error(f"Failed to fetch robot health: {e}")
            return {}
    
    def create_plan(
        self,
        planning_strategy: int,
        allocation_strategy: int,
        goal_ids: List[int],
        name: str,
        description: str
    ) -> Optional[Dict[str, Any]]:
        """Create a new plan."""
        logger.debug("Bridge create_plan called with planning=%s, allocation=%s, goals=%s, name=%s, desc=%s", planning_strategy, allocation_strategy, goal_ids, name, description)
        try:
            # Protobuf enum fields accept integers directly
            planning_enum = planning_strategy
            allocation_enum = allocation_strategy

            logger.debug("Converted to enums: planning=%s, allocation=%s", planning_enum, allocation_enum)
            response = self.client.create_plan(
                planning_strategy=planning_enum,
                goal_ids=goal_ids,
                allocation_strategy=allocation_enum,
                name=name,
                description=description
            )
            if response.plan:
                result = self._plan_to_dict(response.plan, include_tasks=True)
                return result
            logger.debug("No plan in response")
            return None
        except Exception as e:
            logger.exception("Exception in bridge: %s", e)
            return None
    
    def create_manual_plan(self, goal_ids: List[int] = None, name: Optional[str] = None, description: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create an empty plan shell for manual task creation.
        Uses MANUAL planning strategy and NONE allocation strategy."""
        try:
            response = self.client.create_plan(
                planning_strategy=fleet_manager_pb2.PlanningStrategy.MANUAL_PLAN,
                goal_ids=goal_ids or [],
                allocation_strategy=fleet_manager_pb2.AllocationStrategy.NONE,
                name=name,
                description=description
            )
            if response.plan:
                return self._plan_to_dict(response.plan, include_tasks=False)
            return None
        except Exception as e:
            logger.error(f"Error creating manual plan: {e}")
            return None
    
    def allocate_plan(self, plan_id: int, allocation_strategy: int) -> Dict[str, Any]:
        """Allocate robots to tasks in an existing plan."""
        try:
            response = self.client.allocate_plan(plan_id, allocation_strategy)
            if hasattr(response, 'error') and response.error:
                return {"success": False, "message": response.error}
            if response.plan:
                return {"success": True, "plan": self._plan_to_dict(response.plan, include_tasks=True)}
            return {"success": False, "message": "No plan returned"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def start_plan(self, plan_id: int) -> Dict[str, Any]:
        """Start executing a plan.

        Note: Will fail if plan is not fully allocated (all tasks must have robot_id).
        """
        try:
            response = self.client.start_plan(plan_id)
            return {"error": response.error if response.error else None}
        except Exception as e:
            return {"error": str(e)}

    async def update_plan(self, plan_id: int, **kwargs) -> Dict[str, Any]:
        """Update plan fields."""
        try:
            # This would need to be implemented in the gRPC service
            # For now, return success since we're updating via direct database access in executor
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_plan_allocation_status(self, plan_id: int) -> Dict[str, Any]:
        """Get the allocation status of a plan.

        Returns allocation status info including whether the plan is executable.
        """
        try:
            # Get the plan and its tasks
            plan = await self.get_plan(plan_id)
            if not plan:
                return {"error": f"Plan {plan_id} not found"}
            
            tasks = plan.get("tasks", [])
            total = len(tasks)
            allocated = sum(1 for t in tasks if t.get("robot_id"))
            unallocated_ids = [t["task_id"] for t in tasks if not t.get("robot_id")]
            
            if total == 0:
                status = "empty"
            elif allocated == 0:
                status = "unallocated"
            elif allocated < total:
                status = "partially_allocated"
            else:
                status = "fully_allocated"
            
            return {
                "plan_id": plan_id,
                "status": status,
                "total_tasks": total,
                "allocated_tasks": allocated,
                "unallocated_task_ids": unallocated_ids,
                "is_executable": status == "fully_allocated" and total > 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_plan(self, plan_id: int) -> Dict[str, Any]:
        """Delete a plan."""
        try:
            response = self.client.delete_plan(plan_id)
            return {"success": bool(response.plan), "message": response.error or "Deleted"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    # =========================================================================
    # Task Operations
    # =========================================================================
    
    def list_tasks(
        self,
        plan_ids: Optional[List[int]] = None,
        goal_ids: Optional[List[int]] = None,
        robot_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        response = self.client.list_tasks(
            plan_ids=plan_ids,
            goal_ids=goal_ids,
            robot_ids=robot_ids
        )
        return [self._task_to_dict(t) for t in response.tasks]
    
    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific task."""
        try:
            response = self.client.get_task(task_id)
            if response.task and response.task.task_id:
                return self._task_to_dict(response.task)
            return None
        except Exception as e:
            logger.error("Error getting task %s: %s", task_id, e)
            return None
    
    def create_task(
        self,
        description: str,
        goal_id: int,
        plan_id: int,
        robot_id: Optional[str] = None,
        robot_type: Optional[str] = None,
        dependency_task_ids: Optional[List[int]] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a new task."""
        try:
            response = self.client.create_task(
                description=description,
                robot_id=robot_id,
                robot_type=robot_type,
                goal_id=goal_id,
                plan_id=plan_id,
                dependency_task_ids=dependency_task_ids or []
            )
            if response.task:
                return self._task_to_dict(response.task)
            logger.warning("create_task returned no task for plan %s", plan_id)
            return None
        except Exception as e:
            logger.error("Error creating task for plan %s: %s", plan_id, e)
            return None

    def update_task(
        self,
        task_id: int,
        description: Optional[str] = None,
        goal_id: Optional[int] = None,
        robot_id: Optional[str] = None,
        dependency_task_ids: Optional[List[int]] = None,
        update_dependency_task_ids: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Update fields on a task.

        robot_id semantics:
        - robot_id is None: do not modify assignment
        - robot_id is "": clear assignment
        - robot_id is non-empty: set assignment
        """
        try:
            response = self.client.update_task(
                task_id=task_id,
                description=description,
                goal_id=goal_id,
                robot_id=robot_id,
                dependency_task_ids=dependency_task_ids or [],
                update_dependency_task_ids=update_dependency_task_ids,
            )
            if response.task and response.task.task_id:
                return self._task_to_dict(response.task)
            return None
        except Exception as e:
            logger.error("Error updating task %s: %s", task_id, e)
            return None

    def delete_task(self, task_id: int) -> Dict[str, Any]:
        """Delete a task (server will unlink dependents automatically)."""
        try:
            response = self.client.delete_task(task_id)
            return {
                "success": bool(getattr(response, "success", False)),
                "deleted_task_id": int(getattr(response, "deleted_task_id", 0) or 0),
                "updated_task_ids": list(getattr(response, "updated_task_ids", [])),
                "error": getattr(response, "error", "") or "",
            }
        except Exception as e:
            return {"success": False, "deleted_task_id": task_id, "updated_task_ids": [], "error": str(e)}
    
    # =========================================================================
    # World Statement Operations
    # =========================================================================
    
    def list_world_statements(self) -> List[Dict[str, Any]]:
        """List all world statements."""
        ws_list = self.client.list_world_statements()
        return [self._world_statement_to_dict(ws) for ws in ws_list]
    
    def add_world_statement(self, statement: str) -> Optional[Dict[str, Any]]:
        """Add a new world statement."""
        try:
            ws = self.client.add_world_statement(statement)
            if ws:
                return self._world_statement_to_dict(ws)
            return None
        except Exception as e:
            logger.error("Error adding world statement: %s", e)
            return None
    
    def delete_world_statement(self, statement_id: str) -> bool:
        """Delete a world statement."""
        try:
            return self.client.delete_world_statement(statement_id)
        except Exception as e:
            logger.error("Error deleting world statement %s: %s", statement_id, e)
            return False
