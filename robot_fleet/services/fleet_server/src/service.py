import grpc
from concurrent import futures
from typing import Dict, Optional
import asyncio
import os
import logging
from google.protobuf import timestamp_pb2
from packages.proto import fleet_manager_pb2
from packages.proto import fleet_manager_pb2_grpc
from packages.fleet_sdk.src.instance_registry import RobotInstanceRegistry
from packages.fleet_sdk.src.models import Base
from packages.config import DATABASE_URL, GRPC_SERVER_PORT
from datetime import datetime, timedelta
from sqlalchemy import text
from .events import emit_task_changed, emit_plan_changed, emit_robot_changed

# Simple class to store task server connection information
class TaskServerClient:
    """Simple class to store task server connection information"""
    def __init__(self, task_server_host: str, task_server_port: int):
        self.task_server_host = task_server_host
        self.task_server_port = task_server_port

# Configure root logger if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.WARNING,  # Default to WARNING level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

logger = logging.getLogger(__name__)

def configure_logging(verbose: bool = False, sql_debug: bool = False):
    """Configure logging based on verbose and sql_debug flags
    
    Args:
        verbose: If True, enables detailed application logging
        sql_debug: If True, enables SQL debug logging
    """
    # Configure application logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    
    # Configure SQL logging if requested
    if sql_debug:
        logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
    else:
        # Keep SQL logging at WARNING to avoid noise
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        
    # Configure other modules
    task_server_logger = logging.getLogger('task_server')
    task_server_logger.setLevel(log_level)

class FleetManagerService(fleet_manager_pb2_grpc.FleetManagerServicer):
    """gRPC server for robot fleet management"""

    def __init__(self, db_url: Optional[str] = None):
        """Initialize the service
        
        Args:
            db_url: PostgreSQL connection URL. If None, uses default connection.
        """
        self.registry = RobotInstanceRegistry(db_url=db_url)
        self._streams_contexts = {}  # Store stream contexts by robot_id
        self._session_contexts = {}  # Store session contexts by robot_id
        
    async def initialize(self):
        """Initialize the service by setting up the database"""
        await self.registry.initialize()

    async def cleanup(self):
        """Dispose the async engine and release DB connections."""
        await self.registry.engine.dispose()

    async def _mark_plan_manual(self, plan_id: int) -> None:
        """Mark a plan as manually defined/allocated based on its current tasks.

        - planning_strategy is always set to MANUAL_PLAN
        - allocation_strategy is MANUAL_ALLOCATION if ANY task has a robot_id, else NONE
        """
        tasks = await self.registry.list_tasks(plan_ids=[plan_id])
        has_any_robot = any(getattr(t, "robot_id", "") for t in tasks)
        goal_ids = sorted({int(getattr(t, "goal_id", 0)) for t in tasks if int(getattr(t, "goal_id", 0) or 0) > 0})

        allocation_strategy = (
            fleet_manager_pb2.AllocationStrategy.MANUAL_ALLOCATION
            if has_any_robot
            else fleet_manager_pb2.AllocationStrategy.NONE
        )

        updated = await self.registry.update_plan(
            plan_id=plan_id,
            goal_ids=goal_ids,
            planning_strategy=int(fleet_manager_pb2.PlanningStrategy.MANUAL_PLAN),
            allocation_strategy=int(allocation_strategy),
        )
        if not updated:
            raise RuntimeError(f"Failed to update plan strategies for plan_id={plan_id}")

    async def RegisterRobot(self, request, context):
        """Register a robot with the fleet manager"""
        logger.info(f"Registering robot: {request.robot_id}")
        try:
            # Use the instance registry to register the robot
            robot = await self.registry.register_robot(
                robot_id=request.robot_id,
                robot_type=request.robot_type,
                description=request.description,
                capabilities=list(request.capabilities),
                container_info=request.container if hasattr(request, 'container') else None,
                deployment_info=request.deployment if hasattr(request, 'deployment') else None,
                task_server_info=request.task_server_info if hasattr(request, 'task_server_info') else None
            )
            
            if not robot:
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details(f"Robot with ID {request.robot_id} already exists")
                return fleet_manager_pb2.RegisterRobotResponse(
                    success=False,
                    message=f"Robot with ID {request.robot_id} already exists"
                )
            
            logger.info(f"Successfully registered robot: {request.robot_id}")
            emit_robot_changed(request.robot_id, action="registered")
            return fleet_manager_pb2.RegisterRobotResponse(
                success=True,
                message=f"Successfully registered robot: {request.robot_id}",
                robot=robot
            )
        
        except Exception as e:
            logger.error(f"Error registering robot: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to register robot: {str(e)}")
            return fleet_manager_pb2.RegisterRobotResponse(
                success=False,
                message=f"Error: {str(e)}"
            )

    async def DeployRobot(self, request, context):
        """Deploy a robot container (not yet implemented)."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Container deployment not yet implemented")
        return fleet_manager_pb2.DeployRobotResponse(
            success=False,
            message="Container deployment not yet implemented"
        )

    async def UndeployRobot(self, request, context):
        """Undeploy a robot container (not yet implemented)."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Container undeployment not yet implemented")
        return fleet_manager_pb2.UndeployRobotResponse(
            success=False,
            message="Container undeployment not yet implemented"
        )

    async def UnregisterRobot(self, request, context):
        """Unregister a robot from the fleet manager"""
        logger.info(f"Unregistering robot: {request.robot_id}")
        try:
            # Check if robot exists first
            robot = await self.registry.get_robot(request.robot_id)
            if not robot:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Robot with ID {request.robot_id} not found")
                return fleet_manager_pb2.UnregisterRobotResponse(
                    success=False,
                    message=f"Robot with ID {request.robot_id} not found"
                )
            
            # Check if robot is deployed (has container)
            if robot.HasField('container') and robot.container.container_id:
                # Try to undeploy the robot first
                try:
                    await self.container_manager.stop_robot(
                        robot_id=request.robot_id,
                        host=robot.deployment.docker_host,
                        docker_port=robot.deployment.docker_port
                    )
                except Exception as e:
                    logger.warning(f"Failed to undeploy robot {request.robot_id} before unregistering: {str(e)}")
            
            # Delete the robot from the registry
            success = await self.registry.delete_robot(request.robot_id)
            
            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to unregister robot {request.robot_id}")
                return fleet_manager_pb2.UnregisterRobotResponse(
                    success=False,
                    message=f"Failed to unregister robot {request.robot_id}"
                )
            
            
            logger.info(f"Successfully unregistered robot: {request.robot_id}")
            emit_robot_changed(request.robot_id, action="unregistered")
            return fleet_manager_pb2.UnregisterRobotResponse(
                success=True,
                message=f"Successfully unregistered robot: {request.robot_id}"
            )
            
        except Exception as e:
            logger.error(f"Error unregistering robot: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to unregister robot: {str(e)}")
            return fleet_manager_pb2.UnregisterRobotResponse(
                success=False,
                message=f"Error: {str(e)}"
            )

    async def ListRobots(self, request, context):
        """List all robots in the fleet manager"""
        logger.info("Listing robots")
        try:
            # List all robots from the registry
            robots = await self.registry.list_robots()
            
            # Apply filter if specified
            filtered_robots = []
            for robot in robots:
                if request.filter == fleet_manager_pb2.ListRobotsRequest.Filter.ALL:
                    filtered_robots.append(robot)
                elif request.filter == fleet_manager_pb2.ListRobotsRequest.Filter.DEPLOYED:
                    if robot.HasField('container') and robot.container.container_id:
                        filtered_robots.append(robot)
                elif request.filter == fleet_manager_pb2.ListRobotsRequest.Filter.REGISTERED:
                    if (robot.status.state == fleet_manager_pb2.RobotStatus.State.REGISTERED or 
                        robot.status.state == fleet_manager_pb2.RobotStatus.State.RUNNING):
                        filtered_robots.append(robot)
            
            logger.info(f"Found {len(filtered_robots)} robots")
            return fleet_manager_pb2.ListRobotsResponse(
                robots=filtered_robots
            )
            
        except Exception as e:
            logger.error(f"Error listing robots: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to list robots: {str(e)}")
            return fleet_manager_pb2.ListRobotsResponse()

    async def GetRobot(self, request, context):
        """Get a specific robot by ID."""
        logger.info(f"Getting robot: {request.robot_id}")
        try:
            robot = await self.registry.get_robot(request.robot_id)
            if not robot:
                return fleet_manager_pb2.GetRobotResponse(
                    error=f"Robot with ID {request.robot_id} not found"
                )
            return fleet_manager_pb2.GetRobotResponse(robot=robot, error="")
        except Exception as e:
            logger.error(f"Error getting robot: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get robot: {str(e)}")
            return fleet_manager_pb2.GetRobotResponse(error=f"Failed to get robot: {str(e)}")

    async def GetRobotStatus(self, request, context):
        """Get the status of a robot"""
        logger.info(f"Getting status for robot: {request.robot_id}")
        try:
            # Get robot from registry
            robot = await self.registry.get_robot(request.robot_id)
            if not robot:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Robot with ID {request.robot_id} not found")
                return fleet_manager_pb2.RobotStatus(
                    state=fleet_manager_pb2.RobotStatus.State.UNKNOWN,
                    message=f"Robot with ID {request.robot_id} not found"
                )
            
            # Determine status based on container info
            status_state = fleet_manager_pb2.RobotStatus.State.REGISTERED
            status_message = f"Robot {request.robot_id} is registered"
            
            if robot.HasField('container') and robot.container.container_id:
                status_state = fleet_manager_pb2.RobotStatus.State.RUNNING
                status_message = f"Robot {request.robot_id} container is running"
            
            # Create and return RobotStatus
            return fleet_manager_pb2.RobotStatus(
                state=status_state,
                message=status_message
            )
            
        except Exception as e:
            logger.error(f"Error getting robot status: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get robot status: {str(e)}")
            return fleet_manager_pb2.RobotStatus(
                state=fleet_manager_pb2.RobotStatus.State.UNKNOWN,
                message=f"Error: {str(e)}"
            )

    async def CreateGoal(self, request, context):
        """Create a new goal"""
        logger.info(f"Creating new goal: {request.description}")
        try:
            # Create goal using registry
            goal = await self.registry.create_goal(
                description=request.description
            )
            
            if not goal:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Failed to create goal")
                return fleet_manager_pb2.CreateGoalResponse(
                    error="Failed to create goal"
                )
            
            logger.info(f"Successfully created goal: {goal.goal_id}")
            return fleet_manager_pb2.CreateGoalResponse(
                goal=goal
            )
            
        except Exception as e:
            logger.error(f"Error creating goal: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to create goal: {str(e)}")
            return fleet_manager_pb2.CreateGoalResponse(
                error=f"Error: {str(e)}"
            )

    async def GetGoal(self, request, context):
        """Get a specific goal by ID"""
        logger.info(f"Getting goal: {request.goal_id}")
        try:
            # Get goal from registry
            goal = await self.registry.get_goal(request.goal_id)
            
            if not goal:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Goal with ID {request.goal_id} not found")
                return fleet_manager_pb2.GetGoalResponse(
                    error=f"Goal with ID {request.goal_id} not found"
                )
            
            logger.info(f"Successfully retrieved goal: {request.goal_id}")
            return fleet_manager_pb2.GetGoalResponse(
                goal=goal
            )
            
        except Exception as e:
            logger.error(f"Error getting goal: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get goal: {str(e)}")
            return fleet_manager_pb2.GetGoalResponse(
                error=f"Error: {str(e)}"
            )

    async def ListGoals(self, request, context):
        """List all goals"""
        logger.info("Listing goals")
        try:
            # Get goals from registry
            goals = await self.registry.list_goals()
            
            logger.info(f"Found {len(goals)} goals")
            return fleet_manager_pb2.ListGoalsResponse(
                goals=goals
            )
            
        except Exception as e:
            logger.error(f"Error listing goals: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to list goals: {str(e)}")
            return fleet_manager_pb2.ListGoalsResponse(
                error=f"Error: {str(e)}"
            )

    async def DeleteGoal(self, request, context):
        """Delete a goal by ID"""
        logger.info(f"Deleting goal: {request.goal_id}")
        try:
            # Get goal first to return it in the response
            goal = await self.registry.get_goal(request.goal_id)
            
            if not goal:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Goal with ID {request.goal_id} not found")
                return fleet_manager_pb2.DeleteGoalResponse(
                    error=f"Goal with ID {request.goal_id} not found"
                )
            
            # Delete goal
            success = await self.registry.delete_goal(request.goal_id)
            
            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to delete goal {request.goal_id}")
                return fleet_manager_pb2.DeleteGoalResponse(
                    error=f"Failed to delete goal {request.goal_id}"
                )
            
            logger.info(f"Successfully deleted goal: {request.goal_id}")
            return fleet_manager_pb2.DeleteGoalResponse(
                goal=goal
            )
            
        except Exception as e:
            logger.error(f"Error deleting goal: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to delete goal: {str(e)}")
            return fleet_manager_pb2.DeleteGoalResponse(
                error=f"Error: {str(e)}"
            )

    async def CreateTask(self, request, context):
        """Create a new task"""
        logger.info(f"Creating new task: {request.description}")
        try:
            logger.info(f"Creating new task DEBUG: {request}")
            robot_type = request.robot_type if request.robot_type else None
            if request.robot_id and not robot_type:
                robot = await self.registry.get_robot(request.robot_id)
                if robot:
                    robot_type = robot.robot_type
            # Create task using registry
            task = await self.registry.create_task(
                description=request.description,
                robot_id=request.robot_id if request.robot_id else None,
                goal_id=request.goal_id if request.goal_id else None,
                plan_id=request.plan_id if request.plan_id else None,
                robot_type=robot_type,
                dependency_task_ids=list(request.dependency_task_ids) if request.dependency_task_ids else None
            )
            
            if not task:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Failed to create task")
                return fleet_manager_pb2.CreateTaskResponse(
                    error="Failed to create task"
                )

            # Any task mutation should mark the plan as manual (strategy recomputation happens server-side)
            if request.plan_id:
                await self._mark_plan_manual(int(request.plan_id))
            
            logger.info(f"Successfully created task: {task.task_id}")
            emit_task_changed(task.task_id, plan_id=request.plan_id or None, status="pending")
            return fleet_manager_pb2.CreateTaskResponse(
                task=task
            )
            
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to create task: {str(e)}")
            return fleet_manager_pb2.CreateTaskResponse(
                error=f"Error: {str(e)}"
            )

    async def UpdateTask(self, request, context):
        """Update fields of an existing task."""
        logger.info(f"Updating task: {request.task_id}")
        try:
            existing = await self.registry.get_task(request.task_id)
            if not existing:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Task with ID {request.task_id} not found")
                return fleet_manager_pb2.UpdateTaskResponse(
                    error=f"Task with ID {request.task_id} not found"
                )

            update_kwargs: Dict[str, object] = {}

            if hasattr(request, "HasField") and request.HasField("description"):
                update_kwargs["description"] = request.description

            if hasattr(request, "HasField") and request.HasField("goal_id"):
                update_kwargs["goal_id"] = int(request.goal_id)

            if hasattr(request, "HasField") and request.HasField("robot_id"):
                # If present and empty string, clear assignment
                if request.robot_id == "":
                    update_kwargs["robot_id"] = None
                    update_kwargs["robot_type"] = None
                else:
                    update_kwargs["robot_id"] = request.robot_id
                    # Infer robot_type from robot_id (so DAG/UI can render it)
                    robot = await self.registry.get_robot(request.robot_id)
                    if robot:
                        update_kwargs["robot_type"] = robot.robot_type

            if request.update_dependency_task_ids:
                update_kwargs["dependency_task_ids"] = list(request.dependency_task_ids)

            updated_task = await self.registry.update_task(request.task_id, **update_kwargs)
            if not updated_task:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to update task with ID {request.task_id}")
                return fleet_manager_pb2.UpdateTaskResponse(
                    error=f"Failed to update task with ID {request.task_id}"
                )

            plan_id = int(getattr(existing, "plan_id", 0) or 0)
            if plan_id:
                await self._mark_plan_manual(plan_id)

            logger.info(f"Successfully updated task: {request.task_id}")
            emit_task_changed(request.task_id, plan_id=plan_id or None, status="updated")
            return fleet_manager_pb2.UpdateTaskResponse(task=updated_task)

        except Exception as e:
            logger.error(f"Error updating task: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to update task: {str(e)}")
            return fleet_manager_pb2.UpdateTaskResponse(
                error=f"Error: {str(e)}"
            )

    async def GetTask(self, request, context):
        """Get a specific task by ID"""
        logger.info(f"Getting task: {request.task_id}")
        try:
            # Get task from registry
            task = await self.registry.get_task(request.task_id)
            
            if not task:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Task with ID {request.task_id} not found")
                return fleet_manager_pb2.GetTaskResponse(
                    error=f"Task with ID {request.task_id} not found"
                )
            
            logger.info(f"Successfully retrieved task: {request.task_id}")
            return fleet_manager_pb2.GetTaskResponse(
                task=task
            )
            
        except Exception as e:
            logger.error(f"Error getting task: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get task: {str(e)}")
            return fleet_manager_pb2.GetTaskResponse(
                error=f"Error: {str(e)}"
            )

    async def ListTasks(self, request, context):
        """List tasks with optional filtering"""
        logger.info(f"Listing tasks with filters: plan_ids={request.plan_ids}, goal_ids={request.goal_ids}, robot_ids={request.robot_ids}")
        
        try:
            tasks = await self.registry.list_tasks(
                plan_ids=list(request.plan_ids) if request.plan_ids else None,
                goal_ids=list(request.goal_ids) if request.goal_ids else None,
                robot_ids=list(request.robot_ids) if request.robot_ids else None
            )
            
            return fleet_manager_pb2.ListTasksResponse(
                tasks=tasks
            )
        except Exception as e:
            logger.error(f"Error listing tasks: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to list tasks: {str(e)}")
            return fleet_manager_pb2.ListTasksResponse(
                tasks=[],
                error=f"Error: {str(e)}"
            )
            
    async def DeleteTask(self, request, context):
        """Delete a task by ID"""
        logger.info(f"Deleting task: {request.task_id}")
        
        try:
            success, updated_task_ids, plan_id = await self.registry.delete_task(request.task_id)
            if not success:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Task with ID {request.task_id} not found")
                return fleet_manager_pb2.DeleteTaskResponse(
                    success=False,
                    deleted_task_id=request.task_id,
                    updated_task_ids=[],
                    error=f"Task with ID {request.task_id} not found",
                )

            # Any task mutation should mark the plan as manual (strategy recomputation happens server-side)
            if plan_id is not None:
                await self._mark_plan_manual(plan_id)
            
            logger.info(f"Successfully deleted task: {request.task_id}, updated dependents: {updated_task_ids}")
            emit_task_changed(request.task_id, plan_id=plan_id, status="deleted")
            return fleet_manager_pb2.DeleteTaskResponse(
                success=True,
                deleted_task_id=request.task_id,
                updated_task_ids=updated_task_ids,
                error="",
            )
            
        except Exception as e:
            logger.error(f"Error deleting task: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to delete task: {str(e)}")
            return fleet_manager_pb2.DeleteTaskResponse(
                success=False,
                deleted_task_id=request.task_id,
                updated_task_ids=[],
                error=f"Error: {str(e)}",
            )
            
    async def CreatePlan(self, request, context):
        """Create a new plan.
        
        If planning_strategy is MANUAL, creates an empty plan shell (no auto-planning).
        If allocation_strategy is NONE, skips robot allocation.
        """
        try:
            # Get the requested planning strategy and allocation strategy
            # These are already enum objects from protobuf deserialization
            planning_strategy = request.planning_strategy
            allocation_strategy = request.allocation_strategy
            goal_ids = list(request.goal_ids) if request.goal_ids else []
            name = request.name
            description = request.description

            # MANUAL strategy: create empty plan shell for user-defined tasks
            if planning_strategy == fleet_manager_pb2.PlanningStrategy.MANUAL_PLAN:
                logger.info("Creating manual plan shell (no auto-planning)")
                plan = await self.registry.create_plan(
                    planning_strategy=planning_strategy,
                    allocation_strategy=allocation_strategy,
                    goal_ids=goal_ids,
                    task_ids=[],
                    name=name,
                    description=description
                )
                emit_plan_changed(plan.plan_id, status="created")
                return fleet_manager_pb2.CreatePlanResponse(plan=plan)
            
            # Auto-planning requires goals
            if not goal_ids:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("At least one goal ID must be provided for auto-planning")
                return fleet_manager_pb2.CreatePlanResponse(
                    error="At least one goal ID must be provided for auto-planning"
                )
            
            # For testing environments, create a plan directly without using the planner
            test_mode = os.getenv("TESTING", "false").lower() in ("true", "1", "yes")
            if test_mode:
                logger.info("Running in test mode, bypassing planner")
                plan = await self.registry.create_plan(
                    planning_strategy=planning_strategy,
                    allocation_strategy=allocation_strategy,
                    goal_ids=goal_ids,
                    task_ids=[],
                    name=name,
                    description=description
                )
                emit_plan_changed(plan.plan_id, status="created")
                return fleet_manager_pb2.CreatePlanResponse(plan=plan)
            
            # Normal flow using the planner
            from .planners.base import get_planner
            from .allocators.base import get_allocator
            planner = get_planner(planning_strategy, registry=self.registry)
            
            # Generate plan using the planner
            try:
                logger.info(f"Generating plan for goals {goal_ids} using {planning_strategy} planner")
                plan_json = await planner.plan(goal_ids)

                # Debug: Check what the planner has stored
                logger.debug("PLANNER DEBUG: planning_prompts = %s", getattr(planner, 'planning_prompts', 'NOT SET'))
                logger.debug("PLANNER DEBUG: planning_artifacts = %s", getattr(planner, 'planning_artifacts', 'NOT SET'))
                logger.debug("PLANNER DEBUG: server_logs = %s", getattr(planner, 'server_logs', 'NOT SET'))

                # Collect server logs from planner
                server_logs = getattr(planner, 'server_logs', [])
                if not server_logs:
                    server_logs = [f"Planning completed successfully for goals {goal_ids} using {planning_strategy} strategy"]

                plan_id = await planner.save_plan_to_db(plan_json, planning_strategy, allocation_strategy, goal_ids, name=name, description=description)
                logger.info(f"Successfully created and saved plan {plan_id}")
            except Exception as e:
                logger.error(f"Error during planning: {str(e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Planning failed: {str(e)}")
                return fleet_manager_pb2.CreatePlanResponse(
                    error=f"Planning failed: {str(e)}"
                )

            # Allocate tasks if allocation strategy is not NONE
            if allocation_strategy != fleet_manager_pb2.AllocationStrategy.NONE:
                allocator = get_allocator(allocation_strategy, registry=self.registry)
                allocation = await allocator.allocate(plan_id)
                logger.info("Task allocation complete: %s", allocation)

                # Store allocation artifacts in the plan
                await self.registry.update_plan(
                    plan_id,
                    allocation_prompts=getattr(allocator, 'allocation_prompts', {}),
                    allocation_artifacts=getattr(allocator, 'allocation_artifacts', {}),
                    server_logs=getattr(allocator, 'server_logs', [])
                )
            else:
                logger.info("Skipping allocation (strategy=NONE)")
            
            # Get the plan with all tasks
            plan = await self.registry.get_plan(plan_id)
            logger.debug("Retrieved plan: %s", plan)

            logger.info(f"Successfully created plan: {plan_id}")
            emit_plan_changed(plan_id, status="created")
            response = fleet_manager_pb2.CreatePlanResponse(plan=plan)
            logger.debug("Created response: %s", response)
            return response
            
        except Exception as e:
            logger.debug("Exception in gRPC CreatePlan: %s", e)
            import traceback
            logger.debug("Traceback: %s", traceback.format_exc())
            logger.error(f"Error creating plan: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to create plan: {str(e)}")
            return fleet_manager_pb2.CreatePlanResponse(
                error=f"Error: {str(e)}"
            )
    
    async def AllocatePlan(self, request, context):
        """Allocate robots to tasks in an existing plan."""
        logger.info(f"Allocating plan {request.plan_id} with strategy: {request.allocation_strategy}")
        try:
            plan_id = request.plan_id
            allocation_strategy = request.allocation_strategy
            
            # Check plan exists
            plan = await self.registry.get_plan(plan_id)
            if not plan:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Plan {plan_id} not found")
                return fleet_manager_pb2.AllocatePlanResponse(
                    error=f"Plan {plan_id} not found"
                )
            
            # Check there are tasks to allocate
            if not plan.task_ids:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(f"Plan {plan_id} has no tasks to allocate")
                return fleet_manager_pb2.AllocatePlanResponse(
                    error=f"Plan {plan_id} has no tasks to allocate"
                )
            
            # Run allocation
            from .allocators.base import get_allocator
            allocator = get_allocator(allocation_strategy, registry=self.registry)
            allocation = await allocator.allocate(plan_id)
            logger.info(f"Allocation complete for plan {plan_id}: {allocation}")

            # Update plan's allocation strategy and allocation data in DB
            await self.registry.update_plan(
                plan_id=plan_id,
                allocation_strategy=allocation_strategy,
                allocation_prompts=getattr(allocator, 'allocation_prompts', None),
                allocation_artifacts=getattr(allocator, 'allocation_artifacts', None)
            )
            
            # Get updated plan
            updated_plan = await self.registry.get_plan(plan_id)
            
            return fleet_manager_pb2.AllocatePlanResponse(plan=updated_plan)
            
        except Exception as e:
            logger.error(f"Error allocating plan: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to allocate plan: {str(e)}")
            return fleet_manager_pb2.AllocatePlanResponse(
                error=f"Error: {str(e)}"
            )

    async def GetPlan(self, request, context):
        """Get a specific plan by ID"""
        logger.info(f"Getting plan: {request.plan_id}")
        try:
            # Get plan from registry
            plan = await self.registry.get_plan(request.plan_id)
            
            if not plan:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Plan with ID {request.plan_id} not found")
                return fleet_manager_pb2.GetPlanResponse(
                    error=f"Plan with ID {request.plan_id} not found"
                )
            
            logger.info(f"Successfully retrieved plan: {request.plan_id}")
            return fleet_manager_pb2.GetPlanResponse(
                plan=plan
            )
            
        except Exception as e:
            logger.error(f"Error getting plan: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get plan: {str(e)}")
            return fleet_manager_pb2.GetPlanResponse(
                error=f"Error: {str(e)}"
            )

    async def ListPlans(self, request, context):
        """List all plans"""
        logger.info("Listing plans")
        try:
            # Get plans from registry
            plans = await self.registry.list_plans()
            
            logger.info(f"Found {len(plans)} plans")
            return fleet_manager_pb2.ListPlansResponse(
                plans=plans
            )
            
        except Exception as e:
            logger.error(f"Error listing plans: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to list plans: {str(e)}")
            return fleet_manager_pb2.ListPlansResponse(
                error=f"Error: {str(e)}"
            )

    async def DeletePlan(self, request, context):
        """Delete a plan by ID"""
        logger.info(f"Deleting plan: {request.plan_id}")
        try:
            # Get plan first to return it in the response
            plan = await self.registry.get_plan(request.plan_id)
            
            if not plan:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Plan with ID {request.plan_id} not found")
                return fleet_manager_pb2.DeletePlanResponse(
                    error=f"Plan with ID {request.plan_id} not found"
                )
            
            # Delete plan
            success = await self.registry.delete_plan(request.plan_id)
            
            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to delete plan {request.plan_id}")
                return fleet_manager_pb2.DeletePlanResponse(
                    error=f"Failed to delete plan {request.plan_id}"
                )
            
            logger.info(f"Successfully deleted plan: {request.plan_id}")
            emit_plan_changed(request.plan_id, status="deleted")
            return fleet_manager_pb2.DeletePlanResponse(
                plan=plan
            )
            
        except Exception as e:
            logger.error(f"Error deleting plan: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to delete plan: {str(e)}")
            return fleet_manager_pb2.DeletePlanResponse(
                error=f"Error: {str(e)}"
            )

    async def AddWorldStatement(self, request, context):
        """Add a new world statement."""
        logger.info(f"Received AddWorldStatement request: Statement='{request.statement[:50]}...'")
        try:
            ws_proto = await self.registry.add_world_statement(request.statement)
            return fleet_manager_pb2.AddWorldStatementResponse(world_statement=ws_proto, error="")
        except Exception as e:
            logger.error(f"Error adding world statement: {e}", exc_info=True)
            return fleet_manager_pb2.AddWorldStatementResponse(world_statement=None, error=f"Failed to add world statement: {e}")

    async def GetWorldStatement(self, request, context):
        """Get a world statement by ID."""
        logger.info(f"Received GetWorldStatement request for ID: {request.world_statement_id}")
        try:
            ws_proto = await self.registry.get_world_statement(request.world_statement_id)
            if ws_proto is None:
                # Do not set world_statement at all if not found
                return fleet_manager_pb2.GetWorldStatementResponse(error="World statement not found")
            return fleet_manager_pb2.GetWorldStatementResponse(world_statement=ws_proto, error="")
        except Exception as e:
            logger.error(f"Error getting world statement: {e}", exc_info=True)
            return fleet_manager_pb2.GetWorldStatementResponse(error=f"Failed to get world statement: {e}")

    async def ListWorldStatements(self, request, context):
        """List all world statements."""
        logger.info("Received ListWorldStatements request.")
        try:
            ws_protos = await self.registry.list_world_statements()
            return fleet_manager_pb2.ListWorldStatementsResponse(world_statements=ws_protos, error="")
        except Exception as e:
            logger.error(f"Error listing world statements: {e}", exc_info=True)
            return fleet_manager_pb2.ListWorldStatementsResponse(world_statements=[], error=f"Failed to list world statements: {e}")

    async def DeleteWorldStatement(self, request, context):
        """Delete a world statement by ID."""
        logger.info(f"Received DeleteWorldStatement request for ID: {request.world_statement_id}")
        try:
            success = await self.registry.delete_world_statement(request.world_statement_id)
            if not success:
                logger.warning(f"Failed to delete world statement ID={request.world_statement_id} (likely not found or invalid ID).")
                return fleet_manager_pb2.DeleteWorldStatementResponse(success=False, error="World statement not found")
            return fleet_manager_pb2.DeleteWorldStatementResponse(success=True, error="")
        except ValueError:
            logger.error(f"Invalid world statement ID format for deletion: {request.world_statement_id}")
            return fleet_manager_pb2.DeleteWorldStatementResponse(success=False, error="Invalid world statement ID format")
        except Exception as e:
            logger.error(f"Error deleting world statement: {e}", exc_info=True)
            return fleet_manager_pb2.DeleteWorldStatementResponse(success=False, error=f"Failed to delete world statement: {e}")

    async def StartPlan(
        self, request: fleet_manager_pb2.StartPlanRequest, context
    ) -> fleet_manager_pb2.StartPlanResponse:
        """Start executing a plan.
        
        IMPORTANT: Only fully allocated plans can be executed. A plan is executable
        if and only if ALL tasks have a robot_id assigned. This is the invariant
        that distinguishes an ExecutablePlan from an UnallocatedPlan.
        """
        from .executor.executor import Executor
        plan_id = request.plan_id
        logger.info(f"Received StartPlan request for plan_id: {plan_id}")
        
        try:
            # Step 1: Verify plan exists
            plan = await self.registry.get_plan(plan_id)
            if not plan:
                error_msg = f"Plan {plan_id} not found"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(error_msg)
                return fleet_manager_pb2.StartPlanResponse(error=error_msg)
            
            # Step 2: Check allocation status - CRITICAL VALIDATION
            allocation_status = await self.registry.get_plan_allocation_status(plan_id)
            
            if not allocation_status['is_executable']:
                status = allocation_status['status']
                total = allocation_status['total_tasks']
                allocated = allocation_status['allocated_tasks']
                unallocated_ids = allocation_status['unallocated_task_ids']
                
                if status == 'empty':
                    error_msg = f"Plan {plan_id} has no tasks. Cannot execute an empty plan."
                elif status == 'unallocated':
                    error_msg = (
                        f"Plan {plan_id} is UNALLOCATED. "
                        f"All {total} tasks need robot assignments before execution. "
                        f"Run allocation first using the 'allocate' command or API."
                    )
                elif status == 'partially_allocated':
                    error_msg = (
                        f"Plan {plan_id} is PARTIALLY ALLOCATED ({allocated}/{total} tasks assigned). "
                        f"Tasks without robots: {unallocated_ids}. "
                        f"All tasks must have robot assignments before execution."
                    )
                else:
                    error_msg = f"Plan {plan_id} is not executable (status: {status})"
                
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(error_msg)
                return fleet_manager_pb2.StartPlanResponse(error=error_msg)
            
            # Step 3: Plan is executable - proceed with execution
            logger.info(f"Plan {plan_id} is fully allocated ({allocation_status['total_tasks']} tasks). Starting execution...")

            # Start execution asynchronously in the background
            # The executor will handle updating plan status to executing and completed
            executor = Executor(plan_id=plan_id, registry=self.registry)
            asyncio.create_task(executor.execute())

            # Return immediately - execution will continue in background
            return fleet_manager_pb2.StartPlanResponse(error="")
            
        except Exception as e:
            error_msg = f"Failed to start plan {plan_id}: {e}"
            logger.error(error_msg, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return fleet_manager_pb2.StartPlanResponse(error=error_msg)


async def serve(port: int = GRPC_SERVER_PORT, db_url: str = None, reset_db: bool = False, verbose: bool = False, sql_debug: bool = False):
    """Start the gRPC server
    
    Args:
        port: Port to listen on
        db_url: PostgreSQL connection URL. If None, uses default connection.
        reset_db: If True, drops and recreates all database tables on startup
        verbose: If True, enables detailed application logging
        sql_debug: If True, enables SQL debug logging
    """
    # Configure logging based on verbose and sql_debug flags
    configure_logging(verbose, sql_debug)

    if not db_url:
        db_url = DATABASE_URL
    logger.info("Using database: %s", db_url)

    # Create the service first
    logger.info("Creating service...")
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    service = FleetManagerService(db_url)
    
    # Initialize the database with reset flag
    if reset_db:
        logger.info("Resetting database tables...")
        async with service.registry.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("All tables dropped successfully")
    
    # Initialize/create tables using the service's registry
    logger.info("Creating database tables...")
    await service.initialize()
    logger.info("Database initialization complete")
    
    # Print all available SQL tables
    async with service.registry.engine.connect() as conn:
        result = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
        tables = result.fetchall()
        logger.info("Available SQL tables:")
        for table in tables:
            logger.info("- %s", table[0])
          
    fleet_manager_pb2_grpc.add_FleetManagerServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    logger.info("Fleet Manager Server is now serving on port %s", port)
    
    try:
        await server.start()
        await server.wait_for_termination()
    finally:
        # Clean up resources
        await service.cleanup() 