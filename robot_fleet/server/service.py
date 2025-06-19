import grpc
from concurrent import futures
from typing import Dict, Optional
import asyncio
import os
import logging
from google.protobuf import timestamp_pb2
from ..proto import fleet_manager_pb2
from ..proto import fleet_manager_pb2_grpc
from ..robots.registry.instance_registry import RobotInstanceRegistry
from ..robots.registry.models import Base
from ..robots.containers.manager import ContainerManager
from datetime import datetime, timedelta
from sqlalchemy import text

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
        self.container_manager = ContainerManager()
        self._streams_contexts = {}  # Store stream contexts by robot_id
        self._session_contexts = {}  # Store session contexts by robot_id
        
    async def initialize(self):
        """Initialize the service by setting up the database"""
        await self.registry.initialize()

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
        """Deploy a robot container"""
        logger.info(f"Deploying robot: {request.robot_id}")
        try:
            # Get robot from registry
            robot = await self.registry.get_robot(request.robot_id)
            if not robot:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Robot with ID {request.robot_id} not found")
                return fleet_manager_pb2.DeployRobotResponse(
                    success=False,
                    message=f"Robot with ID {request.robot_id} not found"
                )

            # Use integer enum for status checks and updates
            running_state = fleet_manager_pb2.RobotStatus.State.RUNNING
            deploying_state = fleet_manager_pb2.RobotStatus.State.DEPLOYING

            # If already running or deploying, return success (idempotent)
            if robot.status.state == running_state or robot.status.state == deploying_state:
                logger.info(f"Robot {request.robot_id} is already running or deploying.")
                return fleet_manager_pb2.DeployRobotResponse(
                    success=True,
                    message=f"Robot {request.robot_id} is already running or deploying.",
                    container=robot.container
                )

            # Set status to DEPLOYING before actual deployment
            await self.registry.update_robot_status(request.robot_id, deploying_state)

            # Extract deployment and container info
            deployment_info = {}
            container_info = {}
            task_server_info = {}
            
            if robot.HasField('deployment'):
                deployment_info = {
                    'docker_host': robot.deployment.docker_host,
                    'docker_port': robot.deployment.docker_port
                }
            
            if robot.HasField('container'):
                container_info = {
                    'image': robot.container.image,
                    'environment': dict(robot.container.environment),
                }
            
            if robot.HasField('task_server_info'):
                task_server_info = {
                    'host': robot.task_server_info.host,
                    'port': robot.task_server_info.port,
                }
            
            # Add required environment variables
            container_info['environment'] = container_info.get('environment', {})
            container_info['environment'].update({
                "ROBOT_ID": robot.robot_id,
                "ROBOT_TYPE": robot.robot_type,
                "HOST": "0.0.0.0",  # Listen on all interfaces
                "PORT": str(task_server_info.get('port', 8080))
            })
            
            # Set the port in container_info
            container_info['port'] = task_server_info.get('port', 8080)
            
            # Deploy container
            try:
                container_result = await self.container_manager.deploy_robot(
                    robot_type=robot.robot_type,
                    robot_id=robot.robot_id,
                    deployment_info=deployment_info,
                    container_info=container_info
                )
            except Exception as container_error:
                logger.error(f"Container deployment error: {str(container_error)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to deploy container: {str(container_error)}")
                return fleet_manager_pb2.DeployRobotResponse(
                    success=False,
                    message=f"Failed to deploy container: {str(container_error)}"
                )
            
            if not container_result:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to deploy container for robot {request.robot_id}")
                return fleet_manager_pb2.DeployRobotResponse(
                    success=False,
                    message=f"Failed to deploy container for robot {request.robot_id}"
                )
            
            # Create ContainerInfo for response
            timestamp = timestamp_pb2.Timestamp()
            if container_result.created_at:
                try:
                    # Parse ISO format string from ContainerInfo.created_at
                    dt = datetime.fromisoformat(container_result.created_at.replace('Z', '+00:00'))
                    timestamp.FromDatetime(dt)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse timestamp: {e}")
                    timestamp.GetCurrentTime()
            else:
                timestamp.GetCurrentTime()

            container_info_proto = fleet_manager_pb2.ContainerInfo(
                container_id=container_result.container_id,
                image=container_result.image,
                host=container_result.host,
                port=container_result.port,
                environment=container_result.environment,
                created_at=timestamp
            )
            
            # Update robot container info in registry
            await self.registry.update_container_info(robot.robot_id, container_info_proto)
            
            # Update robot status
            updated_robot = await self.registry.update_robot_status(request.robot_id, running_state)
            
            logger.info(f"Successfully deployed robot: {request.robot_id}")
            return fleet_manager_pb2.DeployRobotResponse(
                success=True,
                message=f"Successfully deployed robot: {request.robot_id}",
                container=updated_robot.container if updated_robot else None
            )
            
        except Exception as e:
            logger.error(f"Error deploying robot: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to deploy robot: {str(e)}")
            return fleet_manager_pb2.DeployRobotResponse(
                success=False,
                message=f"Error: {str(e)}"
            )

    async def UndeployRobot(self, request, context):
        """Undeploy a robot container"""
        logger.info(f"Undeploying robot: {request.robot_id}")
        try:
            # Get robot from registry
            robot = await self.registry.get_robot(request.robot_id)
            if not robot:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Robot with ID {request.robot_id} not found")
                return fleet_manager_pb2.UndeployRobotResponse(
                    success=False,
                    message=f"Robot with ID {request.robot_id} not found"
                )

            # Use integer enum for status checks and updates
            running_state = fleet_manager_pb2.RobotStatus.State.RUNNING
            deploying_state = fleet_manager_pb2.RobotStatus.State.DEPLOYING
            registered_state = fleet_manager_pb2.RobotStatus.State.REGISTERED

            # Only allow undeployment if robot is running or deploying
            if not (robot.status.state == running_state or robot.status.state == deploying_state):
                logger.warning(f"Robot {request.robot_id} is not running or deploying")
                return fleet_manager_pb2.UndeployRobotResponse(
                    success=False,
                    message=f"Robot {request.robot_id} is not running or deploying"
                )

            # Extract deployment info
            docker_host = "localhost"
            docker_port = 2375
            
            if robot.HasField('deployment'):
                docker_host = robot.deployment.docker_host
                docker_port = robot.deployment.docker_port
            
            # Undeploy container
            try:
                success = await self.container_manager.stop_robot(
                    robot_id=request.robot_id,
                    host=docker_host,
                    docker_port=docker_port
                )
            except Exception as container_error:
                logger.error(f"Container undeployment error: {str(container_error)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to undeploy container: {str(container_error)}")
                return fleet_manager_pb2.UndeployRobotResponse(
                    success=False,
                    message=f"Failed to undeploy container: {str(container_error)}"
                )
            
            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to undeploy container for robot {request.robot_id}")
                return fleet_manager_pb2.UndeployRobotResponse(
                    success=False,
                    message=f"Failed to undeploy container for robot {request.robot_id}"
                )
            
            # Update robot container info in registry (set to empty)
            await self.registry.update_container_info(robot.robot_id, None)
            
            # Update robot status
            updated_robot = await self.registry.update_robot_status(request.robot_id, registered_state)
            logger.info(f"Successfully undeployed robot: {request.robot_id}")
            return fleet_manager_pb2.UndeployRobotResponse(
                success=True,
                message=f"Successfully undeployed robot: {request.robot_id}"
            )
            
        except Exception as e:
            logger.error(f"Error undeploying robot: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to undeploy robot: {str(e)}")
            return fleet_manager_pb2.UndeployRobotResponse(
                success=False,
                message=f"Error: {str(e)}"
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
            # Create task using registry
            task = await self.registry.create_task(
                description=request.description,
                robot_id=request.robot_id if request.robot_id else None,
                goal_id=request.goal_id if request.goal_id else None,
                plan_id=request.plan_id if request.plan_id else None,
                robot_type=request.robot_type if request.robot_type else None,
                dependency_task_ids=list(request.dependency_task_ids) if request.dependency_task_ids else None
            )
            
            if not task:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Failed to create task")
                return fleet_manager_pb2.CreateTaskResponse(
                    error="Failed to create task"
                )
            
            logger.info(f"Successfully created task: {task.task_id}")
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
            # Get the task to return in the response
            task = await self.registry.get_task(request.task_id)
            if not task:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Task with ID {request.task_id} not found")
                return fleet_manager_pb2.DeleteTaskResponse(
                    error=f"Task with ID {request.task_id} not found"
                )
            
            # Delete the task
            success = await self.registry.delete_task(request.task_id)
            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to delete task with ID {request.task_id}")
                return fleet_manager_pb2.DeleteTaskResponse(
                    error=f"Failed to delete task with ID {request.task_id}"
                )
            
            logger.info(f"Successfully deleted task: {request.task_id}")
            return fleet_manager_pb2.DeleteTaskResponse(
                task=task
            )
            
        except Exception as e:
            logger.error(f"Error deleting task: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to delete task: {str(e)}")
            return fleet_manager_pb2.DeleteTaskResponse(
                error=f"Error: {str(e)}"
            )
            
    async def CreatePlan(self, request, context):
        """Create a new plan"""
        logger.info(f"Creating new plan with strategy: {request.planning_strategy} and allocation: {request.allocation_strategy}")
        try:
            # Get the requested planning strategy and allocation strategy
            planning_strategy = request.planning_strategy
            allocation_strategy = request.allocation_strategy
            goal_ids = list(request.goal_ids) if request.goal_ids else []
            
            if not goal_ids:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("At least one goal ID must be provided")
                return fleet_manager_pb2.CreatePlanResponse(
                    error="At least one goal ID must be provided"
                )
            
            # For testing environments, create a plan directly without using the planner
            test_mode = os.getenv("TESTING", "false").lower() in ("true", "1", "yes")
            if test_mode:
                logger.info("Running in test mode, bypassing planner")
                plan = await self.registry.create_plan(
                    planning_strategy=planning_strategy,
                    allocation_strategy=allocation_strategy,
                    goal_ids=goal_ids,
                    task_ids=[]
                )
                return fleet_manager_pb2.CreatePlanResponse(plan=plan)
            
            # Normal flow using the planner
            from robot_fleet.server.planner.planner import get_planner
            from robot_fleet.server.planner.allocator import get_allocator
            planner = get_planner(planning_strategy)
            print(f"Using planner: {planner}")
            allocator = get_allocator(allocation_strategy)
            print(f"Using allocator: {allocator}")
            
            # Generate plan using the planner
            try:
                logger.info(f"Generating plan for goals {goal_ids} using {planning_strategy} planner")
                plan_json = await planner.plan(goal_ids)
                plan_id = await planner.save_plan_to_db(plan_json, planning_strategy, allocation_strategy, goal_ids)
                logger.info(f"Successfully created and saved plan {plan_id}")
            except Exception as e:
                logger.error(f"Error during planning: {str(e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Planning failed: {str(e)}")
                return fleet_manager_pb2.CreatePlanResponse(
                    error=f"Planning failed: {str(e)}"
                )

            # Allocate tasks using the allocator
            allocation = await allocator.allocate(plan_id)

            print(f"Task allocation complete: {allocation}")
            
            # Get the plan with all tasks
            plan = await self.registry.get_plan(plan_id)
            
            logger.info(f"Successfully created plan: {plan_id}")
            return fleet_manager_pb2.CreatePlanResponse(
                plan=plan
            )
            
        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to create plan: {str(e)}")
            return fleet_manager_pb2.CreatePlanResponse(
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
        from robot_fleet.server.executor.executor import Executor
        print(f"Received StartPlan request for plan_id: {request.plan_id}")
        try:
            executor = Executor(plan_id=request.plan_id)
            await executor.execute()
            return fleet_manager_pb2.StartPlanResponse(error="")
        except Exception as e:
            print(f"Error in StartPlan: {e}")
            return fleet_manager_pb2.StartPlanResponse(error=f"Failed to start plan: {e}")


async def serve(port: int = 50051, db_url: str = None, reset_db: bool = False, verbose: bool = False, sql_debug: bool = False):
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
        db_url = "postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet"
        print(f"Using default PostgreSQL database: {db_url}")
    else:
        print(f"Using database: {db_url}")

    # Create the service first
    print("Creating service...")
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    service = FleetManagerService(db_url)
    
    # Initialize the database with reset flag
    if reset_db:
        print("\nResetting database tables...")
        async with service.registry.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            print("All tables dropped successfully")
    
    # Initialize/create tables using the service's registry
    print("Creating database tables...")
    await service.initialize()
    print("Database initialization complete")
    
    # Print all available SQL tables
    async with service.registry.engine.connect() as conn:
        result = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
        tables = result.fetchall()
        print("\nAvailable SQL tables:")
        for table in tables:
            print(f"- {table[0]}")
          
    fleet_manager_pb2_grpc.add_FleetManagerServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    print(f"\n🚀 Fleet Manager Server is now serving on port {port}")
    
    try:
        await server.start()
        await server.wait_for_termination()
    finally:
        # Clean up resources
        await service.cleanup() 