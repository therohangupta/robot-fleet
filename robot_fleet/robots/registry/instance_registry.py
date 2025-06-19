from typing import Dict, Optional, List, Any
from google.protobuf import timestamp_pb2
import grpc

import json
import asyncio
from functools import wraps
from datetime import datetime
import logging
from google.protobuf.json_format import MessageToDict

from sqlalchemy import create_engine, Column, String, Integer, JSON, DateTime, ForeignKey, update, text
from sqlalchemy.orm import sessionmaker, relationship, selectinload, undefer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, AsyncEngine, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.attributes import flag_modified

from robot_fleet.proto import fleet_manager_pb2
from .models import (
    Base, RobotModel, TaskModel, PlanModel, GoalModel, WorldStatement,
    robot_model_to_proto, task_model_to_proto, plan_model_to_proto, goal_model_to_proto,
    robot_proto_to_model, task_proto_to_model, plan_proto_to_model, goal_proto_to_model,
    world_statement_model_to_proto
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# --- Removed Dataclasses --- 
# RobotMetadata, RobotDeployment, RobotContainer, MCPClient, and RegisteredRobot 
# are now redundant. We use RobotModel for DB (with JSON fields for complex types)
# and fleet_manager_pb2 messages (Robot, Goal, Task, Plan) for API/return types.

def configure_registry_logging(verbose: bool = False):
    """Configure logging for the registry module
    
    Args:
        verbose: If True, enables detailed logging
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)  # Only show warnings and errors by default

# Configure SQL logging to only show errors
for logger_name in ['sqlalchemy.engine', 'sqlalchemy.pool', 'sqlalchemy.dialects', 'sqlalchemy.orm']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

def db_retry(max_retries=3, delay=1):
    """Decorator for retrying database operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Don't retry on integrity errors - these are permanent
                    if isinstance(e, IntegrityError):
                        raise
                    last_error = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator

class RobotInstanceRegistry:
    """Registry for managing robot instances using PostgreSQL as backend storage"""
    
    def __init__(self, db_url: str, engine: Optional[AsyncEngine] = None):
        """Initializes the registry with a database URL or an existing engine."""
        if engine:
            self.engine = engine
        else:
            self.engine = create_async_engine(
                db_url,
                echo=False,  # Enable SQL logging for debugging
            )
        self.async_session_factory = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    async def initialize(self):
        """Initialize the database schema"""
        logger.debug("Creating tables if they don't exist...")
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.debug("Database initialization complete")

    @db_retry()
    async def register_robot(
        self,
        robot_id: str,
        robot_type: str,
        description: str,
        capabilities: List[str],
        status: int = fleet_manager_pb2.RobotStatus.State.REGISTERED,
        container_info: Optional[fleet_manager_pb2.ContainerInfo] = None, # Match proto type
        deployment_info: Optional[fleet_manager_pb2.DeploymentInfo] = None, # Match proto type
        task_server_info: Optional[fleet_manager_pb2.TaskServerInfo] = None, # Renamed proto type
    ) -> Optional[fleet_manager_pb2.Robot]:
        """Register a new robot or update existing one."""
        
        # Prepare data for RobotModel (handle potential None values)
        now = datetime.now()
        # Ensure we have the correct proto message types, default to empty if None
        container_info_msg = container_info or fleet_manager_pb2.ContainerInfo()
        deployment_info_msg = deployment_info or fleet_manager_pb2.DeploymentInfo()
        task_server_info_msg = task_server_info or fleet_manager_pb2.TaskServerInfo() # Renamed

        # Convert proto messages to dictionaries for JSON storage
        container_dict = MessageToDict(container_info_msg, preserving_proto_field_name=True) if container_info_msg else {}
        deployment_dict = MessageToDict(deployment_info_msg, preserving_proto_field_name=True) if deployment_info_msg else {}
        task_server_dict = MessageToDict(task_server_info_msg, preserving_proto_field_name=True) if task_server_info_msg else {} # Renamed variable

        async with self.async_session_factory() as session:
            async with session.begin():
                # Check if robot already exists using scalar_one_or_none
                existing_robot_result = await session.execute(
                    select(RobotModel)
                    .options(selectinload(RobotModel.tasks).options(undefer(TaskModel.dependency_task_ids)))
                    .where(RobotModel.robot_id == robot_id)
                )
                existing_robot = existing_robot_result.scalar_one_or_none()

                if existing_robot:
                    # Update existing robot (optional, or raise error)
                    logging.warning(f"Robot {robot_id} already exists. Update not implemented yet.")
                    # Pass the loaded tasks to the converter
                    return robot_model_to_proto(existing_robot, existing_robot.tasks)

                # Create new RobotModel instance with direct arguments
                robot_model = RobotModel(
                    robot_id=robot_id,
                    robot_type=robot_type,
                    description=description,
                    capabilities=capabilities,
                    status=status,
                    # Store the dictionaries directly in the JSON fields matching model names
                    container_info=container_dict, # Use prepared dict
                    deployment_info=deployment_dict, # Use prepared dict
                    task_server_info=task_server_dict, # Use prepared dict
                    last_updated=now
                )
                session.add(robot_model)
                await session.flush() # Flush to get potential errors early

                # Pass empty list for tasks as it's a new robot
                return robot_model_to_proto(robot_model, [])

    @db_retry()
    async def update_robot(
        self,
        robot_id: str,
        # Pass updates as optional dictionaries or individual fields
        metadata_update: Optional[Dict[str, Any]] = None,
        deployment_update: Optional[Dict[str, Any]] = None,
        container_update: Optional[Dict[str, Any]] = None,
        task_server_update: Optional[Dict[str, Any]] = None, # Renamed parameter
        capabilities_update: Optional[List[str]] = None,
        status_update: Optional[str] = None,
        task_ids_update: Optional[List[int]] = None # Use dedicated methods for task assignment? 
    ) -> Optional[fleet_manager_pb2.Robot]:
        """Update an existing robot's information."""
        
        async with self.async_session_factory() as session:
            async with session.begin():
                robot_model = await session.get(RobotModel, robot_id)
                if not robot_model:
                    logger.warning(f"Robot with ID {robot_id} not found for update.")
                    return None

                updated = False
                # Update fields if new data is provided
                if metadata_update is not None:
                    # Ensure existing dict is updated, not replaced if it's None initially
                    if robot_model.robot_metadata is None: robot_model.robot_metadata = {}
                    robot_model.robot_metadata.update(metadata_update)
                    updated = True
                if deployment_update is not None:
                    if robot_model.deployment_info is None: robot_model.deployment_info = {}
                    robot_model.deployment_info.update(deployment_update)
                    updated = True
                if container_update is not None:
                    if robot_model.container_info is None: robot_model.container_info = {}
                    robot_model.container_info.update(container_update)
                    updated = True
                if task_server_update is not None: # Renamed variable
                    if robot_model.task_server_info is None: robot_model.task_server_info = {} # Renamed attribute
                    robot_model.task_server_info.update(task_server_update) # Renamed attribute and variable
                    updated = True
                if capabilities_update is not None:
                    robot_model.capabilities = capabilities_update
                    updated = True
                if status_update is not None:
                    # Add validation against fleet_manager_pb2.RobotStatus enum? 
                    robot_model.status = status_update
                    updated = True
                if task_ids_update is not None:
                    # Consider if overwriting the list is the desired behavior
                    robot_model.task_ids = task_ids_update
                    updated = True

                if updated:
                    robot_model.last_updated = datetime.utcnow()
                    await session.flush()
                    logger.info(f"Updated robot ID {robot_id}")
                else:
                    logger.info(f"No updates specified for robot ID {robot_id}")

                # Re-fetch with tasks loaded to pass to converter
                updated_robot_result = await session.execute(
                    select(RobotModel)
                    .options(selectinload(RobotModel.tasks).options(undefer(TaskModel.dependency_task_ids)))
                    .where(RobotModel.robot_id == robot_id)
                )
                robot_model = updated_robot_result.scalar_one()

                # Pass the loaded tasks to the conversion function
                return robot_model_to_proto(robot_model, robot_model.tasks)

    @db_retry()
    async def update_robot_status(self, robot_id: str, status: int) -> Optional[fleet_manager_pb2.Robot]:
        """Update the status of a robot by ID. Status should be an integer enum value."""
        async with self.async_session_factory() as session:
            async with session.begin():
                robot = await session.get(RobotModel, robot_id)
                if not robot:
                    logger.warning(f"Robot with ID {robot_id} not found for update.")
                    return None
                robot.status = status
                await session.flush()
                # Eagerly load tasks using selectinload in a new query
                stmt = (
                    select(RobotModel)
                    .options(selectinload(RobotModel.tasks))
                    .where(RobotModel.robot_id == robot_id)
                )
                result = await session.execute(stmt)
                robot_with_tasks = result.scalar_one()
                return robot_model_to_proto(robot_with_tasks, robot_with_tasks.tasks)

    @db_retry()
    async def delete_robot(self, robot_id: str) -> bool:
        """Remove a robot registration from the database"""
        async with self.async_session_factory() as session:
            async with session.begin():
                # Find the robot
                result = await session.execute(
                    select(RobotModel).where(RobotModel.robot_id == robot_id)
                )
                robot = result.scalar_one_or_none()
                
                if not robot:
                    logger.warning(f"Attempted to unregister non-existent robot ID: {robot_id}")
                    return False
                
                # Remove robot_id reference from all associated tasks
                await session.execute(
                    update(TaskModel)
                    .where(TaskModel.robot_id == robot_id)
                    .values(robot_id=None)
                )
                logger.info(f"Removed robot ID {robot_id} references from all tasks")
                
                # Delete the robot
                await session.delete(robot)
                logger.info(f"Unregistered robot ID: {robot_id}")
                return True

    @db_retry()
    async def get_robot(self, robot_id: str) -> Optional[fleet_manager_pb2.Robot]:
        """Get a specific robot by ID"""
        async with self.async_session_factory() as session:
            # Eagerly load tasks relationship
            stmt = (
                select(RobotModel)
                .options(selectinload(RobotModel.tasks).options(undefer(TaskModel.dependency_task_ids)))
                .where(RobotModel.robot_id == robot_id)
            )
            result = await session.execute(stmt)
            robot_model = result.scalar_one_or_none()
            
            if robot_model:
                # Convert model to proto (tasks were eager loaded)
                # Pass the loaded tasks to the converter
                return robot_model_to_proto(robot_model, robot_model.tasks)
            else:
                logger.warning(f"Robot with ID {robot_id} not found.")
                return None

    @db_retry()
    async def list_robots(self) -> List[fleet_manager_pb2.Robot]:
        """List all registered robots"""
        async with self.async_session_factory() as session:
            # Eagerly load tasks relationship for all robots
            stmt = (
                select(RobotModel)
                .options(selectinload(RobotModel.tasks).options(undefer(TaskModel.dependency_task_ids)))
            )
            result = await session.execute(stmt)
            robot_models = result.scalars().all()
            
            robot_protos = []
            for model in robot_models:
                logger.debug(f"[list_robots] Processing robot model: {model.robot_id}")
                # Convert model to proto (tasks were eager loaded)
                # Pass the loaded tasks to the conversion function
                robot_protos.append(robot_model_to_proto(model, model.tasks))
                
            logger.info(f"Listed {len(robot_protos)} robots.")
            return robot_protos

    @db_retry()
    async def update_container_info(self, robot_id: str, container_info: fleet_manager_pb2.ContainerInfo) -> None:
        """Update container info for a registered robot"""
        async with self.async_session_factory() as session:
            result = await session.execute(
                select(RobotModel).where(RobotModel.robot_id == robot_id)
            )
            robot_model = result.scalar_one_or_none()
            
            logger.info(f"Updating container info for robot {robot_id}")
            logger.info(f"Container info: {container_info}")
            
            if robot_model:
                # Get existing container data or initialize if None
                container_data = robot_model.container_info if robot_model.container_info else {}
                
                if container_info is None:
                    # For undeployment, preserve the original image and environment
                    container_data = {
                        "image": container_data.get("image", ""),
                        "environment": container_data.get("environment", {}),
                        "container_id": None
                    }
                else:
                    # Convert ContainerInfo proto to dictionary
                    container_data = MessageToDict(container_info, preserving_proto_field_name=True)
                
                # Update the container_info field
                robot_model.container_info = container_data
                
                # Save to database
                session.add(robot_model)
                await session.commit()
            else:
                logger.warning(f"Warning: Robot {robot_id} not found in database")

    async def get_robot_status(self, robot_id: str, deployment_status: str = "unknown") -> Optional[fleet_manager_pb2.RobotStatus]:
        """Get status of a registered robot"""
        robot = await self.get_robot(robot_id)
        if not robot:
            return None

        # Create a RobotStatus with the fields defined in the proto file
        if robot.container.container_info:
            # Robot is deployed and running
            return fleet_manager_pb2.RobotStatus(
                state=fleet_manager_pb2.RobotStatus.State.RUNNING,
                message=f"Robot {robot_id} is running"
            )
        else:
            # Robot is registered but not deployed
            return fleet_manager_pb2.RobotStatus(
                state=fleet_manager_pb2.RobotStatus.State.REGISTERED,
                message=f"Robot {robot_id} is registered"
            )

    @db_retry()
    async def create_goal(self, description: str) -> Optional[fleet_manager_pb2.Goal]:
        """Create a new goal"""
        async with self.async_session_factory() as session:
            async with session.begin():
                # Create GoalModel instance
                new_goal = GoalModel(
                    description=description,
                    # status="PENDING" # Add status if needed in model/proto
                )
                session.add(new_goal)
                await session.flush() # Flush to get the auto-generated ID
                
                logger.info(f"Created goal with ID: {new_goal.goal_id}")
                
                # Optionally raise an error or proceed without linking missing tasks

                # Link found tasks
                goal_model = new_goal
                # tasks_result = await session.execute(select(TaskModel).where(TaskModel.goal_id == new_goal.goal_id))
                # goal_model.tasks = tasks_result.scalars().all() # Assign tasks if relationship isn't automatically loaded

            await session.flush() # Ensure links are persisted before converting

            # Explicitly load the tasks relationship before converting to proto
            # This prevents MissingGreenlet errors caused by lazy loading during conversion
            stmt = (
                select(GoalModel)
                .options(selectinload(GoalModel.tasks).options(undefer(TaskModel.dependency_task_ids)))
                .where(GoalModel.goal_id == goal_model.goal_id)
            )
            result = await session.execute(stmt)
            refreshed_goal = result.scalar_one()

            # Pass the loaded tasks to the conversion function
            return goal_model_to_proto(refreshed_goal, refreshed_goal.tasks)

    @db_retry()
    async def delete_goal(self, goal_id: int) -> bool:
        """Delete a goal by ID and associated tasks"""
        async with self.async_session_factory() as session:
            async with session.begin():
                # Find the goal
                goal = await session.get(GoalModel, goal_id)
                if not goal:
                    logger.warning(f"Goal with ID {goal_id} not found for deletion.")
                    return False
            
                # Find plans associated with this goal and remove the goal_id from their list
                # This assumes goal_ids is a list of integers stored in JSON
                potential_plans = await session.execute(select(PlanModel)) # Select all plans for simplicity
                for plan in potential_plans.scalars().all():
                    if isinstance(plan.goal_ids, list) and goal_id in plan.goal_ids:
                        plan.goal_ids.remove(goal_id)
                        # Explicitly mark JSON column as modified to ensure changes are saved
                        flag_modified(plan, "goal_ids")
                        logger.debug(f"Removed goal ID {goal_id} from plan ID {plan.plan_id}.goal_ids")
            
                # Flush changes to plans before deleting the goal
                await session.flush()

                # Delete the goal - tasks will be deleted automatically because of cascade="all, delete-orphan"
                await session.delete(goal)
                logger.info(f"Deleted goal ID: {goal_id} and all associated tasks")
                return True

    @db_retry()
    async def list_goals(self) -> List[fleet_manager_pb2.Goal]:
        """List all goals in the system, with their tasks."""
        async with self.async_session_factory() as session:
            result = await session.execute(select(GoalModel))
            goal_models = result.scalars().all()
            # Get all tasks in a single query
            tasks_result = await session.execute(select(TaskModel))
            all_tasks = tasks_result.scalars().all()
            # Group tasks by goal_id
            tasks_by_goal = {}
            for task in all_tasks:
                if task.goal_id is not None:
                    tasks_by_goal.setdefault(task.goal_id, []).append(task)
            goal_protos = []
            for goal in goal_models:
                goal_proto = fleet_manager_pb2.Goal()
                goal_proto.goal_id = goal.goal_id
                goal_proto.description = goal.description or ""
                tasks = tasks_by_goal.get(goal.goal_id, [])
                goal_proto.task_ids.extend([task.task_id for task in tasks])
                goal_protos.append(goal_model_to_proto(goal, tasks))
            return goal_protos

    @db_retry()
    async def get_goal(self, goal_id: int) -> Optional[fleet_manager_pb2.Goal]:
        """Retrieve a goal by its ID."""
        async with self.async_session_factory() as session:
            # Eagerly load tasks associated with the goal
            stmt = (
                select(GoalModel)
                .options(selectinload(GoalModel.tasks).options(undefer(TaskModel.dependency_task_ids)))
                .where(GoalModel.goal_id == goal_id)
            )
            result = await session.execute(stmt)
            goal_model = result.scalar_one_or_none()
            if goal_model:
                # Pass both the goal model and its loaded tasks to goal_model_to_proto
                return goal_model_to_proto(goal_model, goal_model.tasks)
            else:
                logger.warning(f"Goal with ID {goal_id} not found.")
                return None

    @db_retry()
    async def create_task(
        self,
        description: str,
        robot_id: Optional[str] = None, # Changed type hint to str
        goal_id: Optional[int] = None,
        plan_id: Optional[int] = None,
        dependency_task_ids: Optional[List[int]] = None,
        status: int = fleet_manager_pb2.TaskStatus.TASK_PENDING,
        robot_type: Optional[str] = None  # Add robot_type parameter
    ) -> Optional[fleet_manager_pb2.Task]:
        """Create a new task and assign it optionally to a robot and goal"""
        async with self.async_session_factory() as session:
            async with session.begin():
                # Validate Robot ID if provided
                if robot_id is not None:
                    # Get RobotModel using string robot_id
                    robot = await session.get(RobotModel, robot_id)
                    if not robot:
                        logger.warning(f"Robot with ID {robot_id} not found for task assignment.")
                        return None

                # Check if plan exists, create if not
                if plan_id is not None:
                    plan = await session.get(PlanModel, plan_id)
                
                if plan_id is None or not plan:
                    # Create a new plan with default strategies
                    logger.info(f"Creating new plan for goal {goal_id}")
                    plan = await self.create_plan(
                        planning_strategy=fleet_manager_pb2.PlanningStrategy.PLANNING_STRATEGY_UNSPECIFIED,  # Default strategy
                        allocation_strategy=fleet_manager_pb2.AllocationStrategy.ALLOCATION_STRATEGY_UNSPECIFIED,  # Default strategy
                        goal_ids=[goal_id] if goal_id else []
                    )
                    plan_id = plan.plan_id
                
                new_task = TaskModel(
                    description=description,
                    robot_id=robot_id,
                    goal_id=goal_id,
                    plan_id=plan_id,
                    dependency_task_ids=dependency_task_ids or [],
                    status=status,
                    robot_type=robot_type  # Pass robot_type to the model
                )
                session.add(new_task)
                await session.flush() # Get the auto-generated task_id
                
                logger.info(f"Created task with ID: {new_task.task_id}")
                
                # Convert to proto
                return task_model_to_proto(new_task)

    @db_retry()
    async def update_task_status(self, task_id: int, status: int) -> Optional[fleet_manager_pb2.Task]:
        """Update the status of a task by ID. Status should be an integer enum value."""
        async with self.async_session_factory() as session:
            async with session.begin():
                task = await session.get(TaskModel, task_id)
                if not task:
                    logger.warning(f"Task with ID {task_id} not found for update.")
                    return None
                task.status = status
                await session.flush()
                return task_model_to_proto(task)

    @db_retry()
    async def update_task(self, task_id: int, **kwargs) -> Optional[fleet_manager_pb2.Task]:
        """Update fields of a task by ID. Accepts any TaskModel column as kwarg (e.g., plan_id, goal_id, robot_id, status, etc.)."""
        async with self.async_session_factory() as session:
            async with session.begin():
                task = await session.get(TaskModel, task_id)
                if not task:
                    logger.warning(f"Task with ID {task_id} not found for update.")
                    return None
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                await session.flush()
                return task_model_to_proto(task)

    @db_retry()
    async def delete_task(self, task_id: int) -> bool:
        """Delete a task by ID."""
        async with self.async_session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    select(TaskModel).where(TaskModel.task_id == task_id)
                )
                task_model = result.scalar_one_or_none()

                if task_model:
                    # No need to update plan_model.task_ids; SQLAlchemy handles relationship via foreign keys
                    await session.delete(task_model)
                    await session.flush()
                    logger.info(f"Deleted task with ID: {task_id}")
                    return True

    @db_retry()
    async def list_tasks(self, plan_ids: Optional[List[int]] = None, goal_ids: Optional[List[int]] = None, 
                         robot_ids: Optional[List[str]] = None) -> List[fleet_manager_pb2.Task]:
        """List tasks with optional filtering.
        
        Args:
            plan_ids: Optional list of plan IDs to filter by
            goal_ids: Optional list of goal IDs to filter by
            robot_ids: Optional list of robot IDs to filter by
            
        Returns:
            List of matching Task protos
        """
        async with self.async_session_factory() as session:
            # Start with a base query
            query = select(TaskModel)
            
            # Add filters if provided
            if plan_ids and len(plan_ids) > 0:
                logger.debug(f"Filtering tasks by plan_ids: {plan_ids}")
                # Since a task is associated with only one plan, check if the task's plan_id is in the list
                query = query.where(TaskModel.plan_id.in_(plan_ids))
                
            if goal_ids and len(goal_ids) > 0:
                logger.debug(f"Filtering tasks by goal_ids: {goal_ids}")
                # Since a task is associated with only one goal, check if the task's goal_id is in the list
                query = query.where(TaskModel.goal_id.in_(goal_ids))
                
            if robot_ids and len(robot_ids) > 0:
                logger.debug(f"Filtering tasks by robot_ids: {robot_ids}")
                query = query.where(TaskModel.robot_id.in_(robot_ids))
            
            # Execute the query
            result = await session.execute(query)
            task_models = result.scalars().all()
            
            task_protos = []
            for model in task_models:
                logger.debug(f"Processing task model: {model.task_id}")
                # Convert model to proto
                task_protos.append(task_model_to_proto(model))
                
            logger.info(f"Listed {len(task_protos)} tasks with filters: plan_ids={plan_ids}, goal_ids={goal_ids}, robot_ids={robot_ids}")
            return task_protos

    @db_retry()
    async def get_task(self, task_id: int) -> Optional[fleet_manager_pb2.Task]:
        """Get a specific task by ID"""
        async with self.async_session_factory() as session:
            task_model = await session.get(TaskModel, task_id)
            
            if task_model:
                logger.debug(f"Found task model: {task_model.task_id}")
                return task_model_to_proto(task_model)
            else:
                logger.warning(f"Task with ID {task_id} not found.")
                return None

    @db_retry()
    async def get_tasks_by_robot(self, robot_id: int) -> List[fleet_manager_pb2.Task]:
        """Get all tasks assigned to a specific robot"""
        async with self.async_session_factory() as session:
            result = await session.execute(
                select(TaskModel).where(TaskModel.robot_id == robot_id)
            )
            task_models = result.scalars().all()
            task_protos = []
            for model in task_models:
                task_protos.append(task_model_to_proto(model))
            logger.info(f"Found {len(task_protos)} tasks for robot ID {robot_id}")
            return task_protos

    @db_retry()
    async def get_tasks_by_goal(self, goal_id: int) -> List[fleet_manager_pb2.Task]:
        """Get all tasks associated with a specific goal"""
        async with self.async_session_factory() as session:
            result = await session.execute(
                select(TaskModel).where(TaskModel.goal_id == goal_id)
            )
            task_models = result.scalars().all()
            task_protos = []
            for model in task_models:
                task_protos.append(task_model_to_proto(model))
            logger.info(f"Found {len(task_protos)} tasks for goal ID {goal_id}")
            return task_protos

    @db_retry()
    async def get_task_status(self, task_id: int) -> Optional[str]:
        """Get the status of a specific task"""
        task_proto = await self.get_task(task_id)
        if task_proto:
            # Convert enum value to string name
            return fleet_manager_pb2.TaskStatus.Name(task_proto.status)
        return None

    # --- Plan Management ---
    @db_retry()
    async def create_plan(
        self, 
        planning_strategy: fleet_manager_pb2.PlanningStrategy, 
        allocation_strategy: fleet_manager_pb2.AllocationStrategy, 
        task_ids: Optional[List[int]] = None,
        goal_ids: Optional[List[int]] = None
    ) -> Optional[fleet_manager_pb2.Plan]:
        """Create a new plan and link it to tasks and optionally goals."""
        logger.info(f"Attempting to create plan with strategy {planning_strategy} and allocation {allocation_strategy} for tasks {task_ids} and goals {goal_ids}")
        async with self.async_session_factory() as session:
            async with session.begin():
                # Convert strategy enum value to its integer representation for storage
                strategy_int = planning_strategy
                allocation_int = allocation_strategy
                new_plan = PlanModel(
                    planning_strategy=strategy_int,
                    allocation_strategy=allocation_int,
                    goal_ids=goal_ids or []
                )
                session.add(new_plan)
                await session.flush() # Persist to get plan_id
                # Link tasks if provided
                if task_ids:
                    for tid in task_ids:
                        task = await session.get(TaskModel, tid)
                        if task:
                            task.plan_id = new_plan.plan_id
                await session.flush()
                # Fetch tasks for the new plan
                plan_tasks = [task for task in (await session.execute(select(TaskModel).where(TaskModel.plan_id == new_plan.plan_id))).scalars().all()]
                # Convert to proto
                return plan_model_to_proto(new_plan, plan_tasks)

    @db_retry()
    async def get_plan(self, plan_id: int) -> Optional[fleet_manager_pb2.Plan]:
        """Retrieve a plan by its ID, including linked tasks."""
        async with self.async_session_factory() as session:
            # Find the plan model
            result = await session.execute(
                select(PlanModel).where(PlanModel.plan_id == plan_id)
            )
            plan_model = result.scalar_one_or_none()
            if plan_model is None:
                logger.warning(f"Plan with ID {plan_id} not found.")
                return None
            
            # Query all tasks for this plan
            plan_tasks = [task for task in (await session.execute(select(TaskModel).where(TaskModel.plan_id == plan_id))).scalars().all()]
            print(plan_tasks)
            # Convert to proto
            return plan_model_to_proto(plan_model, plan_tasks)

    @db_retry()
    async def list_plans(self) -> List[fleet_manager_pb2.Plan]:
        """List all plans in the system"""
        async with self.async_session_factory() as session:
            # Get all plans
            result = await session.execute(select(PlanModel))
            plan_models = result.scalars().all()
            # Get all tasks in a single query
            tasks_result = await session.execute(select(TaskModel))
            all_tasks = tasks_result.scalars().all()
            # Group tasks by plan_id
            tasks_by_plan = {}
            for task in all_tasks:
                if task.plan_id is not None:
                    tasks_by_plan.setdefault(task.plan_id, []).append(task)
            plan_protos = []
            for plan in plan_models:
                # Fetch tasks for the plan
                plan_tasks = tasks_by_plan.get(plan.plan_id, [])
                # Convert to proto
                plan_protos.append(plan_model_to_proto(plan, plan_tasks))
            return plan_protos

    @db_retry()
    async def update_plan(
        self, 
        plan_id: int, 
        goal_ids: Optional[List[int]] = None,
        task_ids: Optional[List[int]] = None,
        planning_strategy: Optional[fleet_manager_pb2.PlanningStrategy] = None,
        allocation_strategy: Optional[fleet_manager_pb2.AllocationStrategy] = None
    ) -> Optional[fleet_manager_pb2.Plan]:
        """Update a plan's information."""
        async with self.async_session_factory() as session:
            async with session.begin():
                # Get the plan
                plan_result = await session.execute(
                    select(PlanModel).where(PlanModel.plan_id == plan_id)
                )
                plan_model = plan_result.scalar_one_or_none()
                if not plan_model:
                    logger.warning(f"Plan with ID {plan_id} not found.")
                    return None

                # Update fields if provided
                if planning_strategy is not None:
                    plan_model.planning_strategy = planning_strategy
                if allocation_strategy is not None:
                    plan_model.allocation_strategy = allocation_strategy
                if goal_ids is not None:
                    plan_model.goal_ids = goal_ids
                # Unlink all existing tasks if task_ids is provided
                if task_ids is not None:
                    # Unlink all current tasks from this plan
                    tasks_result = await session.execute(select(TaskModel).where(TaskModel.plan_id == plan_id))
                    for task in tasks_result.scalars().all():
                        task.plan_id = None
                    # Link new tasks
                    for tid in task_ids:
                        task = await session.get(TaskModel, tid)
                        if task:
                            task.plan_id = plan_id
                await session.flush()
                # Fetch tasks for the updated plan
                plan_tasks = [task for task in (await session.execute(select(TaskModel).where(TaskModel.plan_id == plan_id))).scalars().all()]
                # Convert to proto
                return plan_model_to_proto(plan_model, plan_tasks)

    @db_retry()
    async def delete_plan(self, plan_id: int) -> bool:
        """Delete a plan by its ID and all associated tasks."""
        async with self.async_session_factory() as session:
            async with session.begin():
                result = await session.execute(select(PlanModel).where(PlanModel.plan_id == plan_id))
                plan_model = result.scalar_one_or_none()

                if plan_model:
                    # Delete the plan - tasks will be deleted automatically because of cascade="all, delete-orphan"
                    await session.delete(plan_model)
                    await session.flush()
                    logger.info(f"Deleted plan with ID: {plan_id} and all associated tasks.")
                    return True
                else:
                    logger.warning(f"Plan with ID {plan_id} not found for deletion.")
                    return False

    # --- World Statement CRUD ---
    @db_retry()
    async def add_world_statement(self, statement: str) -> fleet_manager_pb2.WorldStatement:
        """Adds a new world statement to the database."""
        async with self.async_session_factory() as session:
            async with session.begin():
                try:
                    db_statement = WorldStatement(statement=statement)
                    session.add(db_statement)
                    await session.flush()
                    await session.refresh(db_statement)
                    logger.info(f"Added world statement: ID={db_statement.id}, Statement='{statement[:50]}...'" )
                    return world_statement_model_to_proto(db_statement)
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Error adding world statement: {e}")
                    raise

    @db_retry()
    async def get_world_statement(self, world_statement_id: str) -> Optional[fleet_manager_pb2.WorldStatement]:
        """Retrieves a specific world statement by its ID. Returns None if not found."""
        try:
            ws_id_int = int(str(world_statement_id).strip())
        except ValueError:
            logger.error(f"Invalid world statement ID format: {world_statement_id}")
            return None

        async with self.async_session_factory() as session:
            db_statement = await session.get(WorldStatement, ws_id_int)
            if db_statement is not None:
                logger.debug(f"Retrieved world statement: ID={ws_id_int}")
                return world_statement_model_to_proto(db_statement)
            else:
                logger.warning(f"World statement not found: ID={ws_id_int}")
                return None

    @db_retry()
    async def list_world_statements(self) -> List[fleet_manager_pb2.WorldStatement]:
        """Lists all world statements as protos."""
        async with self.async_session_factory() as session:
            db_statements = await session.execute(select(WorldStatement).order_by(WorldStatement.created_at))
            ws_list = db_statements.scalars().all()
            logger.debug(f"Listing {len(ws_list)} world statements.")
            return [world_statement_model_to_proto(ws) for ws in ws_list]

    @db_retry()
    async def delete_world_statement(self, world_statement_id: str) -> bool:
        """Deletes a world statement by its ID. Returns True if deleted, False if not found or invalid."""
        try:
            ws_id_int = int(str(world_statement_id).strip())
            logger.debug(f"Attempting to delete world statement with ID: {ws_id_int} (original: {world_statement_id})")
        except ValueError:
            logger.error(f"Invalid world statement ID format for deletion: {world_statement_id}")
            return False

        async with self.async_session_factory() as session:
            try:
                db_statement = await session.get(WorldStatement, ws_id_int)
                if not db_statement:
                    logger.warning(f"Attempted to delete non-existent world statement: ID={ws_id_int}")
                    return False

                await session.delete(db_statement)
                await session.flush()
                await session.commit()
                logger.info(f"Deleted world statement: ID={ws_id_int}")
                # Close session to ensure no stale cache
                await session.close()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Error deleting world statement ID={ws_id_int}: {e}")
                raise
