from datetime import datetime
from sqlalchemy import Column, String, Integer, JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base, backref, selectinload
from sqlalchemy import select
import logging
from typing import List, Optional
from ...proto import fleet_manager_pb2
from google.protobuf.json_format import MessageToDict, ParseDict
from sqlalchemy import func
from uuid import uuid4
from sqlalchemy.orm import Session

Base = declarative_base()
logger = logging.getLogger(__name__)

# --- SQLAlchemy Models matching proto ---
class RobotModel(Base):
    __tablename__ = 'robots'
    robot_id = Column(String, primary_key=True)
    robot_type = Column(String, nullable=False)
    description = Column(String, nullable=True)
    capabilities = Column(JSON, nullable=False)
    status = Column(Integer, nullable=True)  # Store as integer for enum compatibility
    container_info = Column(JSON, nullable=True)
    deployment_info = Column(JSON, nullable=True)
    task_server_info = Column(JSON, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    tasks = relationship("TaskModel", back_populates="robot", cascade="all, delete-orphan")

class GoalModel(Base):
    __tablename__ = 'goals'
    goal_id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(String, nullable=False)
    tasks = relationship("TaskModel", back_populates="goal", cascade="all, delete-orphan")

class PlanModel(Base):
    __tablename__ = 'plans'
    plan_id = Column(Integer, primary_key=True, autoincrement=True)
    planning_strategy = Column(Integer, nullable=False) # Store as integer
    allocation_strategy = Column(Integer, nullable=False, default=0) # Store as integer
    goal_ids = Column(JSON, nullable=True)  # List of int64 (authoritative, can be empty)
    tasks = relationship("TaskModel", back_populates="plan", cascade="all, delete-orphan")

class TaskModel(Base):
    __tablename__ = 'tasks'
    task_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    description = Column(String, nullable=False)
    robot_id = Column(String, ForeignKey("robots.robot_id"), nullable=True, index=True)
    goal_id = Column(Integer, ForeignKey("goals.goal_id"), nullable=True, index=True)
    plan_id = Column(Integer, ForeignKey("plans.plan_id"), nullable=True, index=True)
    status = Column(Integer, default=fleet_manager_pb2.TaskStatus.TASK_PENDING)
    dependency_task_ids = Column(JSON, nullable=True, default=[])
    robot_type = Column(String, nullable=True)
    goal = relationship("GoalModel", back_populates="tasks")
    robot = relationship("RobotModel", back_populates="tasks")
    plan = relationship("PlanModel", back_populates="tasks")

class WorldStatement(Base):
    __tablename__ = 'world_statements'
    id = Column(Integer, primary_key=True, autoincrement=True)
    statement = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# --- Model to Proto Conversion Utilities ---
def robot_model_to_proto(robot_model: 'RobotModel', tasks: Optional[List['TaskModel']] = None) -> 'fleet_manager_pb2.Robot':
    robot_proto = fleet_manager_pb2.Robot()
    robot_proto.robot_id = robot_model.robot_id
    robot_proto.robot_type = robot_model.robot_type
    robot_proto.description = robot_model.description or ""
    robot_proto.capabilities.extend(robot_model.capabilities or [])

    # Set RobotStatus message
    if isinstance(robot_model.status, int):
        robot_proto.status.state = robot_model.status
    else:
        # Default to REGISTERED if status is not an int
        robot_proto.status.state = fleet_manager_pb2.RobotStatus.State.REGISTERED

    # Populate TaskServerInfo
    if robot_model.task_server_info and isinstance(robot_model.task_server_info, dict):
        ParseDict(robot_model.task_server_info, robot_proto.task_server_info)
    
    # Populate DeploymentInfo
    if robot_model.deployment_info and isinstance(robot_model.deployment_info, dict):
        ParseDict(robot_model.deployment_info, robot_proto.deployment)

    # Populate ContainerInfo
    if robot_model.container_info and isinstance(robot_model.container_info, dict):
        ParseDict(robot_model.container_info, robot_proto.container)

    if tasks:
        robot_proto.task_ids.extend([task.task_id for task in tasks])
    return robot_proto

def task_model_to_proto(task_model: TaskModel) -> fleet_manager_pb2.Task:
    """Convert TaskModel SQLAlchemy model to Task protobuf message."""
    if not task_model:
        return fleet_manager_pb2.Task()
    
    task_proto = fleet_manager_pb2.Task(
        task_id=task_model.task_id,
        description=task_model.description or "",
        status=task_model.status,
        # Ensure dependency_task_ids is a list of integers, handle None or empty string
        dependency_task_ids=task_model.dependency_task_ids if isinstance(task_model.dependency_task_ids, list) else [],
        robot_type=task_model.robot_type if task_model.robot_type is not None else "" # Add robot_type to proto
    )
    # Conditionally set optional fields if they have values
    if task_model.robot_id is not None:
        task_proto.robot_id = task_model.robot_id
    if task_model.goal_id is not None:
        task_proto.goal_id = task_model.goal_id
    if task_model.plan_id is not None:
        task_proto.plan_id = task_model.plan_id
        
    return task_proto

def goal_model_to_proto(goal_model: 'GoalModel', tasks: Optional[List['TaskModel']] = None) -> 'fleet_manager_pb2.Goal':
    goal_proto = fleet_manager_pb2.Goal()
    goal_proto.goal_id = goal_model.goal_id
    goal_proto.description = goal_model.description or ""
    if tasks:
        goal_proto.task_ids.extend([task.task_id for task in tasks])
    return goal_proto

# --- Proto to Model Conversion Functions ---
def robot_proto_to_model(proto: fleet_manager_pb2.Robot) -> RobotModel:
    return RobotModel(
        robot_id=proto.robot_id,
        robot_type=proto.robot_type,
        description=proto.description,
        capabilities=list(proto.capabilities),
        status=proto.status.state,
        # Read from correct proto fields, store in correct model fields
        container_info=MessageToDict(proto.container, preserving_proto_field_name=True) if proto.HasField('container') else None,
        deployment_info=MessageToDict(proto.deployment, preserving_proto_field_name=True) if proto.HasField('deployment') else None,
        task_server_info=MessageToDict(proto.task_server_info, preserving_proto_field_name=True) if proto.HasField('task_server_info') else None,
        last_updated=proto.last_updated.ToDatetime() if proto.HasField('last_updated') else None
    )

def task_proto_to_model(proto: fleet_manager_pb2.Task) -> TaskModel:
    return TaskModel(
        task_id=proto.task_id,
        description=proto.description,
        goal_id=proto.goal_id if proto.goal_id else None,
        plan_id=proto.plan_id if proto.plan_id else None,
        dependency_task_ids=list(proto.dependency_task_ids),
        robot_id=proto.robot_id if proto.robot_id else None,
        status=fleet_manager_pb2.TaskStatus.Name(proto.status),
        robot_type=proto.robot_type if proto.HasField('robot_type') else None
    )

def goal_proto_to_model(proto: fleet_manager_pb2.Goal) -> GoalModel:
    return GoalModel(
        goal_id=proto.goal_id,
        description=proto.description
        # tasks handled separately
    )

# Make sync: Data should be loaded *before* calling this.
def plan_model_to_proto(plan_model: 'PlanModel', tasks: Optional[List['TaskModel']] = None) -> 'fleet_manager_pb2.Plan':
    plan_proto = fleet_manager_pb2.Plan()
    plan_proto.plan_id = plan_model.plan_id
    plan_proto.planning_strategy = plan_model.planning_strategy
    plan_proto.allocation_strategy = getattr(plan_model, 'allocation_strategy', 0)
    if plan_model.goal_ids:
        plan_proto.goal_ids.extend(plan_model.goal_ids)
    if tasks:
        plan_proto.task_ids.extend([task.task_id for task in tasks])
    return plan_proto

def plan_proto_to_model(proto: fleet_manager_pb2.Plan) -> PlanModel:
    """Convert Plan protobuf message to PlanModel SQLAlchemy object."""
    return PlanModel(
        plan_id=proto.plan_id,
        # Store the integer value of the enum
        planning_strategy=proto.planning_strategy,
        allocation_strategy=proto.allocation_strategy
    )

# --- World Statement Conversion ---
def world_statement_model_to_proto(ws_model: 'WorldStatement') -> 'fleet_manager_pb2.WorldStatement':
    """Converts a WorldStatement SQLAlchemy model to its Protobuf message."""
    ws_proto = fleet_manager_pb2.WorldStatement()
    # Protobuf expects string IDs, even if the DB uses int
    ws_proto.id = str(ws_model.id)
    ws_proto.statement = ws_model.statement
    if ws_model.created_at:
        # Ensure created_at is timezone-aware before conversion if needed,
        # though SQLAlchemy usually handles this with DateTime(timezone=True)
        ws_proto.created_at.FromDatetime(ws_model.created_at)
    return ws_proto

# Note: A proto_to_model function isn't strictly necessary for the registry's
# primary operations (add, get, list, delete output), but could be useful elsewhere.
# Adding a statement only requires the 'statement' string, not a full proto message.
# Getting/deleting requires the ID, which is passed directly.
