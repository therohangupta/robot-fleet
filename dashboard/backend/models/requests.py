"""
Request models for API input validation.

These Pydantic models define the expected structure of request bodies
for POST/PUT/PATCH endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Robot Models
# =============================================================================

class RobotRegistration(BaseModel):
    """Legacy robot registration request (deprecated, use RobotInstanceCreate)."""
    config_path: str = Field(..., description="Path to robot YAML configuration")
    robot_id: Optional[str] = Field(None, description="Custom robot ID (auto-generated if not provided)")


class RobotInstanceCreate(BaseModel):
    """
    Request to register a new robot instance.
    
    Combines an embodiment (YAML config) with specific network settings
    to create a deployable robot instance.
    """
    config_path: str = Field(
        ..., 
        description="Path to robot YAML config (relative to project root or absolute)"
    )
    robot_id: str = Field(
        ..., 
        description="Unique identifier for this robot instance"
    )
    host: str = Field(
        default="localhost",
        description="Hostname or IP where the robot's task server is running"
    )
    port: int = Field(
        default=8001,
        description="Port number for the robot's task server"
    )


# =============================================================================
# Goal Models
# =============================================================================

class GoalCreate(BaseModel):
    """Request to create a new goal."""
    description: str = Field(
        ..., 
        min_length=1,
        description="Natural language description of what should be accomplished"
    )


# =============================================================================
# Plan Models
# =============================================================================

class PlanCreate(BaseModel):
    """
    Request to create a plan using automated planning and allocation.
    """
    planning_strategy: int = Field(
        ...,
        description="Planning strategy enum value: 1=monolithic, 2=dag, 3=big_dag, 4=manual"
    )
    allocation_strategy: int = Field(
        ...,
        description="Allocation strategy enum value: 1=lp, 2=llm, 3=cost_based, 4=none"
    )
    goal_ids: List[int] = Field(
        ...,
        min_length=1,
        description="List of goal IDs this plan should accomplish"
    )
    name: str = Field(
        ...,
        description="User-defined name for the plan"
    )
    description: str = Field(
        ...,
        description="User-defined description for the plan"
    )


class ManualTaskDefinition(BaseModel):
    """
    Definition of a task for manual plan creation.
    
    Uses temp_id for referencing dependencies between tasks before
    real task IDs are assigned by the database.
    """
    temp_id: str = Field(
        ..., 
        description="Temporary ID for referencing this task in dependencies"
    )
    description: str = Field(
        ..., 
        min_length=1,
        description="What this task should accomplish"
    )
    goal_id: int = Field(
        ..., 
        description="Which goal this task contributes to"
    )
    robot_id: Optional[str] = Field(
        None, 
        description="Specific robot to assign (leave empty for unallocated)"
    )
    robot_type: Optional[str] = Field(
        None, 
        description="Required robot type/capability for this task"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of temp_ids this task depends on"
    )


class ManualPlanCreate(BaseModel):
    """
    Request to create a plan with manually defined tasks.
    
    Bypasses the automated planner - you define the exact task DAG.
    Plan goal_ids are derived from the tasks' goal_ids.
    """
    tasks: List[ManualTaskDefinition] = Field(
        ...,
        min_length=1,
        description="List of tasks forming the plan's DAG"
    )
    name: str = Field(
        ...,
        description="User-defined name for the plan"
    )
    description: str = Field(
        ...,
        description="User-defined description for the plan"
    )


# =============================================================================
# Task Models
# =============================================================================

class TaskCreate(BaseModel):
    """Request to create an individual task."""
    description: str = Field(..., description="Task description")
    goal_id: int = Field(..., description="Goal this task belongs to")
    plan_id: Optional[int] = Field(None, description="Plan this task belongs to")
    robot_id: Optional[str] = Field(None, description="Assigned robot ID")
    robot_type: Optional[str] = Field(None, description="Required robot type")
    dependency_task_ids: List[int] = Field(
        default_factory=list,
        description="Task IDs this task depends on"
    )


# =============================================================================
# World State Models
# =============================================================================

class WorldStatementCreate(BaseModel):
    """Request to add a world statement."""
    statement: str = Field(
        ..., 
        min_length=1,
        description="Fact about the world state (e.g., 'Object A is on Table B')"
    )
