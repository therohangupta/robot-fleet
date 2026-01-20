"""
Response models for API output validation.

These Pydantic models define the structure of API responses,
providing documentation and type safety for frontend consumers.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# Robot Models
# =============================================================================

class TaskServerInfo(BaseModel):
    """Information about a robot's task server."""
    host: str = Field(..., description="Hostname or IP address")
    port: int = Field(..., description="Port number")


class RobotResponse(BaseModel):
    """Robot information returned by the API."""
    robot_id: str = Field(..., description="Unique robot identifier")
    robot_type: str = Field(..., description="Robot type (from YAML metadata)")
    capabilities: List[str] = Field(default_factory=list, description="Robot capabilities")
    status: str = Field(default="available", description="Current robot status")
    task_server_info: Optional[TaskServerInfo] = Field(
        None, 
        description="Network location of robot's task server"
    )
    current_task_id: Optional[int] = Field(
        None, 
        description="ID of task currently being executed"
    )

    class Config:
        from_attributes = True


# =============================================================================
# Goal Models
# =============================================================================

class GoalResponse(BaseModel):
    """Goal information returned by the API."""
    goal_id: int = Field(..., description="Unique goal identifier")
    description: str = Field(..., description="Goal description")
    status: str = Field(default="pending", description="Goal status")
    task_ids: List[int] = Field(default_factory=list, description="IDs of tasks associated with this goal")
    created_at: Optional[str] = Field(None, description="Creation timestamp")

    class Config:
        from_attributes = True


# =============================================================================
# Task Models
# =============================================================================

class TaskResponse(BaseModel):
    """Task information returned by the API."""
    task_id: int = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Task description")
    goal_id: int = Field(..., description="Associated goal ID")
    plan_id: Optional[int] = Field(None, description="Associated plan ID")
    robot_id: Optional[str] = Field(None, description="Assigned robot ID")
    robot_type: Optional[str] = Field(None, description="Required robot type")
    status: str = Field(default="pending", description="Execution status")
    dependency_task_ids: List[int] = Field(
        default_factory=list, 
        description="IDs of tasks this depends on"
    )
    result: Optional[str] = Field(None, description="Execution result/output")

    class Config:
        from_attributes = True


# =============================================================================
# Plan Models
# =============================================================================

class PlanResponse(BaseModel):
    """Plan information returned by the API."""
    plan_id: int = Field(..., description="Unique plan identifier")
    name: str = Field(..., description="User-defined plan name")
    description: str = Field(..., description="User-defined plan description")
    goal_ids: List[int] = Field(default_factory=list, description="Goals this plan addresses")
    task_ids: List[int] = Field(default_factory=list, description="Tasks in this plan")
    tasks: List[TaskResponse] = Field(default_factory=list, description="Full task objects")
    planning_strategy: int = Field(..., description="Strategy used for planning (enum value)")
    allocation_strategy: int = Field(..., description="Strategy used for allocation (enum value)")
    allocation_status: str = Field(
        default="unknown",
        description="'unallocated', 'partially_allocated', or 'fully_allocated'"
    )
    execution_status: str = Field(
        default="not_executed",
        description="'not_executed', 'executing', 'completed', or 'failed'"
    )
    status: str = Field(default="created", description="Plan execution status")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    planning_prompts: Optional[Dict[str, str]] = Field(None, description="Prompts used for planning")
    allocation_prompts: Optional[Dict[str, str]] = Field(None, description="Prompts used for allocation")
    planning_artifacts: Optional[Dict] = Field(None, description="Artifacts from planning process")
    allocation_artifacts: Optional[Dict] = Field(None, description="Artifacts from allocation process")
    server_logs: Optional[str] = Field(None, description="Server-side logs from planning/allocation")
    dag_structure: Optional[Dict] = Field(None, description="DAG structure for visualization")

    class Config:
        from_attributes = True


# =============================================================================
# World State Models
# =============================================================================

class WorldStatementResponse(BaseModel):
    """World statement information returned by the API."""
    id: str = Field(..., description="Statement identifier")
    statement: str = Field(..., description="The world state fact")
    created_at: Optional[str] = Field(None, description="Creation timestamp")

    class Config:
        from_attributes = True


# =============================================================================
# Embodiment Models
# =============================================================================

class EmbodimentResponse(BaseModel):
    """Robot embodiment (type template) information."""
    name: str = Field(..., description="Embodiment name")
    description: str = Field(default="", description="Embodiment description")
    capabilities: List[str] = Field(default_factory=list, description="Available capabilities")
    default_port: int = Field(default=8000, description="Default task server port")
    config_path: str = Field(..., description="Path to YAML configuration")
    container_image: str = Field(default="", description="Docker image if containerized")

    class Config:
        from_attributes = True
