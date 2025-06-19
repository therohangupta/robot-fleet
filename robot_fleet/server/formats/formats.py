"""
Plan format definitions for the robot fleet scheduler.

This module contains Pydantic models that define the format for robot task plans.
All planners use the same output format for consistency.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class TaskPlanItem(BaseModel):
    """
    Represents a single task in a plan with its dependencies.
    """
    description: str
    goal_id: int
    dependency_task_ids: List[int]
    robot_type: Optional[str] = None

class Plan(BaseModel):
    """
    Unified plan format with a list of tasks and their dependencies.
    """
    tasks: List[TaskPlanItem]

# DAG-specific formats for more natural graph-based planning
class DAGNode(BaseModel):
    """
    Represents a node in a directed acyclic graph (DAG) of tasks.
    """
    id: str  # e.g., "node0", "node1" 
    description: str
    goal_id: int
    depends_on: List[str]  # list of node IDs this depends on
    robot_type: Optional[str] = None

class DAGPlan(BaseModel):
    """
    DAG-based plan format with nodes and explicit dependencies.
    """
    nodes: List[DAGNode]

class AllocatedDAGNode(BaseModel):
    """
    Represents a node in a directed acyclic graph (DAG) of tasks.
    """
    task_id: int  # e.g., "node0", "node1" 
    description: str
    goal_id: int
    robot_id: str
    depends_on: List[int]  # list of node IDs this depends on

class AllocatedDAGPlan(BaseModel):
    """
    DAG-based plan format with nodes and explicit dependencies.
    """
    nodes: List[AllocatedDAGNode]

class RobotTask(BaseModel):
    """
    Allocation format for task-to-robot assignments.
    """
    task_id: int
    robot_id: str
class Allocation(BaseModel):
    allocations: List[RobotTask]


    
