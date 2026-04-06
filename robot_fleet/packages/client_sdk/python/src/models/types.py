from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Literal


TaskStatus = Literal["unknown", "pending", "in_progress", "completed", "cancelled", "failed"]
RobotState = Literal["unknown", "registered", "deploying", "running", "error", "stopped"]


@dataclass(frozen=True)
class Robot:
    robot_id: str
    robot_type: str
    capabilities: list[str]
    status: Optional[RobotState] = None


@dataclass(frozen=True)
class Task:
    task_id: int
    description: str
    dependency_task_ids: list[int]
    status: TaskStatus
    goal_id: Optional[int] = None
    plan_id: Optional[int] = None
    robot_id: Optional[str] = None
    robot_type: Optional[str] = None
    result: Optional[str] = None


@dataclass(frozen=True)
class Plan:
    plan_id: int
    name: str
    description: str
    planning_strategy: int
    allocation_strategy: int
    task_ids: list[int]
    goal_ids: list[int]
    tasks: Optional[list[Task]] = None


JsonDict = dict[str, Any]

