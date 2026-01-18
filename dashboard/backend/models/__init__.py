"""
Pydantic models for request/response validation.

Models are split into:
- requests.py: Input models for API endpoints
- responses.py: Output models for API responses
"""

from .requests import (
    RobotRegistration,
    RobotInstanceCreate,
    GoalCreate,
    PlanCreate,
    ManualPlanCreate,
    ManualTaskDefinition,
    TaskCreate,
    WorldStatementCreate,
)

from .responses import (
    RobotResponse,
    GoalResponse,
    PlanResponse,
    TaskResponse,
    WorldStatementResponse,
    EmbodimentResponse,
)

__all__ = [
    # Requests
    "RobotRegistration",
    "RobotInstanceCreate",
    "GoalCreate",
    "PlanCreate",
    "ManualPlanCreate",
    "ManualTaskDefinition",
    "TaskCreate",
    "WorldStatementCreate",
    # Responses
    "RobotResponse",
    "GoalResponse",
    "PlanResponse",
    "TaskResponse",
    "WorldStatementResponse",
    "EmbodimentResponse",
]
