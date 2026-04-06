from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union


@dataclass(frozen=True)
class Connected:
    type: Literal["connected"]
    message: str


@dataclass(frozen=True)
class Invalidate:
    type: Literal["invalidate"]
    queries: list[str]
    timestamp: float


@dataclass(frozen=True)
class TasksUpdate:
    type: Literal["tasks_update"]
    plan_id: int
    tasks: list[dict[str, Any]]


GatewayRealtimeMessage = Union[Connected, Invalidate, TasksUpdate, dict[str, Any]]

