from abc import ABC, abstractmethod

from ..models import TaskRequest, TaskResult


class RobotServerBase(ABC):
    def __init__(self, robot_id: str, port: int):
        self.robot_id = robot_id
        self.port = port
        self.setup_tools()

    def setup_tools(self):
        """Setup the basic tools that all robots must expose."""
        pass

    @abstractmethod
    async def _execute_task(self, task_request: TaskRequest) -> TaskResult:
        """Robot-specific implementation of task execution."""
        raise NotImplementedError()

