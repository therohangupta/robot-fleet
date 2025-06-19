from abc import ABC, abstractmethod
from robot_fleet.robots.models import TaskRequest, TaskResult

class RobotServerBase(ABC):
    def __init__(self, robot_id: str, port: int):
        self.robot_id = robot_id
        self.port = port
        self.setup_tools()

    def setup_tools(self):
        """Setup the basic tools that all robots must expose"""
        # This can be overridden by subclasses if needed
        pass

    @abstractmethod
    async def _execute_task(self, task_description: str) -> TaskResult:
        """
        Robot-specific implementation of task execution.
        Each robot type implements this differently based on their capabilities.
        """
        pass