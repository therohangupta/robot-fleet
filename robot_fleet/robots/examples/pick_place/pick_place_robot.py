import asyncio
import os
import sys
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP
from robot_fleet.robots.robot_server_base import RobotServerBase, TaskResult

# Simulated robot arm only
class RobotArm:
    async def pick(self, object_id: str):
        print(f"Simulating picking object {object_id}")
        await asyncio.sleep(1)  # Simulate pick time
        
    async def place(self):
        print("Simulating placing object")
        await asyncio.sleep(1)  # Simulate place time

class PickPlaceRobot(RobotServerBase):
    def __init__(self, robot_id: str, port: int):
        super().__init__(robot_id=robot_id, port=port)

        # only arm, no navigation
        self.arm = RobotArm()

        self.mcp = FastMCP(
            f"robot_{robot_id}",
            host="0.0.0.0",
            port=port
        )
        self.setup_tools()

    async def _execute_task(self,
                            task_description: str,
                            parameters: Dict[str, Any]) -> TaskResult:
        """
        Handles pick+place tasks without any navigation.
        Expects:
          - task_description containing "pick" and "place"
          - parameters including "object_id"
        """
        try:
            desc = task_description.lower()
            if "pick" in desc and "place" in desc:
                if "object_id" not in parameters:
                    return TaskResult(
                        False,
                        "Missing required parameter: object_id"
                    )

                obj = parameters["object_id"]
                await self.arm.pick(obj)
                await self.arm.place()

                return TaskResult(
                    True,
                    "Pick and place completed successfully",
                    details={"object_id": obj}
                )

            return TaskResult(
                False,
                "Unsupported task. This robot only supports combined pick+place operations."
            )

        except Exception as e:
            return TaskResult(
                False,
                f"Task failed: {e}",
                details={"error_type": type(e).__name__}
            )

def main():
    robot_id = os.environ.get("ROBOT_ID")
    if not robot_id:
        print("Error: ROBOT_ID environment variable must be set", file=sys.stderr)
        sys.exit(1)

    port = int(os.environ.get("PORT", "5001"))
    print(f"Starting pick-place robot server with ID {robot_id} on port {port}", file=sys.stderr)
    robot = PickPlaceRobot(robot_id, port)
    robot.run()

if __name__ == "__main__":
    main()
