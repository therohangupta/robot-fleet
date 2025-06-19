import asyncio
import os
import sys
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP
from robot_fleet.robots.robot_server_base import RobotServerBase, TaskResult

class NavigationStack:
    async def move_to(self, location: list):
        print(f"Simulating navigation to location {location}")
        await asyncio.sleep(2)  

class NavRobot(RobotServerBase):
    def __init__(self, robot_id: str, port: int):
        super().__init__(robot_id=robot_id, port=port)

        self.nav = NavigationStack()

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
        Handles pure navigation tasks.
        Expects:
          task_description contains “navigate”
          parameters contains “target_location”: [x, y, z]
        """
        try:
            desc = task_description.lower()
            if "navigate" in desc:
                if "target_location" not in parameters:
                    return TaskResult(
                        False,
                        "Missing required parameter: target_location"
                    )

                target = parameters["target_location"]
                await self.nav.move_to(target)

                return TaskResult(
                    True,
                    "Navigation completed successfully",
                    details={"final_location": target}
                )

            return TaskResult(
                False,
                "Unsupported task. This robot only supports navigation."
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
    print(f"Starting navigation robot server with ID {robot_id} on port {port}", file=sys.stderr)
    robot = NavRobot(robot_id, port)
    robot.run()

if __name__ == "__main__":
    main()
