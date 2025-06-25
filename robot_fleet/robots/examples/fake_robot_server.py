import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from robot_fleet.robots.robot_server_base import RobotServerBase
from robot_fleet.robots.models import TaskRequest, TaskResult
import asyncio
import argparse
from robot_fleet.robots.schema.yaml_validator import YAMLValidator


# ------------------------
# Fake Robot Server (based on fill_node_hardcode)
# ------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

class FakeRobotServer(RobotServerBase):
    def __init__(self, robot_id: str, port: int):
        super().__init__(robot_id, port)
        logger.info(f"FakeRobotServer '{robot_id}' initialized on port {port}.")

    async def _execute_task(self, task_request: TaskRequest) -> TaskResult:
        logger.info(f"Received task: {task_request.task_description}")
        actual_task_description = task_request.task_description.split("DO THE FOLLOWING TASK:")[1].strip()
        await asyncio.sleep(2)
        return TaskResult(
            success=True,
            message=f"""Succeeded task!
            Task Given by Planner: '{actual_task_description}'
            Task Result Status by Robot: f'Completed: {actual_task_description}'""",
            replan=False
        )

fake_server_instance = FakeRobotServer(robot_id="fake_echo_bot_01", port=8001)

@app.post("/do_task", response_model=TaskResult)
async def do_task(request: TaskRequest):
    logger.info(f"/do_task hit with: {request.task_description}")
    try:
        result = await fake_server_instance._execute_task(request)
        return result
    except Exception as e:
        logger.error(f"Error processing task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to robot YAML config file (e.g., nav.yaml)")
    parser.add_argument("--robot_id", required=False, help="Override robot_id (otherwise use value from YAML)")
    args = parser.parse_args()

    # Validate and load YAML config
    validator = YAMLValidator()
    config = validator.validate_file(args.config)
    robot_id = args.robot_id if args.robot_id else config["metadata"]["name"]
    port = int(config["taskServer"]["port"])

    fake_server_instance = FakeRobotServer(robot_id=robot_id, port=port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")