import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from robot_fleet.robots.robot_server_base import RobotServerBase
from robot_fleet.robots.models import TaskRequest, TaskResult
import asyncio


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
        logger.info(f"Task '{actual_task_description}' completed") 
        if actual_task_description == "Locobot searches for a cup in the living room.":
            return TaskResult(
                success=True,
                message=f"""Succeeded task!
                Task Given by Planner: '{actual_task_description}'
                Task Result Status by Robot: f'New info found: Found a cup in the living room table.'""",
                replan=True
            )
        else:
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

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_id", default="fake_echo_bot_01")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    fake_server_instance = FakeRobotServer(robot_id=args.robot_id, port=args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")