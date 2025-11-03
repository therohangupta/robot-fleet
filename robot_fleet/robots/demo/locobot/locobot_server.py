"""Navigation robot implementation"""
import sys
import os
three_levels_up = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(three_levels_up)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(sys.path)
import asyncio
import json
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from robot_fleet.robots.robot_server_base import RobotServerBase
from robot_fleet.robots.models import TaskRequest, TaskResult
import openai
from locobot_tools import TOOL_DESCRIPTIONS, TOOL_IMPORT_STRING
from dotenv import load_dotenv
import subprocess
import tempfile
import stat
import pdb
import ast


# hardcoded for our .env file — change this path to match your .env file
load_dotenv(dotenv_path="/home/elle/elle_ws/multirobot-task/.env")  # This will look for a .env file in the project root directory


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=openai_api_key)

app = FastAPI()

# ✅ Backport replacement for asyncio.to_thread. Necessary for asyncio for python 3.8, which is necessary for ros1 noetic envs. 
async def to_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

class Locobot(RobotServerBase):
    def __init__(self, robot_id: str, port: int = 5001):
        super().__init__(robot_id, port)

    async def _execute_task(self, task_description: str) -> TaskResult:
        actual_task_description = task_description.split("DO THE FOLLOWING TASK:")[1].strip()
        try:
            # currently tries to complete the task 3 times
            for i in range(3):
                if i > 0:
                    print("___________RETRYING THE TASK___________")
                print(f"Executing task: {task_description}")

                planning_response = await to_thread(
                    openai_client.chat.completions.create,
                    model="gpt-4o", # change your model if needed
                    messages=[
                        {"role": "system", "content":
                        "You are a Locobot robot high level to low level task planner. Create a high-level plan for the given task using the available functions/tools. "
                        "Another AI agent will be writing the python code to execute the plan, so make sure the plan is written in a way that is easy to translate into code."
                        "Think about the best way to accomplish the task if you were the robot with the available tools."
                        "Write the plan out step by step"
                        "Only describe tools to the coding agent with the name of the tool. Never ever ever under any circumstances describe them with the functions namespace."
                        "Do not use multi_tool_use.parallel. Parallel tool calls are strictly forbidden in your plan."
                        "Good example: navigate_loc , Terrible example: functions.navigate_loc"
                        "Make sure the plan for this task starts at the end state of the most recent succeed task in the above list."
                        },
                        {"role": "user", "content": f"Task: {task_description}"}
                    ],
                    tools=TOOL_DESCRIPTIONS
                )
                plan = planning_response.choices[0].message.content

                print(f"Generated plan:\n{plan}")
                plan = plan.replace("functions.", "")
                
                execution_response = await to_thread(
                    openai_client.chat.completions.create,
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a robot control system."
                        "You will be developing a python script that ensures a robot can execute the plan"
                        "The functions you have available are the tools that the robot has. Ensure that you call each function using the parameters without a json input like this navigate(1, 2, 0.5)"
                        "I want you to write a python file with the general following structure to ensure the robot is able to complete the task effectively."
                        " - a while loop that will make sure the robot is alive, and that robot continues to execute the plan if the task is not complete."
                        " - a series of tool calls within if branching based on the plan."
                        " - whenever you are returning in the code, always return the value of return_status(msg) to indicate the success or failure status of the task."
                        "the ONLY functions you can call are the tools that are available to the robot. If you call any other functions the program will crash!"
                        "The ONLY exception to this rule is if you would like to implement your own helper functions"
                        "Do not import anything as that will be all be added before the code you generate."
                        "If you import anything the code will crash, you must only write a main."
                        "Never ever ever under any circumstances uses functions namespace! You must always call each function standalone. Example: function(param1, param2). Bad example: functions.function(param1, param2). Using the 'functions.' namespace is absolutely forbidden."
                        "Never use any parallel tool calls. Parallelism is already implemented in the functions you are provided, you don't need to call them in parallel ever again. Parallel tool calls are strictly and absolutely forbidden."
                        "Don't use a json as input to any function list the parameters out like a normal python function"
                        "Make sure that there is a proper main set up so that the file is executable as is. In the code, do not or run the main(), just define it."
                        "Only return the python code, no other text or comments."
                        "Write the bare minimum code possible to complete the task. Don't make the code do anything else but what the task asks for."
                        "Be verbose by using the log(statement) tool instead of print(). Make log statements after every function call and make sure it includes the status returned from the tool and any other relevant info."
                        },
                        {"role": "user", "content": f"Plan:\n{plan}\n\nExecute this plan using the available tools."}
                    ],
                    tools=TOOL_DESCRIPTIONS,
                )
                code = execution_response.choices[0].message.content
                code = code.strip()
                code = code.replace("```python", "")
                code = code.replace("```", "")
                code = code.replace('functions.', '')
                code = code.replace('namespace functions', '')
                code = code.split("if __name__ == ")[0]
                print(f"Generated code:\n{code}")
                
                injection = "import sys\nimport os\nsys.path.append(os.path.dirname(os.path.abspath(__file__)))\n\n"+ TOOL_IMPORT_STRING+"\n\n"
                main_injection = """\n\nif __name__ == "__main__":
    result = main()
    print("__DONE WITH TASK__")
    print(result)
                """
                code = injection + code + main_injection

                # Write code to a temporary Python file
                project_dir = os.path.dirname(__file__)
                with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.py', dir=project_dir) as f:
                    script_path = f.name
                    f.write(code)

                # Make it executable
                os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)

                # Run the script
                result = subprocess.run(
                    ["python3", script_path],
                    capture_output=True,
                    text=True
                )

                print("Execution stdout:\n", result.stdout)
                print("Execution stderr:\n", result.stderr)

                # Extract the result information and capture (status, msg, replan) tuple
                task_result_string = result.stdout.split("__DONE WITH TASK__")[1]
                task_result_string = task_result_string.strip()
                task_result = ast.literal_eval(task_result_string)
                task_result_status = task_result[0]
                task_result_msg = task_result[1]
                task_result_replan_flag = task_result[2]
                
                if task_result_status == True:
                    message = f"""Succeeded task!
                            Task Given by Planner: '{actual_task_description}'
                            Task Result Status by Robot: '{task_result_msg}'"""
                    return TaskResult(
                        success=True,
                        message=message,
                        replan=task_result_replan_flag
                    )

            # after some number of attempts at the task, if it still fails, then the task fails and doesn't call replan
            return TaskResult(
                success=False,
                message=f"""Failed task: 
                            Original Task: '{actual_task_description}'
                            Task Result Message: '{task_result_msg}'""",
                replan=False
            )

        finally:
            try:
                if script_path and os.path.exists(script_path):
                    os.remove(script_path)
            except Exception as cleanup_error:
                print(f"Warning: Failed to delete temp file {script_path}: {cleanup_error}")

locobot_robot = Locobot(robot_id=os.getenv("ROBOT_ID", "locobot_demo"), port=int(os.getenv("PORT", 5001)))

@app.post("/do_task")
async def do_task(request: TaskRequest):
    result = await locobot_robot._execute_task(request.task_description)
    return result

# pip install openai fastapi dotenv uvicorn ultralytics
# To run: cd /root/TEMP/telemoma & uvicorn locobot_server:app --host 0.0.0.0 --port 5001
