"""
LLM-based task allocator.

Uses OpenAI's GPT-4 to intelligently allocate tasks to robots
based on their capabilities, task requirements, and current state.
"""

import json
import os
from openai import OpenAI
from typing import Dict

from robot_fleet.server.allocators.base import BaseAllocator
from robot_fleet.server.formats.formats import Allocation, RobotTask


class LLMAllocator(BaseAllocator):
    """
    LLM-based allocator that uses GPT-4 for intelligent task allocation.

    Considers robot capabilities, task requirements, robot types,
    and world statements to make optimal assignments.
    """

    async def allocate(self, plan_id: int) -> Dict[int, str]:
        """
        Allocate tasks using LLM reasoning.

        Args:
            plan_id: ID of the plan to allocate tasks for

        Returns:
            Dict mapping task_id to assigned robot_id
        """
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print(f"Starting allocation for plan_id: {plan_id}")
        plan = await self.registry.get_plan(plan_id)
        if not plan:
            print(f"No plan found for plan_id={plan_id}")
            return {}
        tasks = await self.registry.list_tasks()
        tasks = [task for task in tasks if task.plan_id == plan_id]
        print(f"Fetched {len(tasks)} tasks for the plan")
        robots = await self.registry.list_robots()
        if not robots:
            print("No robots found in the registry.")
            return {}
        print(f"Fetched {len(robots)} robots from the registry")
        world_statements = await self.registry.list_world_statements()
        world_statements = [ws.statement for ws in world_statements]
        print(f"Loaded {len(world_statements)} world statements")
        print("tasks: ", tasks)
        task_descriptions = [
            {
                "task_id": str(t.task_id),
                "description": str(t.description),
                "goal_id": str(t.goal_id),
                "dependencies": [str(dep) for dep in getattr(t, "dependency_task_ids", [])],
                "robot_type": getattr(t, "robot_type", None)
            }
            for t in tasks
        ]
        robot_descriptions = [
            {
                "robot_id": str(r.robot_id),
                "robot_type": getattr(r, "robot_type", None),
                "capabilities": str(r.capabilities),
            }
            for r in robots
        ]

        # Load prompt templates (co-located with this allocator)
        system_prompt = self._load_prompt("system")
        user_prompt_template = self._load_prompt("user")

        # Format the user prompt with context
        user_prompt = user_prompt_template.format(
            world_statements=world_statements,
            task_descriptions=json.dumps(task_descriptions),
            robot_descriptions=json.dumps(robot_descriptions)
        )

        # Store allocation prompts and artifacts
        self.allocation_prompts = {
            "system": system_prompt,
            "user": user_prompt
        }

        self.allocation_artifacts = {
            "task_descriptions": task_descriptions,
            "robot_descriptions": robot_descriptions,
            "world_statements": world_statements
        }

        print("Prepared prompts for LLM")
        print("Calling OpenAI API...")
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=4000,
            response_format=Allocation,
        )
        allocation_content = response.choices[0].message.content
        try:
            allocation_dict = json.loads(allocation_content)
            print(f"Generated allocation:\n{allocation_dict}")
            allocation_obj = Allocation(allocations=[RobotTask(**rt) for rt in allocation_dict['allocations']])
            print(f"Successfully parsed allocation: {allocation_obj}")
        except Exception as e:
            print(f"Failed to parse LLM allocation response: {e}\nRaw response: {allocation_content}")
            return {}
        # Store allocation result in artifacts
        self.allocation_artifacts["final_allocation"] = allocation_obj.dict()

        # Add logs
        self.server_logs = [
            f"LLM allocator started for plan {plan_id}",
            f"Processed {len(task_descriptions)} tasks and {len(robot_descriptions)} robots",
            f"Generated allocation with {len(allocation_obj.allocations)} assignments",
            f"Allocation completed successfully"
        ]

        print("Updating tasks in database with assigned robots...")
        for robot_task in allocation_obj.allocations:
            try:
                await self.registry.update_task(robot_task.task_id, robot_id=robot_task.robot_id)
                print(f"Assigned robot {robot_task.robot_id} to task {robot_task.task_id}")
                self.server_logs.append(f"Assigned robot {robot_task.robot_id} to task {robot_task.task_id}")
            except Exception as e:
                error_msg = f"Failed to assign robot {robot_task.robot_id} to task {robot_task.task_id}: {e}"
                print(error_msg)
                self.server_logs.append(error_msg)
        print(f"Task allocation complete. Final allocation: {allocation_obj}")
        return allocation_obj