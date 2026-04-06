"""
Cost-based iterative task allocator.

Uses iterative assignment with LLM reasoning to minimize task switching
costs while respecting dependencies and robot capabilities.
"""

import json
import logging
import os
from openai import OpenAI
from typing import Dict, List

from ...base import BaseAllocator
from ....formats.formats import Allocation, RobotTask

logger = logging.getLogger(__name__)


class CostBasedAllocator(BaseAllocator):
    """
    Cost-based allocator that minimizes task switching costs.

    Uses iterative assignment with LLM reasoning to:
    - Respect task dependencies
    - Minimize switching costs between tasks
    - Consider robot capabilities and current state
    - Handle complex dependency graphs
    """

    async def allocate(self, plan_id: int) -> Dict[int, str]:
        """
        Allocate tasks using cost-based iterative assignment.

        Args:
            plan_id: ID of the plan to allocate tasks for

        Returns:
            Dict mapping task_id to assigned robot_id
        """
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("CostBasedAllocator: Starting cost-based allocation...")

        # 1. Fetch plan, tasks, robots, and world statements
        plan = await self.registry.get_plan(plan_id)
        if not plan:
            logger.error("No plan found for plan_id=%s", plan_id)
            return {}
        tasks = await self.registry.list_tasks()
        tasks = [task for task in tasks if task.plan_id == plan_id]
        logger.info("Fetched %s tasks for the plan", len(tasks))
        robots = await self.registry.list_robots()
        if not robots:
            logger.error("No robots found in the registry.")
            return {}
        logger.info("Fetched %s robots from the registry", len(robots))
        world_statements = await self.registry.list_world_statements()
        world_statements = [ws.statement for ws in world_statements]
        logger.info("Loaded %s world statements", len(world_statements))

        # 2. Build DAG structure
        task_map = {t.task_id: t for t in tasks}
        dependency_map = {}
        for t in tasks:
            # Use getattr to support both 'dependencies' and 'dependency_task_ids'
            deps = set(getattr(t, "dependency_task_ids", getattr(t, "dependencies", [])))
            dependency_map[t.task_id] = set(deps)

        # 2.5. Build initial available_queue: tasks with all dependencies satisfied
        assigned_tasks = set()
        robot_states = {r.robot_id: None for r in robots}  # robot_id -> last assigned task_id
        allocation_result: List[Dict] = []

        # 4. Iterative allocation
        while True:
            # Prune the DAG: only consider root nodes (tasks whose dependencies are all satisfied)
            available_tasks = [t for t in tasks if t.task_id not in assigned_tasks and all(dep in assigned_tasks for dep in dependency_map[t.task_id])]
            if not available_tasks:
                break
            robot_descriptions = [
                {
                    "robot_id": str(r.robot_id),
                    "capabilities": str(r.capabilities),
                    "robot_type": getattr(r, "robot_type", None),
                    "previous_task_id": str(robot_states[r.robot_id]) if robot_states[r.robot_id] else None
                }
                for r in robots
            ]
            task_descriptions = [
                {
                    "task_id": str(t.task_id),
                    "description": str(t.description),
                    "robot_type": getattr(t, "robot_type", None)
                }
                for t in available_tasks
            ]
            logger.debug("------------------------------------")
            logger.debug("Available tasks (root nodes): %s", available_tasks)
            logger.debug("Robot descriptions: %s", robot_descriptions)
            logger.debug("Task descriptions: %s", task_descriptions)

            # Load prompt templates (co-located with this allocator)
            system_prompt = self._load_prompt("system")
            user_prompt_template = self._load_prompt("user")

            # Format the user prompt with context
            user_prompt = user_prompt_template.format(
                world_statements=world_statements,
                available_tasks=json.dumps(task_descriptions),
                robot_descriptions=json.dumps(robot_descriptions)
            )

            response = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            allocation_content = response.choices[0].message.content
            logger.debug("LLM allocation response: %s", allocation_content)
            logger.debug("------------------------------------")
            try:
                robot_to_task = json.loads(allocation_content)
            except Exception as e:
                logger.error(f"Failed to parse LLM allocation response: {allocation_content}")
                raise
            assignments_this_round = 0
            assigned_this_round = set()
            # Convert mapping to RobotTask objects and update allocation_result
            for robot_id, task_id in robot_to_task.items():
                if task_id is None:
                    logger.info(f"Robot {robot_id} not assigned a task this round. Skipping.")
                    continue
                try:
                    task_id_int = int(task_id)
                except Exception:
                    logger.error(f"LLM returned non-integer task_id: {task_id} for robot {robot_id}")
                    continue
                allocation_result.append({
                    "robot_id": str(robot_id),
                    "task_id": task_id_int
                })
                # Mark task as assigned and update robot state
                assigned_tasks.add(task_id_int)
                assigned_this_round.add(task_id_int)
                robot_states[str(robot_id)] = task_id_int
                assignments_this_round += 1
            # Infinite loop protection: break if no assignments were made in this round
            if assignments_this_round == 0:
                logger.warning("No tasks could be assigned in this round. Breaking to avoid infinite loop.")
                break
        # Final check: assign any remaining unassigned tasks (fallback)
        unassigned = [tid for tid in task_map if tid not in assigned_tasks]
        if unassigned:
            logger.warning(f"Some tasks were still unassigned after LLM allocation: {unassigned}. Assigning to first capable robot as fallback.")
            for tid in unassigned:
                # Find first capable robot
                assigned = False
                for r in robots:
                    # You may want to check actual capability here
                    allocation_result.append({"robot_id": str(r.robot_id), "task_id": tid})
                    assigned_tasks.add(tid)
                    assigned = True
                    break
                if not assigned:
                    logger.error(f"No capable robot found for fallback assignment of task {tid}")
        logger.info("CostBasedAllocator: Final allocation: %s", allocation_result)
        # Update DB: assign each task to its robot
        for a in allocation_result:
            try:
                await self.registry.update_task(a["task_id"], robot_id=a["robot_id"])
                logger.info("Assigned robot %s to task %s", a['robot_id'], a['task_id'])
            except Exception as e:
                logger.error(f"Failed to assign robot {a['robot_id']} to task {a['task_id']}: {e}")
        # Always return Allocation object
        return Allocation(allocations=[RobotTask(**a) for a in allocation_result])