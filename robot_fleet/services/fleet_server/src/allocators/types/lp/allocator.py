"""
Linear programming-based task allocator.

Uses PuLP to solve an optimization problem for balanced task allocation
across robots, considering capabilities, robot types, and task requirements.
"""

import logging
import pulp
from typing import Dict

from ...base import BaseAllocator
from ....formats.formats import Allocation, RobotTask

logger = logging.getLogger(__name__)


class LPAllocator(BaseAllocator):
    """
    Linear programming-based allocator that minimizes maximum robot load.

    Uses integer linear programming to find an optimal assignment that:
    - Assigns each task to exactly one robot
    - Respects robot capabilities and types
    - Balances workload across robots (minimizes max tasks per robot)
    """

    async def allocate(self, plan_id: int) -> Dict[int, str]:
        """
        Allocate tasks using linear programming optimization.

        Args:
            plan_id: ID of the plan to allocate tasks for

        Returns:
            Dict mapping task_id to assigned robot_id
        """
        # Linear programming-based allocation using pulp
        import pulp
        logger.info("LPAllocator: Starting linear programming allocation...")

        # 1. Fetch tasks for the plan
        plan = await self.registry.get_plan(plan_id)
        if not plan:
            logger.error("No plan found for plan_id=%s", plan_id)
            return {}
        tasks = await self.registry.list_tasks()
        tasks = [task for task in tasks if task.plan_id == plan_id]
        logger.info("Fetched %s tasks for the plan", len(tasks))

        # 2. Fetch robots and their capabilities
        robots = await self.registry.list_robots()
        if not robots:
            logger.error("No robots found in the registry.")
            return {}
        logger.info("Fetched %s robots from the registry", len(robots))

        # 3. Build capability map and infer task capabilities
        robot_caps = {r.robot_id: set(r.capabilities) for r in robots}
        task_caps = {}

        # Find a fallback robot (one with most capabilities)
        fallback_robot = max(robots, key=lambda r: len(r.capabilities))

        logger.debug("Selected fallback robot %s with capabilities: %s", fallback_robot.robot_id, fallback_robot.capabilities)

        for t in tasks:
            # If task has no required capabilities, infer them from description
            if not getattr(t, 'required_capabilities', []):
                caps = set()
                desc = t.description.lower()
                # Only add navigate if it's explicitly about navigation
                if 'navigate' in desc or 'move' in desc or 'go' in desc:
                    caps.add('navigate')
                # Only add pick if it's explicitly about picking up
                if 'pick' in desc:
                    caps.add('pick')
                # Only add place if it's explicitly about placing
                if 'place' in desc:
                    caps.add('place')
                if 'carry' in desc:
                    caps.add('carry')
                # Only add explore if it's explicitly about exploration
                if 'explore' in desc and 'area' in desc:
                    caps.add('explore_known_locations')
                # Only add capture_image if it's explicitly about capturing images
                if 'image' in desc or 'picture' in desc:
                    caps.add('capture_image')

                # If no capabilities were inferred, use fallback robot's capabilities
                if not caps:
                    logger.debug("Task %s (%s): No specific capabilities found, using fallback robot capabilities", t.task_id, t.description)
                    caps = set(fallback_robot.capabilities)

                task_caps[t.task_id] = caps
                logger.debug("Task %s (%s): Inferred capabilities: %s", t.task_id, t.description, caps)
            else:
                task_caps[t.task_id] = set(t.required_capabilities)
                logger.debug("Task %s (%s): Using explicit capabilities: %s", t.task_id, t.description, task_caps[t.task_id])

        # 4. Build LP problem
        prob = pulp.LpProblem("TaskAllocation", pulp.LpMinimize)
        # Decision vars: x_{t,r} = 1 if task t assigned to robot r
        x = pulp.LpVariable.dicts(
            "assign",
            ((t.task_id, r.robot_id) for t in tasks for r in robots),
            cat=pulp.LpBinary
        )
        # Objective: minimize max load (number of tasks per robot)
        # Introduce variable for max load
        max_load = pulp.LpVariable("max_load", lowBound=0, cat=pulp.LpInteger)
        # Each task assigned to exactly one robot
        for t in tasks:
            prob += pulp.lpSum([x[(t.task_id, r.robot_id)] for r in robots]) == 1, f"OneRobotPerTask_{t.task_id}"
        # Only assign if robot has all required capabilities and matches robot_type if specified
        for t in tasks:
            task_robot_type = getattr(t, "robot_type", None)
            for r in robots:
                robot_actual_type = getattr(r, "robot_type", None)
                # Constraint 1: Robot type matching (if task specifies a type)
                type_match = True # Assume match if task doesn't specify a type or robot doesn't have a type
                if task_robot_type and robot_actual_type:
                    if task_robot_type != robot_actual_type:
                        type_match = False
                elif task_robot_type and not robot_actual_type:
                    # Task specifies a type, but robot has no type defined. Consider this a mismatch for typed tasks.
                    type_match = False

                # Constraint 2: Capability matching
                capability_match = task_caps[t.task_id].issubset(robot_caps[r.robot_id])

                if not type_match or not capability_match:
                    # If task has no specific capabilities (and thus using fallback), and no specific robot_type,
                    # allow assignment only to fallback robot (original logic for this case)
                    # This condition needs to be carefully placed. If a robot_type IS specified, it should take precedence.
                    if not task_caps[t.task_id] and not task_robot_type: # Task has no specific caps AND no specific type
                        if r.robot_id != fallback_robot.robot_id:
                            prob += x[(t.task_id, r.robot_id)] == 0, f"FallbackOnly_{t.task_id}_{r.robot_id}"
                        # else: allow assignment to fallback if it got here (type_match and capability_match were true for fallback)
                    else:
                        # If type mismatch OR capability mismatch (and not the special fallback case above)
                        prob += x[(t.task_id, r.robot_id)] == 0, f"Constraint_{t.task_id}_{r.robot_id}"

        # Max load constraint
        for r in robots:
            prob += pulp.lpSum([x[(t.task_id, r.robot_id)] for t in tasks]) <= max_load, f"MaxLoad_{r.robot_id}"
        prob += max_load  # Objective: minimize max_load


        # 5. Solve
        status = prob.solve()
        if pulp.LpStatus[status] != "Optimal":
            logger.error("No feasible allocation found.")
            return {}
        # 6. Build allocation result
        allocation = []
        for t in tasks:
            for r in robots:
                if pulp.value(x[(t.task_id, r.robot_id)]) == 1:
                    allocation.append({"task_id": t.task_id, "robot_id": r.robot_id})
                    break
        logger.info("LPAllocator: Allocation result: %s", allocation)
        # Update DB
        for a in allocation:
            try:
                await self.registry.update_task(a["task_id"], robot_id=a["robot_id"])
                logger.info("Assigned robot %s to task %s", a['robot_id'], a['task_id'])
            except Exception as e:
                logger.error("Failed to assign robot %s to task %s: %s", a['robot_id'], a['task_id'], e)
        # Return as Allocation object (for compatibility)
        allocation_obj = Allocation(allocations=[RobotTask(**a) for a in allocation])
        return allocation_obj