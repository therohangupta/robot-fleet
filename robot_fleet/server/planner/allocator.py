import json
from abc import ABC, abstractmethod
from typing import Dict
import logging
from robot_fleet.robots.registry.instance_registry import RobotInstanceRegistry
from .formats.formats import Allocation, RobotTask
from typing import Optional
logger = logging.getLogger(__name__)

class AllocatorBase(ABC):
    """Abstract base class for all allocators."""
    def __init__(self, db_url: Optional[str] = None):
        if not db_url:
            db_url = "postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet"
            print(f"Using default PostgreSQL database: {db_url}")
        else:
             print(f"Using database: {db_url}")
        self.registry = RobotInstanceRegistry(db_url)
        # Load world statements on initialization
        logger.info(f"Initialized {self.__class__.__name__}.")

    @abstractmethod
    async def allocate(self, plan_id: int) -> Dict[int, str]:
        """Allocate tasks from the given plan_id to robots. Returns a mapping from task_id to assigned robot_id."""
        pass

class LLMAllocator(AllocatorBase):
    async def allocate(self, plan_id: int) -> Dict[int, str]:
        import os
        from openai import OpenAI
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
        prompt = (
            "You are an expert multi-robot task allocator. "
            "Given the following list of tasks (each with an optional suggested 'robot_type') and robots (each with a 'robot_type' and 'capabilities'), assign each task to the most suitable robot. "
            "If a task specifies a 'robot_type', prioritize assigning it to a robot of that type, provided its capabilities are also compatible. "
            "If no 'robot_type' is specified for a task, or if no robot of the specified type is suitable/available, assign based on capabilities. "
            "Ensure tasks are assigned to robots with compatible capabilities. "
            "Optimize for goal completion and speed of goal completion. "
            "Consider robot capabilities, robot types, and any world statements. "
            "Output a JSON mapping of task_id to robot_id.\n"
            f"World Statements: {world_statements}\n"
            f"Tasks: {json.dumps(task_descriptions)}\n"
            f"Robots: {json.dumps(robot_descriptions)}\n"
            "Respond with a JSON object: {task_id: robot_id, ...}"
        )
        print("Prepared prompt for LLM")
        print("Calling OpenAI API...")
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for robot fleet task allocation."},
                {"role": "user", "content": prompt}
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
        print("Updating tasks in database with assigned robots...")
        for robot_task in allocation_obj.allocations:
            try:
                await self.registry.update_task(robot_task.task_id, robot_id=robot_task.robot_id)
                print(f"Assigned robot {robot_task.robot_id} to task {robot_task.task_id}")
            except Exception as e:
                print(f"Failed to assign robot {robot_task.robot_id} to task {robot_task.task_id}: {e}")
        print(f"Task allocation complete. Final allocation: {allocation_obj}")
        return allocation_obj

class LPAllocator(AllocatorBase):
    async def allocate(self, plan_id: int) -> Dict[int, str]:
        # Linear programming-based allocation using pulp
        import pulp
        print("LPAllocator: Starting linear programming allocation...")

        # 1. Fetch tasks for the plan
        plan = await self.registry.get_plan(plan_id)
        if not plan:
            print(f"No plan found for plan_id={plan_id}")
            return {}
        tasks = await self.registry.list_tasks()
        tasks = [task for task in tasks if task.plan_id == plan_id]
        print(f"Fetched {len(tasks)} tasks for the plan")

        # 2. Fetch robots and their capabilities
        robots = await self.registry.list_robots()
        if not robots:
            print("No robots found in the registry.")
            return {}
        print(f"Fetched {len(robots)} robots from the registry")

        # 3. Build capability map and infer task capabilities
        robot_caps = {r.robot_id: set(r.capabilities) for r in robots}
        task_caps = {}
        
        # Find a fallback robot (one with most capabilities)
        fallback_robot = max(robots, key=lambda r: len(r.capabilities))
        print(f"Selected fallback robot {fallback_robot.robot_id} with capabilities: {fallback_robot.capabilities}")
        
        for t in tasks:
            # If task has no required capabilities, infer them from description
            if not getattr(t, 'required_capabilities', []):
                caps = set()
                desc = t.description.lower()
                # Only add navigate if it's explicitly about navigation
                if 'navigate to' in desc or 'move to' in desc or 'go to' in desc:
                    caps.add('navigate')
                # Only add pick if it's explicitly about picking up
                if 'pick up' in desc or 'pick the' in desc:
                    caps.add('pick')
                # Only add place if it's explicitly about placing
                if 'place in' in desc or 'place the' in desc:
                    caps.add('place')
                # Only add explore if it's explicitly about exploration
                if 'explore' in desc and 'area' in desc:
                    caps.add('explore_known_locations')
                # Only add capture_image if it's explicitly about capturing images
                if 'capture image' in desc or 'take picture' in desc:
                    caps.add('capture_image')
                
                # If no capabilities were inferred, use fallback robot's capabilities
                if not caps:
                    print(f"Task {t.task_id} ({t.description}): No specific capabilities found, using fallback robot capabilities")
                    caps = set(fallback_robot.capabilities)
                
                task_caps[t.task_id] = caps
                print(f"Task {t.task_id} ({t.description}): Inferred capabilities: {caps}")
            else:
                task_caps[t.task_id] = set(t.required_capabilities)
                print(f"Task {t.task_id} ({t.description}): Using explicit capabilities: {task_caps[t.task_id]}")

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
            print("No feasible allocation found.")
            return {}
        # 6. Build allocation result
        allocation = []
        for t in tasks:
            for r in robots:
                if pulp.value(x[(t.task_id, r.robot_id)]) == 1:
                    allocation.append({"task_id": t.task_id, "robot_id": r.robot_id})
                    break
        print(f"LPAllocator: Allocation result: {allocation}")
        # Update DB
        for a in allocation:
            try:
                await self.registry.update_task(a["task_id"], robot_id=a["robot_id"])
                print(f"Assigned robot {a['robot_id']} to task {a['task_id']}")
            except Exception as e:
                print(f"Failed to assign robot {a['robot_id']} to task {a['task_id']}: {e}")
        # Return as Allocation object (for compatibility)
        from .formats.formats import Allocation, RobotTask
        allocation_obj = Allocation(allocations=[RobotTask(**a) for a in allocation])
        return allocation_obj

class CostBasedAllocator(AllocatorBase):
    async def allocate(self, plan_id: int) -> Dict[int, str]:
        import os
        from openai import OpenAI
        import asyncio
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("CostBasedAllocator: Starting cost-based allocation...")

        # 1. Fetch plan, tasks, robots, and world statements
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
        allocation_result = []

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
            print("------------------------------------")
            print("Available tasks (root nodes):", available_tasks)
            print("Robot descriptions:", robot_descriptions)
            print("Task descriptions:", task_descriptions)
            prompt = (
                "You are an expert multi-robot task allocator. "
                "Given the following list of available tasks (all have no unmet dependencies, each with an optional suggested 'robot_type') and robots (each with a 'robot_type', 'capabilities', and previous task), assign each robot at most one available task to work on next. "
                "If a task specifies a 'robot_type', prioritize assigning it to a robot of that type, provided its capabilities and current state (previous task) are also suitable. "
                "If no 'robot_type' is specified for a task, or if no robot of the specified type is suitable/available, assign based on capabilities and other factors. "
                "Consider robot capabilities, robot types, the world statements, and the previous task each robot was working on. If the previous task is None, you are assigning the first task for the robot. "
                "Minimize switching cost (prefer to continue similar or related tasks based on world state and robot state if possible). "
                "Every robot is available in each round. Assign each robot a task if there is an available task within its capabilities and matching its type (if specified by the task). "
                "If there are more robots than available tasks, or no suitable tasks for a robot, do not assign a task to that robot in this round. "
                "Respond ONLY with a JSON object mapping robot IDs to assigned integer task IDs, e.g. {\"robot_id\": 5, ...}. Do not use null, string names, or non-integer task IDs."
                f"World Statements: {world_statements}\n"
                f"Available Tasks: {json.dumps(task_descriptions)}\n"
                f"Robots: {json.dumps(robot_descriptions)}\n"
            )
            response = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for robot fleet task allocation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            allocation_content = response.choices[0].message.content
            print("LLM allocation response:", allocation_content)
            print("------------------------------------")
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
        print(f"CostBasedAllocator: Final allocation: {allocation_result}")
        # Update DB: assign each task to its robot
        for a in allocation_result:
            try:
                await self.registry.update_task(a["task_id"], robot_id=a["robot_id"])
                print(f"Assigned robot {a['robot_id']} to task {a['task_id']}")
            except Exception as e:
                logger.error(f"Failed to assign robot {a['robot_id']} to task {a['task_id']}: {e}")
        # Always return Allocation object
        return Allocation(allocations=[RobotTask(**a) for a in allocation_result])

def get_allocator(allocation_strategy: int, db_url: str = None):
    """
    Return the appropriate Allocator instance based on the allocation strategy enum value.
    If registry is not provided, it must be set later.
    """
    from robot_fleet.proto import fleet_manager_pb2

    if allocation_strategy == fleet_manager_pb2.AllocationStrategy.LP:
        return LPAllocator(db_url)
    elif allocation_strategy == fleet_manager_pb2.AllocationStrategy.LLM:
        return LLMAllocator(db_url)
    elif allocation_strategy == fleet_manager_pb2.AllocationStrategy.COST_BASED:
        return CostBasedAllocator(db_url)
    else:
        raise ValueError(f"Unknown allocation strategy: {allocation_strategy}")