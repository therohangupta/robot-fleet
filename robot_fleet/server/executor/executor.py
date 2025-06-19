from robot_fleet.robots.client.robot_client import RobotClient
from robot_fleet.server.planner.formats.formats import AllocatedDAGNode, AllocatedDAGPlan
from robot_fleet.robots.registry.instance_registry import RobotInstanceRegistry
import asyncio
from typing import Optional
from argparse import ArgumentParser
from robot_fleet.server.planner.types.replanner import Replanner

class Executor:
    def __init__(self, plan_id: int, db_url: Optional[str] = None):
        if not db_url:
            db_url = "postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet"
            print(f"Using default PostgreSQL database: {db_url}")
        else:
             print(f"Using database: {db_url}")
        self.registry = RobotInstanceRegistry(db_url)
        self.mutex = asyncio.Lock()
        self.robot_to_idle_bool = {}
        self.complete_task_ids = set()
        self.robot_task_map = {}
        self.plan_id = plan_id
        self.replan = False
        self.previous_task_status_messages = []

    async def _generate_dag(self) -> AllocatedDAGPlan:
        plan = await self.registry.get_plan(self.plan_id)
        dag = AllocatedDAGPlan(nodes=[])
        for task_id in plan.task_ids:
            task = await self.registry.get_task(task_id)
            node = AllocatedDAGNode(
                task_id=task_id,
                description=task.description,
                goal_id=task.goal_id,
                robot_id=task.robot_id,
                depends_on=[dep_id for dep_id in task.dependency_task_ids]
            )
            dag.nodes.append(node)
        return dag
    
    async def _get_robot_task_map(self, dag: AllocatedDAGPlan) -> dict:
        """
        For each robot, return a list of task IDs sorted numerically.
        Output: Dict[robot_id, List[task_id]]
        """
        robot_task_map = {}
        for node in dag.nodes:
            if node.robot_id not in robot_task_map:
                robot_task_map[node.robot_id] = []
            robot_task_map[node.robot_id].append(node.task_id)
        
        for robot_id in robot_task_map:
            robot_task_map[robot_id].sort()
            print(f"Robot {robot_id} tasks in order: {robot_task_map[robot_id]}")
        
        return robot_task_map

    # tries to do the task 3 times and if its not successful on the third try, raises an exception
    async def _start_task(self, robot_id: int, task_id: int, task_description: str):
        try:
            print(f"Starting task {task_description} for robot {robot_id}")
            robot = await self.registry.get_robot(robot_id)
            robot_client = RobotClient(robot.task_server_info.host, robot.task_server_info.port)

            if len(self.previous_task_status_messages) > 0:
                previous_msgs = "\n".join(self.previous_task_status_messages)
                task_description = f"""Here are the task status messages of the tasks that have been completed up to this point:\n{previous_msgs}\nDO THE FOLLOWING TASK:\n{task_description}"""
            else:
                task_description = f"DO THE FOLLOWING TASK:\n{task_description}"

            result = await robot_client.do_task(task_description)
            print(f"Task {task_description} for robot {robot_id} completed with result: {result}")
            async with self.mutex:
                if result.success and not result.replan:
                    # complete the task, discard curr task, and set robot to idle
                    self.complete_task_ids.add(task_id)
                    self.robot_to_idle_bool[robot_id] = True
                    self.robot_task_map[robot_id].pop(0)
                    self.previous_task_status_messages.append(result.message.replace("Succeeded task!", ""))
                    return
                elif result.replan:
                    breakpoint()
                    replanner = Replanner(db_url="postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet")
                    self.plan_id = await replanner.replan(
                        plan_id=self.plan_id,
                        failed_task_id=task_id,
                        failure_message=result.message,
                        robot_task_assignments={robot_id: self.robot_task_map[robot_id] for robot_id in self.robot_task_map.keys()}
                    )
                    print(f"Replan generated new plan with ID: {self.plan_id}")
                    self.replan = True
                    return
        except Exception as e:
            print(f"Exception in _start_task for robot {robot_id}, task {task_id}: {e}")


    async def execute(self):
        dag = await self._generate_dag()
        task_to_dependency_map = {node.task_id: node.depends_on for node in dag.nodes}
        self.robot_task_map = await self._get_robot_task_map(dag)
        self.robot_to_idle_bool = {robot_id: True for robot_id in self.robot_task_map}
        self.complete_task_ids = set()
        total_tasks = sum(len(tasks) for tasks in self.robot_task_map.values())
        tasks_in_progress = []
        # while not all tasks are complete, keep checking for new tasks to start
        while len(self.complete_task_ids) < total_tasks:
            async with self.mutex:
                for robot_id, tasks in self.robot_task_map.items(): 
                    if not tasks:
                        continue
                    next_task_id = tasks[0]

                    dependencies_met = all(task_id in self.complete_task_ids for task_id in task_to_dependency_map[next_task_id])
                    
                    start_task = await self.registry.get_task(next_task_id)
                    if start_task is None:
                        continue
                    if self.robot_to_idle_bool[robot_id] and dependencies_met:
                        t = asyncio.create_task(self._start_task(robot_id, next_task_id, start_task.description))
                        t.robot_id = robot_id
                        t.task_id = next_task_id
                        tasks_in_progress.append(t)
                        self.robot_to_idle_bool[robot_id] = False

                    following_task = "NO FOLLOWING TASK"
                    if len(self.robot_task_map[robot_id]) > 1:
                        following_task = self.robot_task_map[robot_id][1]
                    
                    if self.robot_to_idle_bool[robot_id] and not dependencies_met:
                        print(f"Task {start_task.description} waiting on dependencies, can't start {following_task}")
                    else:
                        print(f"Task {start_task.description} in progress, can't start {following_task}, robot {robot_id} busy")
            # Check for exceptions in tasks
            for t in tasks_in_progress:
                if t.done() and t.exception() is not None:
                    logger.error(f"Task {t.task_id} for robot {t.robot_id} failed with exception: {t.exception()}")
            await asyncio.sleep(1)

            # TODO: if replan is true, generate a new plan and update global variables to execute from where the replanner left off
            async with self.mutex:
                if self.replan:
                    dag = await self._generate_dag()
                    task_to_dependency_map = {node.task_id: node.depends_on for node in dag.nodes}
                    self.robot_task_map = await self._get_robot_task_map(dag)
                    self.robot_to_idle_bool = {robot_id: True for robot_id in self.robot_task_map}
                    self.complete_task_ids = set()
                    total_tasks = sum(len(tasks) for tasks in self.robot_task_map.values())
                    # self.previous_task_status_messages = []
                    self.replan = False
        print("Plan Completed")

        
async def main():
    executor = Executor(db_url="postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet")
    parser = ArgumentParser()
    parser.add_argument("plan_id", type=int)
    args = parser.parse_args() 
    executor.plan_id = int(args.plan_id)
    await executor.execute()

if __name__ == "__main__":
    asyncio.run(main())