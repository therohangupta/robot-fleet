from packages.robot_sdk.src.client.robot_client import RobotClient
from ..formats.formats import AllocatedDAGNode, AllocatedDAGPlan
from packages.fleet_sdk.src.instance_registry import RobotInstanceRegistry
from packages.config import DATABASE_URL, DEFAULT_ROBOT_HOST
from packages.metrics import track_operation
import asyncio
import logging
from typing import Optional
from argparse import ArgumentParser
from ..planners.types.replanner import Replanner
from ..events import emit_task_changed, emit_plan_changed

logger = logging.getLogger(__name__)

class Executor:
    def __init__(self, plan_id: int, db_url: Optional[str] = None, registry: Optional[RobotInstanceRegistry] = None):
        self.registry = registry or RobotInstanceRegistry(db_url or DATABASE_URL)
        self.mutex = asyncio.Lock()
        self.robot_to_idle_bool = {}
        self.complete_task_ids = set()
        self.abort = False
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
        For each robot, return task IDs in topological (dependency) order.
        Kahn's algorithm on the full DAG, then partition by robot while
        preserving that global ordering.
        """
        dep_map = {node.task_id: list(node.depends_on) for node in dag.nodes}
        all_ids = set(dep_map.keys())

        in_degree: dict[int, int] = {tid: 0 for tid in all_ids}
        children: dict[int, list[int]] = {tid: [] for tid in all_ids}
        for tid, deps in dep_map.items():
            for d in deps:
                if d in all_ids:
                    in_degree[tid] += 1
                    children[d].append(tid)

        queue = sorted(tid for tid, deg in in_degree.items() if deg == 0)
        topo_order: list[int] = []
        while queue:
            tid = queue.pop(0)
            topo_order.append(tid)
            for child in children[tid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
            queue.sort()

        task_to_robot = {node.task_id: node.robot_id for node in dag.nodes}
        robot_task_map: dict[str, list[int]] = {}
        for tid in topo_order:
            rid = task_to_robot[tid]
            if rid not in robot_task_map:
                robot_task_map[rid] = []
            robot_task_map[rid].append(tid)

        for robot_id, tasks in robot_task_map.items():
            logger.debug("Robot %s tasks (topo order): %s", robot_id, tasks)

        return robot_task_map

    async def _start_task(self, robot_id: int, task_id: int, task_description: str):
        async with track_operation(self.registry, "fleet_server", "task_execution", str(task_id)) as meta:
            meta["robot_id"] = str(robot_id)
            meta["plan_id"] = self.plan_id
            await self._start_task_inner(robot_id, task_id, task_description)

    async def _start_task_inner(self, robot_id: int, task_id: int, task_description: str):
        try:
            logger.info("Starting task %s for robot %s", task_description, robot_id)

            await self.registry.update_task_status(task_id, 2)  # TASK_IN_PROGRESS = 2
            emit_task_changed(task_id, plan_id=self.plan_id, status="in_progress")

            robot = await self.registry.get_robot(robot_id)
            host = robot.task_server_info.host
            if DEFAULT_ROBOT_HOST != "localhost" and host in ("localhost", "127.0.0.1"):
                host = DEFAULT_ROBOT_HOST
            robot_client = RobotClient(host, robot.task_server_info.port)

            if len(self.previous_task_status_messages) > 0:
                previous_msgs = "\n".join(self.previous_task_status_messages)
                task_description = f"""Here are the task status messages of the tasks that have been completed up to this point:\n{previous_msgs}\nDO THE FOLLOWING TASK:\n{task_description}"""
            else:
                task_description = f"DO THE FOLLOWING TASK:\n{task_description}"

            result = await robot_client.do_task(task_description)
            logger.info("Task %s for robot %s completed with result: %s", task_description, robot_id, result)

            # Store execution result in database
            result_message = f"Success: {result.message}" if result.success else f"Failed: {result.message}"
            await self.registry.update_task(task_id, result=result_message)

            async with self.mutex:
                if result.success and not result.replan:
                    await self.registry.update_task_status(task_id, 3)  # TASK_COMPLETED = 3
                    emit_task_changed(task_id, plan_id=self.plan_id, status="completed")

                    self.complete_task_ids.add(task_id)
                    self.robot_to_idle_bool[robot_id] = True
                    self.robot_task_map[robot_id].pop(0)
                    self.previous_task_status_messages.append(result.message.replace("Succeeded task!", ""))
                    return
                elif result.replan:
                    await self.registry.update_task_status(task_id, 5)  # TASK_FAILED = 5
                    emit_task_changed(task_id, plan_id=self.plan_id, status="failed")

                    replanner = Replanner(registry=self.registry)
                    self.plan_id = await replanner.replan(
                        plan_id=self.plan_id,
                        failed_task_id=task_id,
                        failure_message=result.message,
                        robot_task_assignments={robot_id: self.robot_task_map[robot_id] for robot_id in self.robot_task_map.keys()}
                    )
                    logger.info("Replan generated new plan with ID: %s", self.plan_id)
                    self.replan = True
                    return
                else:
                    await self.registry.update_task_status(task_id, 5)  # TASK_FAILED = 5
                    emit_task_changed(task_id, plan_id=self.plan_id, status="failed")
                    self.abort = True
                    logger.error("Task %s failed (non-replan) — aborting plan execution", task_id)
        except Exception as e:
            logger.error("Exception in _start_task for robot %s, task %s: %s", robot_id, task_id, e, exc_info=True)
            try:
                await self.registry.update_task_status(task_id, 5)  # TASK_FAILED = 5
                await self.registry.update_task(task_id, result=f"Exception: {str(e)}")
                emit_task_changed(task_id, plan_id=self.plan_id, status="failed")
            except Exception as update_error:
                logger.error("Failed to update task status on exception: %s", update_error)
            async with self.mutex:
                self.abort = True

    # main function that starts the full execution from DAG generation to sending tasks in a queue
    async def execute(self):
        async with track_operation(self.registry, "fleet_server", "plan_execution", str(self.plan_id)):
            await self._execute_inner()

    async def _execute_inner(self):
        await self.registry.update_plan(self.plan_id, execution_status=1)  # 1 = executing
        emit_plan_changed(self.plan_id, status="executing")

        dag = await self._generate_dag()
        task_to_dependency_map = {node.task_id: node.depends_on for node in dag.nodes}
        self.robot_task_map = await self._get_robot_task_map(dag)
        self.robot_to_idle_bool = {robot_id: True for robot_id in self.robot_task_map}
        self.complete_task_ids = set()
        total_tasks = sum(len(tasks) for tasks in self.robot_task_map.values())
        tasks_in_progress: list[asyncio.Task] = []

        while len(self.complete_task_ids) < total_tasks:
            if self.abort:
                break

            async with self.mutex:
                for robot_id, tasks in self.robot_task_map.items(): 
                    if not tasks:
                        continue
                    next_task_id = tasks[0]

                    dependencies_met = all(d in self.complete_task_ids for d in task_to_dependency_map.get(next_task_id, []))
                    
                    start_task = await self.registry.get_task(next_task_id)
                    if start_task is None:
                        continue
                    if self.robot_to_idle_bool[robot_id] and dependencies_met:
                        t = asyncio.create_task(self._start_task(robot_id, next_task_id, start_task.description))
                        t.robot_id = robot_id
                        t.task_id = next_task_id
                        tasks_in_progress.append(t)
                        self.robot_to_idle_bool[robot_id] = False

            tasks_in_progress = [t for t in tasks_in_progress if not t.done()]
            await asyncio.sleep(1)

            async with self.mutex:
                if self.replan:
                    dag = await self._generate_dag()
                    task_to_dependency_map = {node.task_id: node.depends_on for node in dag.nodes}
                    self.robot_task_map = await self._get_robot_task_map(dag)
                    self.robot_to_idle_bool = {robot_id: True for robot_id in self.robot_task_map}
                    self.complete_task_ids = set()
                    self.abort = False
                    total_tasks = sum(len(tasks) for tasks in self.robot_task_map.values())
                    self.replan = False

        if self.abort:
            # Wait for any in-flight tasks to finish before marking remaining as failed
            if tasks_in_progress:
                logger.info("Waiting for %d in-flight tasks to finish before cleanup", len(tasks_in_progress))
                await asyncio.gather(*tasks_in_progress, return_exceptions=True)

            # Mark all remaining pending tasks as failed
            for robot_id, remaining in self.robot_task_map.items():
                for tid in remaining:
                    try:
                        t = await self.registry.get_task(tid)
                        if t and t.status not in (3, 5):  # not completed or failed
                            await self.registry.update_task_status(tid, 5)
                            await self.registry.update_task(tid, result="Cancelled: plan aborted due to task failure")
                            emit_task_changed(tid, plan_id=self.plan_id, status="failed")
                    except Exception:
                        pass

            await self.registry.update_plan(self.plan_id, execution_status=3)  # 3 = failed
            emit_plan_changed(self.plan_id, status="failed")
            logger.info("Plan aborted — marked as failed")
        else:
            await self.registry.update_plan(self.plan_id, execution_status=2)  # 2 = completed
            emit_plan_changed(self.plan_id, status="completed")
            logger.info("Plan completed successfully")

        
async def main():
    executor = Executor(db_url=DATABASE_URL)
    parser = ArgumentParser()
    parser.add_argument("plan_id", type=int)
    args = parser.parse_args() 
    executor.plan_id = int(args.plan_id)
    await executor.execute()

if __name__ == "__main__":
    asyncio.run(main())
