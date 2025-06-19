"""Fleet Manager gRPC client"""
import grpc
from typing import Optional, List, Dict, Union
from ..proto import fleet_manager_pb2
from ..proto import fleet_manager_pb2_grpc
from .printer import PLANNING_STRATEGY_ENUMS, ALLOCATION_STRATEGY_ENUMS
from collections import defaultdict
from tabulate import tabulate

# Default connection settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 50051

class FleetManagerClient:
    def __init__(self, server_address: str = f"{DEFAULT_HOST}:{DEFAULT_PORT}"):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = fleet_manager_pb2_grpc.FleetManagerStub(self.channel)

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def register_robot(self, robot_id: str, robot_type: str, description: str, capabilities: list, 
                      task_server_host: str, task_server_port: int, docker_host: str, docker_port: int, 
                      container_image: str, container_env: dict = None):
        """Register a new robot with the fleet manager.
        
        Returns:
            RegisterRobotResponse with success, message, and robot object
        """
        task_server_info = fleet_manager_pb2.TaskServerInfo(
            host = task_server_host,
            port = task_server_port
        )
        deployment_info = fleet_manager_pb2.DeploymentInfo(
            docker_host=docker_host,
            docker_port=docker_port
        )
        container_config = fleet_manager_pb2.ContainerConfig(
            image=container_image,
            environment=container_env or {}
        )
        
        request = fleet_manager_pb2.RegisterRobotRequest(
            robot_id=robot_id,
            robot_type=robot_type,
            description=description,
            capabilities=capabilities,
            task_server_info=task_server_info,
            deployment=deployment_info,
            container=container_config
        )
        return self.stub.RegisterRobot(request)

    def deploy_robot(self, robot_id: str):
        """Deploy a robot container using stored configuration.
        
        Returns:
            DeployRobotResponse with success, message, and container info
        """
        request = fleet_manager_pb2.DeployRobotRequest(robot_id=robot_id)
        return self.stub.DeployRobot(request)

    def undeploy_robot(self, robot_id: str):
        """Undeploy (stop and remove) a robot container.
        
        Returns:
            UndeployRobotResponse with success and message
        """
        request = fleet_manager_pb2.UndeployRobotRequest(robot_id=robot_id)
        return self.stub.UndeployRobot(request)

    def unregister_robot(self, robot_id: str):
        """Unregister a robot from the fleet manager.
        
        Returns:
            UnregisterRobotResponse with success and message
        """
        request = fleet_manager_pb2.UnregisterRobotRequest(robot_id=robot_id)
        return self.stub.UnregisterRobot(request)

    def list_robots(self, filter_type: str = "ALL"):
        """List robots with optional filtering.
        
        Args:
            filter_type: One of "all", "deployed", or "registered"
            
        Returns:
            ListRobotsResponse with list of Robot objects
        """
        filter_map = {
            "all": fleet_manager_pb2.ListRobotsRequest.ALL,
            "deployed": fleet_manager_pb2.ListRobotsRequest.DEPLOYED,
            "registered": fleet_manager_pb2.ListRobotsRequest.REGISTERED
        }
        request = fleet_manager_pb2.ListRobotsRequest(
            filter=filter_map[filter_type.lower()]
        )
        return self.stub.ListRobots(request)

    def get_robot_status(self, robot_id: str):
        """Get detailed status of a specific robot.
        
        Returns:
            RobotStatus object
        """
        request = fleet_manager_pb2.GetRobotStatusRequest(robot_id=robot_id)
        return self.stub.GetRobotStatus(request)

    # Goal management methods
    def create_goal(self, description: str, task_ids: Optional[List[int]] = None):
        """Create a new goal.
        
        Args:
            description: Description of the goal
            task_ids: Optional list of task IDs to associate with the goal
            
        Returns:
            CreateGoalResponse with goal object and error message
        """
        request = fleet_manager_pb2.CreateGoalRequest(
            description=description,
            task_ids=task_ids or []
        )
        return self.stub.CreateGoal(request)

    def delete_goal(self, goal_id: int):
        """Delete a goal by ID.
        
        Returns:
            DeleteGoalResponse with goal object and error message
        """
        request = fleet_manager_pb2.DeleteGoalRequest(goal_id=goal_id)
        return self.stub.DeleteGoal(request)

    def list_goals(self):
        """List all goals.
        
        Returns:
            ListGoalsResponse with list of goal objects and error message
        """
        request = fleet_manager_pb2.ListGoalsRequest()
        return self.stub.ListGoals(request)

    def get_goal(self, goal_id: int):
        """Get a goal by ID.
        
        Returns:
            GetGoalResponse with goal object and error message
        """
        request = fleet_manager_pb2.GetGoalRequest(goal_id=goal_id)
        return self.stub.GetGoal(request)

    # Plan management methods
    def create_plan(self, planning_strategy: str, goal_ids: List[int], allocation_strategy: str):
        """Create a new plan with specified planning strategy, allocator, and goals.
        
        Args:
            planning_strategy: Planning strategy (MONOLITHIC, DAG, or BIG_DAG)
            goal_ids: List of goal IDs to include in the plan
            allocation_strategy: Allocation strategy to use ('llm' or 'lp')
        
        Returns:
            CreatePlanResponse with plan object and error message
        """
        if not isinstance(goal_ids, (list, tuple)):
            goal_ids = [goal_ids]
        goal_ids = [int(gid) for gid in goal_ids]
        planning_strategy = PLANNING_STRATEGY_ENUMS.get(
            planning_strategy.lower(),
            PLANNING_STRATEGY_ENUMS["monolithic"]
        )
        allocation_strategy = ALLOCATION_STRATEGY_ENUMS.get(
            allocation_strategy.lower(),
            ALLOCATION_STRATEGY_ENUMS["lp"]
        )
        request = fleet_manager_pb2.CreatePlanRequest(
            planning_strategy=planning_strategy,
            allocation_strategy=allocation_strategy,
            goal_ids=goal_ids
        )
        return self.stub.CreatePlan(request)

    def get_plan(self, plan_id: int):
        """Get a plan by ID.
        
        Returns:
            GetPlanResponse with plan object and error message
        """
        request = fleet_manager_pb2.GetPlanRequest(plan_id=plan_id)
        return self.stub.GetPlan(request)

    def list_plans(self):
        """List all plans.
        
        Returns:
            ListPlansResponse with list of plan objects and error message
        """
        request = fleet_manager_pb2.ListPlansRequest()
        return self.stub.ListPlans(request)

    def delete_plan(self, plan_id: int):
        """Delete a plan by ID.
        
        Returns:
            DeletePlanResponse with plan object and error message
        """
        request = fleet_manager_pb2.DeletePlanRequest(plan_id=plan_id)
        return self.stub.DeletePlan(request)

    def start_plan(self, plan_id: int):
        """Starts the execution of a specific plan."""
        request = fleet_manager_pb2.StartPlanRequest(plan_id=plan_id)
        try:
            response = self.stub.StartPlan(request)
            return response
        except grpc.RpcError as e:
            return {"error": f"RPC failed: {e.details()} ({e.code()})"}

    # Task management methods
    def create_task(self, description: str, robot_id: str, robot_type: str,
                   goal_id: Optional[int] = None, 
                   plan_id: Optional[int] = None,
                   dependency_task_ids: Optional[List[int]] = None):
        """Create a new task.
        
        Args:
            description: Description of the task
            robot_id: ID of the robot to execute the task
            goal_id: Optional goal ID to associate with the task
            plan_id: Optional plan ID to associate with the task
            dependency_task_ids: Optional list of task IDs that this task depends on
            
        Returns:
            CreateTaskResponse with task object and error message
        """
        request = fleet_manager_pb2.CreateTaskRequest(
            description=description,
            robot_id=robot_id,
            robot_type=robot_type,
            goal_id=goal_id or 0,
            plan_id=plan_id or 0,
            dependency_task_ids=dependency_task_ids or []
        )
        return self.stub.CreateTask(request)

    def get_task(self, task_id: int):
        """Get a task by ID.
        
        Returns:
            GetTaskResponse with task object and error message
        """
        request = fleet_manager_pb2.GetTaskRequest(task_id=task_id)
        return self.stub.GetTask(request)

    def delete_task(self, task_id: int):
        """Delete a task by ID.
        
        Args:
            task_id: ID of the task to delete
            
        Returns:
            Response with success/failure information
        """
        request = fleet_manager_pb2.DeleteTaskRequest(task_id=task_id)
        return self.stub.DeleteTask(request)

    def list_tasks(self, 
                  plan_ids: Optional[List[int]] = None,
                  goal_ids: Optional[List[int]] = None, 
                  robot_ids: Optional[List[str]] = None):
        """List tasks with optional filtering.
        
        Args:
            plan_ids: Optional list of plan IDs to filter by
            goal_ids: Optional list of goal IDs to filter by
            robot_ids: Optional list of robot IDs to filter by
            
        Returns:
            ListTasksResponse with list of task objects and error message
        """
        request = fleet_manager_pb2.ListTasksRequest(
            plan_ids=plan_ids or [],
            goal_ids=goal_ids or [],
            robot_ids=robot_ids or []
        )
        return self.stub.ListTasks(request)


    # --- World Statement Client Methods ---
    def add_world_statement(self, statement: str) -> Optional[fleet_manager_pb2.WorldStatement]:
        """Adds a world statement via the server."""
        request = fleet_manager_pb2.AddWorldStatementRequest(statement=statement)
        response = self.stub.AddWorldStatement(request)
        if getattr(response, 'error', None):
            return None
        return response.world_statement

    def get_world_statement(self, world_statement_id: str) -> Optional[fleet_manager_pb2.WorldStatement]:
        """Gets a specific world statement by ID."""
        request = fleet_manager_pb2.GetWorldStatementRequest(world_statement_id=world_statement_id)
        response = self.stub.GetWorldStatement(request)
        if getattr(response, 'error', None):
            return None
        return response.world_statement

    def list_world_statements(self) -> list[fleet_manager_pb2.WorldStatement]:
        """Lists all world statements."""
        request = fleet_manager_pb2.ListWorldStatementsRequest()
        response = self.stub.ListWorldStatements(request)
        if getattr(response, 'error', None):
            return []
        return list(response.world_statements)

    def delete_world_statement(self, world_statement_id: str) -> bool:
        """Deletes a world statement by ID."""
        request = fleet_manager_pb2.DeleteWorldStatementRequest(world_statement_id=world_statement_id)
        response = self.stub.DeleteWorldStatement(request)
        if getattr(response, 'error', None):
            return False
        return response.success
    
    def analyze_idle_time(self, plan_id: int) -> str:
        """
        Fetch plan & its tasks via gRPC, simulate execution assuming
        each task takes exactly one time unit, and return a text‑table
        + idle‑stats for each robot.
        """
        # 1) Fetch the plan
        plan_resp = self.get_plan(plan_id)
        plan = plan_resp.plan
        if not getattr(plan, 'plan_id', None):
            return f"Error: Plan with ID {plan_id} not found"

        # 2) Fetch tasks under that plan
        tasks = self.list_tasks(plan_ids=[plan_id]).tasks
        if not tasks:
            return f"Analyzing Plan #{plan_id}: 0 tasks\n\nNo robots to analyze."

        # 3) Build robot→tasks and deps maps
        robot_tasks = defaultdict(list)
        deps = {}
        for t in tasks:
            if not t.robot_id:
                continue  # Skip tasks without robot assignments
            robot_tasks[t.robot_id].append(t.task_id)
            deps[t.task_id] = list(t.dependency_task_ids)

        robot_ids = list(robot_tasks.keys())
        if not robot_ids:
            return f"Analyzing Plan #{plan_id}: {len(tasks)} tasks\n\nNo robots assigned to tasks."

        total_tasks = len(tasks)

        # 4) Initialize simulation
        completed = set()
        time = 0
        robot_timelines = {r: [] for r in robot_ids}

        # 5) Simulate until all tasks complete or deadlock
        while len(completed) < total_tasks:
            # a) Find ready tasks for each robot
            assignments = {}
            for r in robot_ids:
                for tid in robot_tasks[r]:
                    if tid in completed:
                        continue
                    if all(d in completed for d in deps[tid]):
                        assignments[r] = tid
                        break

            # Deadlock check
            if not assignments:
                break

            # b) Record what each robot does this timestep
            for r in robot_ids:
                if r in assignments:
                    robot_timelines[r].append(assignments[r])
                else:
                    robot_timelines[r].append("IDLE")

            # c) Mark assigned tasks complete
            completed.update(assignments.values())
            time += 1

        # 6) Build and print the grid
        headers = ["Robot/Time"] + [str(t) for t in range(time)]
        rows = []
        for r in robot_ids:
            row = [r] + [
                f"T{entry}" if entry != "IDLE" else "IDLE"
                for entry in robot_timelines[r]
            ]
            rows.append(row)
        table = tabulate(rows, headers=headers, tablefmt="grid")

        # 7) Compute idle statistics
        idle_stats = {
            r: robot_timelines[r].count("IDLE") / time * 100
            for r in robot_ids
        }
        avg_idle = sum(idle_stats.values()) / len(idle_stats)

        # 8) Assemble final output
        lines = [
            f"Analyzing Plan #{plan_id}: {total_tasks} tasks",
            "",
            table,
            ""
        ]
        for r, pct in idle_stats.items():
            lines.append(f"Robot {r}: {pct:.1f}% idle time")
        lines.append(f"\nAverage idle time across all robots: {avg_idle:.1f}%")

        return "\n".join(lines)
