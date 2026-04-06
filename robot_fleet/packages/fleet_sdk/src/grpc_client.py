"""
Fleet Manager gRPC client. Uses in-repo packages.proto only.
"""
import grpc
from typing import Optional, List

from packages.proto import fleet_manager_pb2
from packages.proto import fleet_manager_pb2_grpc
from packages.config import GRPC_SERVER_ADDRESS


class FleetManagerClient:
    def __init__(self, server_address: str = GRPC_SERVER_ADDRESS):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = fleet_manager_pb2_grpc.FleetManagerStub(self.channel)

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def register_robot(
        self,
        robot_id: str,
        robot_type: str,
        description: str,
        capabilities: list,
        task_server_host: str,
        task_server_port: int,
        docker_host: str = "",
        docker_port: int = 0,
        container_image: str = "",
        container_env: Optional[dict] = None,
    ):
        task_server_info = fleet_manager_pb2.TaskServerInfo(host=task_server_host, port=task_server_port)
        deployment_info = fleet_manager_pb2.DeploymentInfo(docker_host=docker_host, docker_port=docker_port)
        container_config = fleet_manager_pb2.ContainerConfig(image=container_image, environment=container_env or {})
        request = fleet_manager_pb2.RegisterRobotRequest(
            robot_id=robot_id,
            robot_type=robot_type,
            description=description,
            capabilities=capabilities,
            task_server_info=task_server_info,
            deployment=deployment_info,
            container=container_config,
        )
        return self.stub.RegisterRobot(request)

    def unregister_robot(self, robot_id: str):
        return self.stub.UnregisterRobot(fleet_manager_pb2.UnregisterRobotRequest(robot_id=robot_id))

    def list_robots(self, filter_type: str = "all"):
        filter_map = {
            "all": fleet_manager_pb2.ListRobotsRequest.ALL,
            "deployed": fleet_manager_pb2.ListRobotsRequest.DEPLOYED,
            "registered": fleet_manager_pb2.ListRobotsRequest.REGISTERED,
        }
        request = fleet_manager_pb2.ListRobotsRequest(filter=filter_map.get(filter_type.lower(), fleet_manager_pb2.ListRobotsRequest.ALL))
        return self.stub.ListRobots(request)

    def get_robot(self, robot_id: str):
        return self.stub.GetRobot(fleet_manager_pb2.GetRobotRequest(robot_id=robot_id))

    def list_goals(self):
        return self.stub.ListGoals(fleet_manager_pb2.ListGoalsRequest())

    def get_goal(self, goal_id: int):
        return self.stub.GetGoal(fleet_manager_pb2.GetGoalRequest(goal_id=goal_id))

    def create_goal(self, description: str, task_ids: Optional[List[int]] = None):
        return self.stub.CreateGoal(fleet_manager_pb2.CreateGoalRequest(description=description, task_ids=task_ids or []))

    def delete_goal(self, goal_id: int):
        return self.stub.DeleteGoal(fleet_manager_pb2.DeleteGoalRequest(goal_id=goal_id))

    def list_plans(self):
        return self.stub.ListPlans(fleet_manager_pb2.ListPlansRequest())

    def get_plan(self, plan_id: int):
        return self.stub.GetPlan(fleet_manager_pb2.GetPlanRequest(plan_id=plan_id))

    def create_plan(
        self,
        planning_strategy,
        goal_ids: List[int],
        allocation_strategy,
        name: str = "",
        description: str = "",
    ):
        request = fleet_manager_pb2.CreatePlanRequest(
            planning_strategy=planning_strategy,
            allocation_strategy=allocation_strategy,
            goal_ids=goal_ids,
            name=name,
            description=description,
        )
        return self.stub.CreatePlan(request)

    def allocate_plan(self, plan_id: int, allocation_strategy):
        request = fleet_manager_pb2.AllocatePlanRequest(plan_id=plan_id, allocation_strategy=allocation_strategy)
        return self.stub.AllocatePlan(request)

    def start_plan(self, plan_id: int):
        return self.stub.StartPlan(fleet_manager_pb2.StartPlanRequest(plan_id=plan_id))

    def delete_plan(self, plan_id: int):
        return self.stub.DeletePlan(fleet_manager_pb2.DeletePlanRequest(plan_id=plan_id))

    def list_tasks(
        self,
        plan_ids: Optional[List[int]] = None,
        goal_ids: Optional[List[int]] = None,
        robot_ids: Optional[List[str]] = None,
    ):
        return self.stub.ListTasks(
            fleet_manager_pb2.ListTasksRequest(
                plan_ids=plan_ids or [],
                goal_ids=goal_ids or [],
                robot_ids=robot_ids or [],
            )
        )

    def get_task(self, task_id: int):
        return self.stub.GetTask(fleet_manager_pb2.GetTaskRequest(task_id=task_id))

    def create_task(
        self,
        description: str,
        robot_id: str = "",
        robot_type: str = "",
        goal_id: Optional[int] = None,
        plan_id: Optional[int] = None,
        dependency_task_ids: Optional[List[int]] = None,
    ):
        request = fleet_manager_pb2.CreateTaskRequest(
            description=description,
            robot_id=robot_id or "",
            robot_type=robot_type or "",
            goal_id=goal_id or 0,
            plan_id=plan_id or 0,
            dependency_task_ids=dependency_task_ids or [],
        )
        return self.stub.CreateTask(request)

    def update_task(
        self,
        task_id: int,
        description: Optional[str] = None,
        goal_id: Optional[int] = None,
        robot_id: Optional[str] = None,
        dependency_task_ids: Optional[List[int]] = None,
        update_dependency_task_ids: bool = False,
    ):
        request = fleet_manager_pb2.UpdateTaskRequest(
            task_id=task_id,
            update_dependency_task_ids=update_dependency_task_ids,
            dependency_task_ids=dependency_task_ids or [],
        )
        if description is not None:
            request.description = description
        if goal_id is not None:
            request.goal_id = goal_id
        if robot_id is not None:
            request.robot_id = robot_id
        return self.stub.UpdateTask(request)

    def delete_task(self, task_id: int):
        return self.stub.DeleteTask(fleet_manager_pb2.DeleteTaskRequest(task_id=task_id))

    def list_world_statements(self):
        response = self.stub.ListWorldStatements(fleet_manager_pb2.ListWorldStatementsRequest())
        if getattr(response, 'error', None):
            return []
        return list(response.world_statements)

    def add_world_statement(self, statement: str):
        response = self.stub.AddWorldStatement(fleet_manager_pb2.AddWorldStatementRequest(statement=statement))
        if getattr(response, 'error', None):
            return None
        return response.world_statement

    def delete_world_statement(self, statement_id: str):
        response = self.stub.DeleteWorldStatement(fleet_manager_pb2.DeleteWorldStatementRequest(world_statement_id=statement_id))
        if getattr(response, 'error', None):
            return False
        return response.success
