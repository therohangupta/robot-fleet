# /Users/trevorasbery/Desktop/AgenticNet/multirobot-task/tests/robot_fleet/server/test_service.py
import pytest
import pytest_asyncio
import asyncio
import grpc
from unittest.mock import MagicMock, AsyncMock, patch
from google.protobuf import timestamp_pb2
import sys
import os
from datetime import datetime

# Add project root to sys.path to allow importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from robot_fleet.server.service import FleetManagerService
from robot_fleet.robots.registry.instance_registry import RobotInstanceRegistry
from robot_fleet.robots.registry.models import Base
from robot_fleet.proto import fleet_manager_pb2
from robot_fleet.proto import fleet_manager_pb2_grpc
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from typing import AsyncGenerator
from robot_fleet.robots.containers.manager import ContainerInfo

# Use in-memory SQLite database for testing
DATABASE_URL = "sqlite+aiosqlite:///:memory:"

class MockGrpcContext:
    """Mock implementation of gRPC context for testing."""
    def __init__(self):
        self.code = grpc.StatusCode.OK
        self.details = ""
    
    def set_code(self, code):
        self.code = code
    
    def set_details(self, details):
        self.details = details

# Configure asyncio for pytest
@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="function")
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create and initialize the SQLite database engine."""
    async_engine = create_async_engine(DATABASE_URL, echo=False)
    
    # Create tables before tests
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield async_engine
    
    # Drop tables after tests
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await async_engine.dispose()

@pytest_asyncio.fixture(scope="function")
async def service_instance(engine) -> AsyncGenerator[FleetManagerService, None]:
    """Create and initialize the service with a mocked container manager."""
    # Set test mode flag
    os.environ["TESTING"] = "true"
    
    with patch('robot_fleet.robots.containers.manager.ContainerManager') as mock_container_manager_class:
        # Create a mock ContainerManager instance
        mock_container_manager = mock_container_manager_class.return_value
        
        # Create a service instance
        service = FleetManagerService(db_url=DATABASE_URL)
        
        # Replace the engine in the registry with our test engine
        service.registry = RobotInstanceRegistry(db_url=DATABASE_URL, engine=engine)
        
        # Wait for initialization
        await service.registry.initialize()
        
        # Set up the mock container manager
        timestamp = timestamp_pb2.Timestamp()
        timestamp.GetCurrentTime()
        
        # Create ContainerInfo for mock response
        container_info = ContainerInfo(
            container_id="mock-container-id",
            image="test/image:latest",
            host="localhost",
            port=8080,
            environment={"ENV1": "value1"},
            created_at=datetime.now().isoformat()
        )
        
        # Set up deploy_robot mock as AsyncMock to return a successful ContainerInfo
        mock_container_manager.deploy_robot = AsyncMock(return_value=container_info)
        
        # Set up stop_robot mock as AsyncMock to return True
        mock_container_manager.stop_robot = AsyncMock(return_value=True)
        
        # Replace the service's container manager with our mock
        service.container_manager = mock_container_manager
        
        yield service
        
        # Clean up
        os.environ["TESTING"] = "false"

@pytest.mark.asyncio
async def test_register_robot(service_instance):
    """Test the RegisterRobot method."""
    # Create request
    request = fleet_manager_pb2.RegisterRobotRequest(
        robot_id="test-robot-1",
        robot_type="test-type",
        description="Test robot",
        capabilities=["capability1", "capability2"],
        mcp=fleet_manager_pb2.MCPInfo(host="localhost", port=8080),
        deployment=fleet_manager_pb2.DeploymentInfo(docker_host="localhost", docker_port=2375),
        container=fleet_manager_pb2.ContainerConfig(
            image="test/image:latest",
            environment={"ENV1": "value1"}
        )
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.RegisterRobot(request, context)
    
    # Verify response
    assert response.success
    assert "test-robot-1" in response.message
    assert response.robot.robot_id == "test-robot-1"
    assert response.robot.robot_type == "test-type"
    assert "capability1" in response.robot.capabilities
    assert context.code == grpc.StatusCode.OK

@pytest.mark.asyncio
async def test_deploy_robot(service_instance):
    """Test the DeployRobot method."""
    # Register a robot first
    register_request = fleet_manager_pb2.RegisterRobotRequest(
        robot_id="test-robot-2",
        robot_type="test-type",
        description="Test robot",
        capabilities=["capability1", "capability2"],
        mcp=fleet_manager_pb2.MCPInfo(host="localhost", port=8080),
        deployment=fleet_manager_pb2.DeploymentInfo(docker_host="localhost", docker_port=2375),
        container=fleet_manager_pb2.ContainerConfig(
            image="test/image:latest",
            environment={"ENV1": "value1"}
        )
    )
    context = MockGrpcContext()
    await service_instance.RegisterRobot(register_request, context)
    
    # Create request for deployment
    request = fleet_manager_pb2.DeployRobotRequest(
        robot_id="test-robot-2"
    )
    
    # Call service method
    response = await service_instance.DeployRobot(request, context)
    
    # Verify response
    assert response.success
    assert "test-robot-2" in response.message

@pytest.mark.asyncio
async def test_undeploy_robot(service_instance):
    """Test the UndeployRobot method."""
    # First register and deploy a robot
    register_request = fleet_manager_pb2.RegisterRobotRequest(
        robot_id="test-robot-3",
        robot_type="test-type",
        description="Test robot",
        capabilities=["capability1", "capability2"],
        mcp=fleet_manager_pb2.MCPInfo(host="localhost", port=8080),
        deployment=fleet_manager_pb2.DeploymentInfo(docker_host="localhost", docker_port=2375),
        container=fleet_manager_pb2.ContainerConfig(
            image="test/image:latest",
            environment={"ENV1": "value1"}
        )
    )
    context = MockGrpcContext()
    await service_instance.RegisterRobot(register_request, context)
    
    # Deploy the robot first
    deploy_request = fleet_manager_pb2.DeployRobotRequest(
        robot_id="test-robot-3"
    )
    await service_instance.DeployRobot(deploy_request, context)
    
    # Create request for undeployment
    request = fleet_manager_pb2.UndeployRobotRequest(
        robot_id="test-robot-3"
    )
    
    # Call service method
    response = await service_instance.UndeployRobot(request, context)
    
    # Verify response
    assert response.success
    assert "test-robot-3" in response.message

@pytest.mark.asyncio
async def test_list_robots(service_instance):
    """Test the ListRobots method."""
    # Register multiple robots
    for i in range(3):
        register_request = fleet_manager_pb2.RegisterRobotRequest(
            robot_id=f"test-robot-list-{i}",
            robot_type="test-type",
            description=f"Test robot {i}",
            capabilities=["capability1", "capability2"],
            mcp=fleet_manager_pb2.MCPInfo(host="localhost", port=8080),
            deployment=fleet_manager_pb2.DeploymentInfo(docker_host="localhost", docker_port=2375),
            container=fleet_manager_pb2.ContainerConfig(
                image="test/image:latest",
                environment={"ENV1": "value1"}
            )
        )
        context = MockGrpcContext()
        await service_instance.RegisterRobot(register_request, context)
    
    # Create request to list all robots
    request = fleet_manager_pb2.ListRobotsRequest(
        filter=fleet_manager_pb2.ListRobotsRequest.ALL
    )
    
    # Call service method
    response = await service_instance.ListRobots(request, context)
    
    # Verify response contains all robots
    assert len(response.robots) >= 3
    assert any(robot.robot_id == "test-robot-list-0" for robot in response.robots)
    assert any(robot.robot_id == "test-robot-list-1" for robot in response.robots)
    assert any(robot.robot_id == "test-robot-list-2" for robot in response.robots)
    assert context.code == grpc.StatusCode.OK

@pytest.mark.asyncio
async def test_get_robot_status(service_instance):
    """Test the GetRobotStatus method."""
    # Register a robot
    register_request = fleet_manager_pb2.RegisterRobotRequest(
        robot_id="test-robot-status",
        robot_type="test-type",
        description="Test robot",
        capabilities=["capability1", "capability2"],
        mcp=fleet_manager_pb2.MCPInfo(host="localhost", port=8080),
        deployment=fleet_manager_pb2.DeploymentInfo(docker_host="localhost", docker_port=2375),
        container=fleet_manager_pb2.ContainerConfig(
            image="test/image:latest",
            environment={"ENV1": "value1"}
        )
    )
    context = MockGrpcContext()
    await service_instance.RegisterRobot(register_request, context)
    
    # Create request to get robot status
    request = fleet_manager_pb2.GetRobotStatusRequest(
        robot_id="test-robot-status"
    )
    
    # Call service method
    response = await service_instance.GetRobotStatus(request, context)
    
    # Verify response
    assert response.state == fleet_manager_pb2.RobotStatus.State.REGISTERED
    assert "test-robot-status" in response.message
    assert context.code == grpc.StatusCode.OK

@pytest.mark.asyncio
async def test_unregister_robot(service_instance):
    """Test the UnregisterRobot method."""
    # Register a robot
    register_request = fleet_manager_pb2.RegisterRobotRequest(
        robot_id="test-robot-unregister",
        robot_type="test-type",
        description="Test robot",
        capabilities=["capability1", "capability2"],
        mcp=fleet_manager_pb2.MCPInfo(host="localhost", port=8080),
        deployment=fleet_manager_pb2.DeploymentInfo(docker_host="localhost", docker_port=2375),
        container=fleet_manager_pb2.ContainerConfig(
            image="test/image:latest",
            environment={"ENV1": "value1"}
        )
    )
    context = MockGrpcContext()
    await service_instance.RegisterRobot(register_request, context)
    
    # Create request to unregister robot
    request = fleet_manager_pb2.UnregisterRobotRequest(
        robot_id="test-robot-unregister"
    )
    
    # Call service method
    response = await service_instance.UnregisterRobot(request, context)
    
    # Verify response
    assert response.success
    assert "test-robot-unregister" in response.message
    assert context.code == grpc.StatusCode.OK
    
    # Verify robot is actually unregistered by trying to get it
    get_request = fleet_manager_pb2.GetRobotStatusRequest(
        robot_id="test-robot-unregister"
    )
    get_context = MockGrpcContext()
    await service_instance.GetRobotStatus(get_request, get_context)
    assert get_context.code == grpc.StatusCode.NOT_FOUND

@pytest.mark.asyncio
async def test_create_goal(service_instance):
    """Test the CreateGoal method."""
    # Create request
    request = fleet_manager_pb2.CreateGoalRequest(
        description="Test goal"
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.CreateGoal(request, context)
    
    # Verify response
    assert response.goal.description == "Test goal"
    assert response.goal.goal_id > 0
    assert context.code == grpc.StatusCode.OK
    
    # Store goal ID for later tests
    goal_id = response.goal.goal_id
    return goal_id

@pytest.mark.asyncio
async def test_get_goal(service_instance):
    """Test the GetGoal method."""
    # Create a goal first
    goal_id = await test_create_goal(service_instance)
    
    # Create request to get goal
    request = fleet_manager_pb2.GetGoalRequest(
        goal_id=goal_id
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.GetGoal(request, context)
    
    # Verify response
    assert response.goal.goal_id == goal_id
    assert response.goal.description == "Test goal"
    assert context.code == grpc.StatusCode.OK

@pytest.mark.asyncio
async def test_list_goals(service_instance):
    """Test the ListGoals method."""
    # Create multiple goals
    for i in range(3):
        request = fleet_manager_pb2.CreateGoalRequest(
            description=f"Test goal {i}"
        )
        context = MockGrpcContext()
        await service_instance.CreateGoal(request, context)
    
    # Create request to list goals
    request = fleet_manager_pb2.ListGoalsRequest()
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.ListGoals(request, context)
    
    # Verify response
    assert len(response.goals) >= 3
    assert context.code == grpc.StatusCode.OK

@pytest.mark.asyncio
async def test_delete_goal(service_instance):
    """Test the DeleteGoal method."""
    # Create a goal first
    goal_id = await test_create_goal(service_instance)
    
    # Create request to delete goal
    request = fleet_manager_pb2.DeleteGoalRequest(
        goal_id=goal_id
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.DeleteGoal(request, context)
    
    # Verify response
    assert response.goal.goal_id == goal_id
    assert context.code == grpc.StatusCode.OK
    
    # Verify goal is actually deleted by trying to get it
    get_request = fleet_manager_pb2.GetGoalRequest(
        goal_id=goal_id
    )
    get_context = MockGrpcContext()
    get_response = await service_instance.GetGoal(get_request, get_context)
    assert get_context.code == grpc.StatusCode.NOT_FOUND

@pytest.mark.asyncio
async def test_create_task(service_instance):
    """Test the CreateTask method."""
    # Create a goal first
    goal_id = await test_create_goal(service_instance)
    
    # Register a robot
    register_request = fleet_manager_pb2.RegisterRobotRequest(
        robot_id="test-robot-task",
        robot_type="test-type",
        description="Test robot",
        capabilities=["capability1", "capability2"],
        mcp=fleet_manager_pb2.MCPInfo(host="localhost", port=8080),
        deployment=fleet_manager_pb2.DeploymentInfo(docker_host="localhost", docker_port=2375),
        container=fleet_manager_pb2.ContainerConfig(
            image="test/image:latest",
            environment={"ENV1": "value1"}
        )
    )
    context = MockGrpcContext()
    await service_instance.RegisterRobot(register_request, context)
    
    # Create request to create task
    request = fleet_manager_pb2.CreateTaskRequest(
        description="Test task",
        goal_id=goal_id,
        robot_id="test-robot-task",
        dependency_task_ids=[]
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.CreateTask(request, context)
    
    # Verify response
    assert response.task.description == "Test task"
    assert response.task.goal_id == goal_id
    assert response.task.robot_id == "test-robot-task"
    assert response.task.task_id > 0
    assert context.code == grpc.StatusCode.OK
    
    # Store task ID for later tests
    task_id = response.task.task_id
    return task_id

@pytest.mark.asyncio
async def test_get_task(service_instance):
    """Test the GetTask method."""
    # Create a task first
    task_id = await test_create_task(service_instance)
    
    # Create request to get task
    request = fleet_manager_pb2.GetTaskRequest(
        task_id=task_id
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.GetTask(request, context)
    
    # Verify response
    assert response.task.task_id == task_id
    assert response.task.description == "Test task"
    assert context.code == grpc.StatusCode.OK

@pytest.mark.asyncio
async def test_list_tasks(service_instance):
    """Test the ListTasks method."""
    # Create a goal
    goal_id = await test_create_goal(service_instance)
    
    # Register a robot
    register_request = fleet_manager_pb2.RegisterRobotRequest(
        robot_id="test-robot-list-tasks",
        robot_type="test-type",
        description="Test robot",
        capabilities=["capability1", "capability2"],
        mcp=fleet_manager_pb2.MCPInfo(host="localhost", port=8080),
        deployment=fleet_manager_pb2.DeploymentInfo(docker_host="localhost", docker_port=2375),
        container=fleet_manager_pb2.ContainerConfig(
            image="test/image:latest",
            environment={"ENV1": "value1"}
        )
    )
    context = MockGrpcContext()
    await service_instance.RegisterRobot(register_request, context)
    
    # Create multiple tasks
    for i in range(3):
        request = fleet_manager_pb2.CreateTaskRequest(
            description=f"Test task {i}",
            goal_id=goal_id,
            robot_id="test-robot-list-tasks",
            dependency_task_ids=[]
        )
        context = MockGrpcContext()
        await service_instance.CreateTask(request, context)
    
    # Create request to list tasks
    request = fleet_manager_pb2.ListTasksRequest(
        goal_ids=[goal_id],
        robot_ids=["test-robot-list-tasks"]
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.ListTasks(request, context)
    
    # Verify response
    assert len(response.tasks) >= 3
    assert all(task.goal_id == goal_id for task in response.tasks)
    assert all(task.robot_id == "test-robot-list-tasks" for task in response.tasks)
    assert context.code == grpc.StatusCode.OK

@pytest.mark.asyncio
async def test_create_plan(service_instance):
    """Test the CreatePlan method."""
    # Create a goal first
    goal_id = await test_create_goal(service_instance)
    
    # Create request
    request = fleet_manager_pb2.CreatePlanRequest(
        planning_strategy=fleet_manager_pb2.PlanningStrategy.DAG,
        goal_ids=[goal_id]
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.CreatePlan(request, context)
    
    # Verify response
    assert response.plan.planning_strategy == fleet_manager_pb2.PlanningStrategy.DAG
    assert goal_id in response.plan.goal_ids
    assert response.plan.plan_id > 0
    assert context.code == grpc.StatusCode.OK
    
    # Store plan ID for later tests
    plan_id = response.plan.plan_id
    return plan_id

@pytest.mark.asyncio
async def test_get_plan(service_instance):
    """Test the GetPlan method."""
    # Create a plan first
    plan_id = await test_create_plan(service_instance)
    
    # Create request to get plan
    request = fleet_manager_pb2.GetPlanRequest(
        plan_id=plan_id
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.GetPlan(request, context)
    
    # Verify response
    assert response.plan.plan_id == plan_id
    assert response.plan.planning_strategy == fleet_manager_pb2.PlanningStrategy.DAG
    assert context.code == grpc.StatusCode.OK

@pytest.mark.asyncio
async def test_list_plans(service_instance):
    """Test the ListPlans method."""
    # Create multiple plans
    for i in range(3):
        # Create a goal for each plan
        goal_request = fleet_manager_pb2.CreateGoalRequest(
            description=f"Goal for plan {i}"
        )
        goal_context = MockGrpcContext()
        goal_response = await service_instance.CreateGoal(goal_request, goal_context)
        
        # Create a plan
        plan_request = fleet_manager_pb2.CreatePlanRequest(
            planning_strategy=fleet_manager_pb2.PlanningStrategy.DAG,
            goal_ids=[goal_response.goal.goal_id]
        )
        plan_context = MockGrpcContext()
        await service_instance.CreatePlan(plan_request, plan_context)
    
    # Create request to list plans
    request = fleet_manager_pb2.ListPlansRequest()
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.ListPlans(request, context)
    
    # Verify response
    assert len(response.plans) >= 3
    assert context.code == grpc.StatusCode.OK

@pytest.mark.asyncio
async def test_delete_plan(service_instance):
    """Test the DeletePlan method."""
    # Create a plan first
    plan_id = await test_create_plan(service_instance)
    
    # Create request to delete plan
    request = fleet_manager_pb2.DeletePlanRequest(
        plan_id=plan_id
    )
    
    # Create mock context
    context = MockGrpcContext()
    
    # Call service method
    response = await service_instance.DeletePlan(request, context)
    
    # Verify response
    assert response.plan.plan_id == plan_id
    assert context.code == grpc.StatusCode.OK
    
    # Verify plan is actually deleted by trying to get it
    get_request = fleet_manager_pb2.GetPlanRequest(
        plan_id=plan_id
    )
    get_context = MockGrpcContext()
    get_response = await service_instance.GetPlan(get_request, get_context)
    assert get_context.code == grpc.StatusCode.NOT_FOUND

@pytest.mark.asyncio
async def test_task_with_dependencies(service_instance):
    """Test creating tasks with dependencies."""
    # Create a goal first
    goal_id = await test_create_goal(service_instance)
    
    # Create first task
    task1_request = fleet_manager_pb2.CreateTaskRequest(
        description="Task 1",
        goal_id=goal_id
    )
    task1_context = MockGrpcContext()
    task1_response = await service_instance.CreateTask(task1_request, task1_context)
    task1_id = task1_response.task.task_id
    
    # Create second task with dependency on first task
    task2_request = fleet_manager_pb2.CreateTaskRequest(
        description="Task 2",
        goal_id=goal_id,
        dependency_task_ids=[task1_id]
    )
    task2_context = MockGrpcContext()
    task2_response = await service_instance.CreateTask(task2_request, task2_context)
    
    # Verify dependency
    assert task1_id in task2_response.task.dependency_task_ids
    
    # Get second task and verify dependency is preserved
    get_request = fleet_manager_pb2.GetTaskRequest(
        task_id=task2_response.task.task_id
    )
    get_context = MockGrpcContext()
    get_response = await service_instance.GetTask(get_request, get_context)
    
    assert task1_id in get_response.task.dependency_task_ids

@pytest.mark.asyncio
async def test_service_error_handling(service_instance):
    """Test error handling in service methods."""
    # Test getting a non-existent robot
    robot_request = fleet_manager_pb2.GetRobotStatusRequest(
        robot_id="non-existent-robot"
    )
    robot_context = MockGrpcContext()
    await service_instance.GetRobotStatus(robot_request, robot_context)
    assert robot_context.code == grpc.StatusCode.NOT_FOUND
    
    # Test getting a non-existent goal
    goal_request = fleet_manager_pb2.GetGoalRequest(
        goal_id=9999
    )
    goal_context = MockGrpcContext()
    await service_instance.GetGoal(goal_request, goal_context)
    assert goal_context.code == grpc.StatusCode.NOT_FOUND
    
    # Test getting a non-existent task
    task_request = fleet_manager_pb2.GetTaskRequest(
        task_id=9999
    )
    task_context = MockGrpcContext()
    await service_instance.GetTask(task_request, task_context)
    assert task_context.code == grpc.StatusCode.NOT_FOUND
    
    # Test getting a non-existent plan
    plan_request = fleet_manager_pb2.GetPlanRequest(
        plan_id=9999
    )
    plan_context = MockGrpcContext()
    await service_instance.GetPlan(plan_request, plan_context)
    assert plan_context.code == grpc.StatusCode.NOT_FOUND

# --- World Statement API Tests ---
import random
import string

@pytest.mark.asyncio
async def test_add_get_list_delete_world_statement(service_instance):
    ctx = MockGrpcContext()
    # Add
    add_req = fleet_manager_pb2.AddWorldStatementRequest(statement="Test world statement")
    add_resp = await service_instance.AddWorldStatement(add_req, ctx)
    assert add_resp.error == ""
    assert add_resp.world_statement.statement == "Test world statement"
    ws_id = add_resp.world_statement.id

    # Get (success)
    get_req = fleet_manager_pb2.GetWorldStatementRequest(world_statement_id=ws_id)
    get_resp = await service_instance.GetWorldStatement(get_req, ctx)
    assert get_resp.error == ""
    assert get_resp.world_statement.id == ws_id

    # Get (not found)
    get_req_nf = fleet_manager_pb2.GetWorldStatementRequest(world_statement_id="99999")
    get_resp_nf = await service_instance.GetWorldStatement(get_req_nf, ctx)
    assert get_resp_nf.error != ""
    assert not get_resp_nf.HasField("world_statement")

    # Get (invalid ID)
    get_req_inv = fleet_manager_pb2.GetWorldStatementRequest(world_statement_id="abc")
    get_resp_inv = await service_instance.GetWorldStatement(get_req_inv, ctx)
    assert get_resp_inv.error != ""
    assert not get_resp_inv.HasField("world_statement")

    # List (should contain our statement)
    list_req = fleet_manager_pb2.ListWorldStatementsRequest()
    list_resp = await service_instance.ListWorldStatements(list_req, ctx)
    assert list_resp.error == ""
    assert any(ws.id == ws_id for ws in list_resp.world_statements)

    # Delete (success)
    del_req = fleet_manager_pb2.DeleteWorldStatementRequest(world_statement_id=ws_id)
    del_resp = await service_instance.DeleteWorldStatement(del_req, ctx)
    assert del_resp.success is True
    assert del_resp.error == ""

    # Delete (not found)
    del_req_nf = fleet_manager_pb2.DeleteWorldStatementRequest(world_statement_id="99999")
    del_resp_nf = await service_instance.DeleteWorldStatement(del_req_nf, ctx)
    assert del_resp_nf.success is False
    assert del_resp_nf.error != ""

    # Delete (invalid ID)
    del_req_inv = fleet_manager_pb2.DeleteWorldStatementRequest(world_statement_id="badid")
    del_resp_inv = await service_instance.DeleteWorldStatement(del_req_inv, ctx)
    assert del_resp_inv.success is False
    assert del_resp_inv.error != ""

    # List (should be empty again)
    list_resp2 = await service_instance.ListWorldStatements(list_req, ctx)
    assert list_resp2.error == ""
    assert len(list_resp2.world_statements) == 0
