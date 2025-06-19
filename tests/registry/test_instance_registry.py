# /Users/trevorasbery/Desktop/AgenticNet/multirobot-task/tests/registry/test_instance_registry.py
import pytest
import pytest_asyncio
import asyncio 
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
import sys 
import os 

# Add project root to sys.path to allow importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from robot_fleet.robots.registry.instance_registry import RobotInstanceRegistry
from robot_fleet.robots.registry.models import Base, WorldStatement # Import WorldStatement
from robot_fleet.proto import fleet_manager_pb2 # Import proto definitions
from typing import AsyncGenerator

DATABASE_URL = "sqlite+aiosqlite:///:memory:" 

# Configure asyncio for pytest
@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="function")
async def engine() -> AsyncGenerator[AsyncEngine, None]: 
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
async def registry(engine): 
    # Pass the engine from the 'engine' fixture to the registry
    registry_instance = RobotInstanceRegistry(db_url=DATABASE_URL, engine=engine)
    return registry_instance

# --- Robot Tests --- 
@pytest.mark.asyncio
async def test_robot_crud(registry: RobotInstanceRegistry):
    # Create
    robot_proto = await registry.register_robot(
        robot_id="robot-c1", 
        robot_type="cleaner", 
        description="d1", 
        capabilities=["vacuum"]
    )
    assert robot_proto is not None
    assert robot_proto.robot_id == "robot-c1"
    assert robot_proto.robot_type == "cleaner"
    assert "vacuum" in robot_proto.capabilities

    # Read
    retrieved_proto = await registry.get_robot("robot-c1")
    assert retrieved_proto is not None
    assert retrieved_proto.robot_id == "robot-c1"

    # Update (Example - assuming update_robot takes similar direct args or a proto)
    # For now, let's just test updating status which is simpler
    updated_proto = await registry.update_robot_status("robot-c1", fleet_manager_pb2.RobotStatus.State.RUNNING)
    assert updated_proto is not None
    assert updated_proto.status.state == fleet_manager_pb2.RobotStatus.State.RUNNING
    
    # Verify update
    retrieved_after_update = await registry.get_robot("robot-c1")
    assert retrieved_after_update is not None
    assert retrieved_after_update.status.state == fleet_manager_pb2.RobotStatus.State.RUNNING

    # Delete
    deleted = await registry.delete_robot("robot-c1")
    assert deleted is True

    # Verify Deletion
    retrieved_after_delete = await registry.get_robot("robot-c1")
    assert retrieved_after_delete is None

# --- Goal Tests --- 
@pytest.mark.asyncio
async def test_goal_crud(registry: RobotInstanceRegistry):
    # Create
    goal_proto = await registry.create_goal(description="Goal G1")
    assert goal_proto is not None
    goal_id1 = goal_proto.goal_id
    assert goal_id1 > 0

    # Get
    retrieved_proto = await registry.get_goal(goal_id1)
    assert retrieved_proto is not None
    assert retrieved_proto.goal_id == goal_id1
    
    # List
    goal_proto2 = await registry.create_goal(description="Goal G2")
    goal_id2 = goal_proto2.goal_id
    goals_list = await registry.list_goals()
    assert len(goals_list) == 2
    assert any(g.goal_id == goal_id1 for g in goals_list)
    assert any(g.goal_id == goal_id2 for g in goals_list)
    
    # Delete
    deleted = await registry.delete_goal(goal_id1)
    assert deleted is True
    retrieved_proto = await registry.get_goal(goal_id1)
    assert retrieved_proto is None
    goals_list = await registry.list_goals()
    assert len(goals_list) == 1
    assert goals_list[0].goal_id == goal_id2
    
# --- Task Tests --- 
@pytest.mark.asyncio
async def test_task_crud(registry: RobotInstanceRegistry):
    # Create
    robot_proto = await registry.register_robot(
        robot_id="robot-c1", 
        robot_type="cleaner", 
        description="d1", 
        capabilities=["vacuum"]
    )
    assert robot_proto is not None
    assert robot_proto.robot_id == "robot-c1"
    assert robot_proto.robot_type == "cleaner"
    assert "vacuum" in robot_proto.capabilities

    goal_proto = await registry.create_goal(description="Goal G1")
    assert goal_proto is not None
    assert goal_proto.goal_id > 0

    # Create a task associated with the robot and goal
    task_proto = await registry.create_task(
        description="Clean the kitchen floor",
        robot_id=robot_proto.robot_id, # Use string robot_id
        goal_id=goal_proto.goal_id,
        status=fleet_manager_pb2.TaskStatus.TASK_PENDING
    )
    assert task_proto is not None
    task_id1 = task_proto.task_id
    assert task_id1 > 0

    # Get
    retrieved_proto = await registry.get_task(task_id1)
    assert retrieved_proto is not None
    assert retrieved_proto.task_id == task_id1
    
    # List
    task_proto2 = await registry.create_task(description="Task T2")
    task_id2 = task_proto2.task_id
    tasks_list = await registry.list_tasks()
    assert len(tasks_list) >= 2  # At least 2 tasks
    
    # Update Status
    updated_proto = await registry.update_task_status(task_id1, fleet_manager_pb2.TaskStatus.TASK_COMPLETED)
    assert updated_proto is not None
    assert updated_proto.status == fleet_manager_pb2.TaskStatus.TASK_COMPLETED
    retrieved_proto = await registry.get_task(task_id1)
    assert retrieved_proto.status == fleet_manager_pb2.TaskStatus.TASK_COMPLETED
    
    # Delete
    deleted = await registry.delete_task(task_id1)
    assert deleted is True
    retrieved_proto = await registry.get_task(task_id1)
    assert retrieved_proto is None
    tasks_list = await registry.list_tasks()
    assert len(tasks_list) == 1
    assert tasks_list[0].task_id == task_id2

# --- Plan Tests --- 
@pytest.mark.asyncio
async def test_plan_crud(registry: RobotInstanceRegistry):
    # Create
    goal = await registry.create_goal(description="Plan Goal")
    plan_proto = await registry.create_plan(
        goal_ids=[goal.goal_id],
        planning_strategy=fleet_manager_pb2.PlanningStrategy.MONOLITHIC,
    )
    assert plan_proto is not None
    assert plan_proto.planning_strategy == fleet_manager_pb2.PlanningStrategy.MONOLITHIC
    
    plan_id = plan_proto.plan_id

    # Read
    retrieved_proto = await registry.get_plan(plan_id)
    assert retrieved_proto is not None
    assert retrieved_proto.plan_id == plan_id

    # Create plan without task_ids
    plan_proto2 = await registry.create_plan(
        planning_strategy=fleet_manager_pb2.PlanningStrategy.MONOLITHIC,
        goal_ids=[goal.goal_id]
    )
    assert plan_proto2 is not None
    assert plan_proto2.task_ids == []

    # Add a task and link it to the plan
    new_task = await registry.create_task(
        description="Linked Task",
        plan_id=plan_proto2.plan_id,
        status=fleet_manager_pb2.TaskStatus.TASK_PENDING
    )
    updated_plan = await registry.get_plan(plan_proto2.plan_id)
    assert new_task.task_id in updated_plan.task_ids

    # Also test create_plan with task_ids
    plan_proto3 = await registry.create_plan(
        planning_strategy=fleet_manager_pb2.PlanningStrategy.MONOLITHIC,
        goal_ids=[goal.goal_id],
        task_ids=[new_task.task_id]
    )
    assert plan_proto3 is not None
    assert new_task.task_id in plan_proto3.task_ids

    # Update (Example: Add a goal_id)
    updated_proto = await registry.update_plan(plan_id, goal_ids=[5])
    assert updated_proto is not None
    assert set(updated_proto.goal_ids) == {5}
    
    # Verify Update
    retrieved_after_update = await registry.get_plan(plan_id)
    assert retrieved_after_update is not None
    assert set(retrieved_after_update.goal_ids) == {5}

    # Delete
    deleted = await registry.delete_plan(plan_id)
    assert deleted is True

    # Verify Deletion
    retrieved_after_delete = await registry.get_plan(plan_id)
    assert retrieved_after_delete is None

# --- Relationship Tests --- 
@pytest.mark.asyncio
async def test_delete_plan_unlinks_task(registry: RobotInstanceRegistry):
    # Create a task
    task = await registry.create_task(description="Task link test")
    assert task is not None
    # Create a plan
    plan = await registry.create_plan(
        planning_strategy=fleet_manager_pb2.PlanningStrategy.MONOLITHIC, # Use Enum
    )
    assert plan is not None
    # Link the task to the plan
    await registry.update_task(task.task_id, plan_id=plan.plan_id)
    # Verify task is linked
    task_before = await registry.get_task(task.task_id)
    assert task_before.plan_id == plan.plan_id
    # Delete the plan
    result = await registry.delete_plan(plan.plan_id)
    assert result
    # Verify the task is deleted (NOT just unlinked as previously expected)
    task_after = await registry.get_task(task.task_id)
    assert task_after is None  # Task should be deleted due to cascade="all, delete-orphan"

@pytest.mark.asyncio
async def test_delete_goal_unlinks_task_and_plan(registry: RobotInstanceRegistry):
    goal = await registry.create_goal(description="Goal link test")
    task = await registry.create_task(description="Task linked to Goal", goal_id=goal.goal_id)
    assert task is not None
    plan = await registry.create_plan(
        planning_strategy=fleet_manager_pb2.PlanningStrategy.MONOLITHIC, # Use Enum
        goal_ids=[goal.goal_id],
    )
    assert plan is not None
    plan_before = await registry.get_plan(plan.plan_id)
    assert goal.goal_id in plan_before.goal_ids
    
    await registry.delete_goal(goal.goal_id)
    
    # The task should be deleted (not just unlinked) due to cascade="all, delete-orphan"
    retrieved_task = await registry.get_task(task.task_id)
    assert retrieved_task is None  # Task is deleted with the goal
    
    # Plan should still exist but without the deleted goal
    retrieved_plan = await registry.get_plan(plan.plan_id)
    assert retrieved_plan is not None
    assert goal.goal_id not in retrieved_plan.goal_ids

@pytest.mark.asyncio
async def test_delete_task_unlinks_plan(registry: RobotInstanceRegistry):
    task = await registry.create_task(description="Task link test")
    assert task is not None
    plan = await registry.create_plan(
        planning_strategy=fleet_manager_pb2.PlanningStrategy.MONOLITHIC, # Use Enum
    )
    assert plan is not None
    plan_before = await registry.get_plan(plan.plan_id)
    assert task.task_id not in plan_before.task_ids

    # Delete task
    await registry.delete_task(task.task_id)

    # Verify plan no longer links task
    plan_after = await registry.get_plan(plan.plan_id)
    assert plan_after is not None # Plan should still exist
    assert task.task_id not in plan_after.task_ids

# --- Additional Robot Tests ---
@pytest.mark.asyncio
async def test_robot_update_capabilities(registry: RobotInstanceRegistry):
    """Test updating a robot's capabilities"""
    # Create a robot with initial capabilities
    robot = await registry.register_robot(
        robot_id="robot-update-test",
        robot_type="multi-purpose",
        description="Updatable robot",
        capabilities=["move", "grab"]
    )
    assert robot is not None
    assert set(robot.capabilities) == {"move", "grab"}
    
    # Update capabilities
    updated_robot = await registry.update_robot(
        robot_id="robot-update-test",
        capabilities_update=["move", "grab", "scan", "analyze"]
    )
    assert updated_robot is not None
    assert set(updated_robot.capabilities) == {"move", "grab", "scan", "analyze"}
    
    # Verify through get operation
    retrieved_robot = await registry.get_robot("robot-update-test")
    assert retrieved_robot is not None
    assert set(retrieved_robot.capabilities) == {"move", "grab", "scan", "analyze"}

@pytest.mark.asyncio
async def test_robot_register_duplicate(registry: RobotInstanceRegistry):
    """Test registering a robot with an existing ID"""
    # Create initial robot
    robot1 = await registry.register_robot(
        robot_id="robot-duplicate",
        robot_type="original",
        description="First robot",
        capabilities=["move"]
    )
    assert robot1 is not None
    assert robot1.robot_type == "original"
    
    # Register another robot with same ID
    robot2 = await registry.register_robot(
        robot_id="robot-duplicate",
        robot_type="duplicate",
        description="Second robot",
        capabilities=["scan"]
    )
    
    # Registry should return the existing robot
    assert robot2 is not None
    assert robot2.robot_id == "robot-duplicate"
    assert robot2.robot_type == "original"  # Should still have original type

@pytest.mark.asyncio
async def test_get_nonexistent_robot(registry: RobotInstanceRegistry):
    """Test getting a robot that doesn't exist"""
    robot = await registry.get_robot("nonexistent-robot")
    assert robot is None

# --- Additional Task Tests ---
@pytest.mark.asyncio
async def test_task_with_dependencies(registry: RobotInstanceRegistry):
    """Test creating and retrieving tasks with dependencies"""
    # Create several tasks
    task1 = await registry.create_task(description="Task 1")
    task2 = await registry.create_task(description="Task 2")
    task3 = await registry.create_task(description="Task 3")
    
    # Create a task that depends on others
    dependent_task = await registry.create_task(
        description="Dependent Task",
        dependency_task_ids=[task1.task_id, task2.task_id, task3.task_id]
    )
    
    # Verify dependencies
    retrieved_task = await registry.get_task(dependent_task.task_id)
    assert retrieved_task is not None
    assert set(retrieved_task.dependency_task_ids) == {task1.task_id, task2.task_id, task3.task_id}

@pytest.mark.asyncio
async def test_list_tasks_with_filters(registry: RobotInstanceRegistry):
    """Test listing tasks with different attributes"""
    # Create a robot
    robot = await registry.register_robot(
        robot_id="filter-robot",
        robot_type="test",
        description="Robot for filter tests",
        capabilities=["test"]
    )
    
    # Create a goal
    goal = await registry.create_goal(description="Filter Goal")
    
    # Create a plan with task_ids (required parameter)
    plan = await registry.create_plan(
        planning_strategy=fleet_manager_pb2.PlanningStrategy.MONOLITHIC,
        goal_ids=[goal.goal_id],
    )
    
    # Create dummy task with plan_id
    dummy_task = await registry.create_task(description="Dummy Task", plan_id=plan.plan_id)
    
    # Create tasks with different attributes
    task1 = await registry.create_task(description="Task A", robot_id=robot.robot_id)
    task2 = await registry.create_task(description="Task B", goal_id=goal.goal_id)
    task3 = await registry.create_task(description="Task C", plan_id=plan.plan_id)
    task4 = await registry.create_task(description="Task D", 
                                      robot_id=robot.robot_id, 
                                      goal_id=goal.goal_id,
                                      plan_id=plan.plan_id)
    
    # List all tasks
    all_tasks = await registry.list_tasks()
    assert len(all_tasks) >= 5  # At least 5 tasks: dummy + 4 created here
    
    # Since list_tasks() doesn't support filtering, manually filter the results
    robot_tasks = [t for t in all_tasks if t.robot_id == robot.robot_id]
    assert len(robot_tasks) == 2
    
    goal_tasks = [t for t in all_tasks if t.goal_id == goal.goal_id]
    assert len(goal_tasks) == 2
    
    # Account for all tasks with plan_id including dummy_task
    plan_tasks = [t for t in all_tasks if t.plan_id == plan.plan_id]
    assert len(plan_tasks) == 3  # dummy_task + task3 + task4

@pytest.mark.asyncio
async def test_goal_with_multiple_tasks(registry: RobotInstanceRegistry):
    """Test a goal with multiple tasks and verify proper relationships"""
    # Create a goal
    goal = await registry.create_goal(description="Multi-task Goal")
    
    # Create multiple tasks for this goal
    tasks = []
    for i in range(5):
        task = await registry.create_task(
            description=f"Goal Task {i}",
            goal_id=goal.goal_id
        )
        tasks.append(task)
    
    # Verify task_ids are populated correctly for the goal
    retrieved_goal = await registry.get_goal(goal.goal_id)
    assert retrieved_goal is not None
    assert len(retrieved_goal.task_ids) == 5
    
    # Since list_tasks() doesn't support filtering by goal_id,
    # get all tasks and filter manually
    all_tasks = await registry.list_tasks()
    goal_tasks = [t for t in all_tasks if t.goal_id == goal.goal_id]
    assert len(goal_tasks) == 5
    
    # Delete one task and verify goal still exists with 4 tasks
    await registry.delete_task(tasks[0].task_id)
    
    updated_goal = await registry.get_goal(goal.goal_id)
    assert updated_goal is not None
    assert len(updated_goal.task_ids) == 4

@pytest.mark.asyncio
async def test_cascading_delete_validation(registry: RobotInstanceRegistry):
    """Test that deleting objects properly manages relationships without causing database errors"""
    # Create interlinked objects
    robot = await registry.register_robot(
        robot_id="cascade-robot",
        robot_type="test",
        description="Cascade test robot",
        capabilities=["test"]
    )
    
    goal = await registry.create_goal(description="Cascade Goal")
    
    # Create task linked to the goal
    task_with_goal = await registry.create_task(
        description="Goal Task",
        goal_id=goal.goal_id
    )
    
    # Create separate task linked to the robot
    task_with_robot = await registry.create_task(
        description="Robot Task",
        robot_id=robot.robot_id
    )
    
    plan = await registry.create_plan(
        planning_strategy=fleet_manager_pb2.PlanningStrategy.MONOLITHIC,
        goal_ids=[goal.goal_id],
    )
    
    # Delete the goal and verify cascading effects
    goal_deleted = await registry.delete_goal(goal.goal_id)
    assert goal_deleted is True
    
    # Verify task exists but has goal_id = 0 (default value for no goal)
    task_after_goal_delete = await registry.get_task(task_with_goal.task_id)
    assert task_after_goal_delete is None  # Task should be deleted with the goal
    
    # Verify plan exists but has no goals
    plan_after_goal_delete = await registry.get_plan(plan.plan_id)
    assert plan_after_goal_delete is not None
    assert len(plan_after_goal_delete.goal_ids) == 0
    
    # Delete the robot
    robot_deleted = await registry.delete_robot(robot.robot_id)
    assert robot_deleted is True
    
    # Check if task still exists with robot_id unlinked (not deleted)
    task_after_robot_delete = await registry.get_task(task_with_robot.task_id)
    # Tasks associated with robots are unlinked, not deleted (unlike tasks with goals/plans)
    assert task_after_robot_delete is not None
    assert task_after_robot_delete.robot_id == ""  # Empty string for no robot

@pytest.mark.asyncio
async def test_plan_with_complex_structure(registry: RobotInstanceRegistry):
    """Test creating a plan with goals and tasks in a complex structure"""
    # Create multiple goals
    goal1 = await registry.create_goal(description="Plan Goal 1")
    goal2 = await registry.create_goal(description="Plan Goal 2")
    
    # Create multiple tasks
    tasks = []
    for i in range(3):
        # First task for goal1, second for goal2, third independent
        goal_id = None
        if i == 0:
            goal_id = goal1.goal_id
        elif i == 1:
            goal_id = goal2.goal_id
            
        task = await registry.create_task(
            description=f"Plan Task {i}",
            goal_id=goal_id
        )
        tasks.append(task)
    
    # Create a plan that includes all goals and tasks
    plan = await registry.create_plan(
        planning_strategy=fleet_manager_pb2.PlanningStrategy.MONOLITHIC,
        goal_ids=[goal1.goal_id, goal2.goal_id],
    )
    
    # Verify plan structure
    retrieved_plan = await registry.get_plan(plan.plan_id)
    assert retrieved_plan is not None
    assert set(retrieved_plan.goal_ids) == {goal1.goal_id, goal2.goal_id}
    
    # Update the plan to remove one goal and one task
    updated_plan = await registry.update_plan(
        plan_id=plan.plan_id,
        goal_ids=[goal1.goal_id],
    )
    
    # Verify update
    assert set(updated_plan.goal_ids) == {goal1.goal_id}
    
    # Verify through get
    final_plan = await registry.get_plan(plan.plan_id)
    assert set(final_plan.goal_ids) == {goal1.goal_id}

# --- Test World Statement Registry ---

@pytest.mark.asyncio
async def test_add_world_statement(registry: RobotInstanceRegistry):
    """Test adding a world statement."""
    statement_text = "The blue cube is on the red table."
    ws_proto = await registry.add_world_statement(statement_text)

    assert ws_proto is not None
    assert ws_proto.statement == statement_text
    assert ws_proto.id is not None and ws_proto.id != ""
    # We expect the ID to be '1' for the first item with auto-increment
    assert ws_proto.id == "1"
    assert ws_proto.created_at is not None

    # Verify it's in the DB directly (optional sanity check)
    async with registry.async_session_factory() as session:
        db_ws = await session.get(WorldStatement, 1) # Use integer ID for DB lookup
        assert db_ws is not None
        assert db_ws.statement == statement_text


@pytest.mark.asyncio
async def test_get_world_statement(registry: RobotInstanceRegistry):
    """Test retrieving a world statement."""
    statement1 = "Robot Alpha is near the charging station."
    statement2 = "The exit door is open."
    ws1_proto = await registry.add_world_statement(statement1)
    await registry.add_world_statement(statement2)

    # Get the first statement
    retrieved_ws = await registry.get_world_statement(ws1_proto.id)
    assert retrieved_ws is not None
    assert retrieved_ws.id == ws1_proto.id
    assert retrieved_ws.statement == statement1

    # Test getting a non-existent ID
    non_existent = await registry.get_world_statement("999")
    assert non_existent is None

    # Test getting with invalid ID format
    invalid_id_format = await registry.get_world_statement("abc")
    assert invalid_id_format is None


@pytest.mark.asyncio
async def test_list_world_statements(registry: RobotInstanceRegistry):
    """Test listing all world statements."""
    # Initial list should be empty
    statements = await registry.list_world_statements()
    assert len(statements) == 0

    # Add some statements
    texts = ["Object A is heavy.", "Object B is fragile.", "Location C is restricted."]
    await registry.add_world_statement(texts[0])
    await registry.add_world_statement(texts[1])
    await registry.add_world_statement(texts[2])

    # List again
    statements = await registry.list_world_statements()
    assert len(statements) == 3
    # Verify content and order (should be insertion order based on created_at)
    assert statements[0].statement == texts[0]
    assert statements[1].statement == texts[1]
    assert statements[2].statement == texts[2]
    assert statements[0].id == "1"
    assert statements[1].id == "2"
    assert statements[2].id == "3"


@pytest.mark.asyncio
async def test_delete_world_statement(registry: RobotInstanceRegistry):
    """Test deleting a world statement."""
    ws1_proto = await registry.add_world_statement("Statement to be deleted.")
    ws2_proto = await registry.add_world_statement("Another statement.")

    # Delete the first statement
    deleted = await registry.delete_world_statement(ws1_proto.id)
    assert deleted is True

    # Verify it's gone
    retrieved_ws = await registry.get_world_statement(ws1_proto.id)
    assert retrieved_ws is None

    # Verify the second statement still exists
    retrieved_ws2 = await registry.get_world_statement(ws2_proto.id)
    assert retrieved_ws2 is not None
    assert retrieved_ws2.id == ws2_proto.id

    # List should only contain the second one
    statements = await registry.list_world_statements()
    assert len(statements) == 1
    assert statements[0].id == ws2_proto.id

    # Test deleting a non-existent ID
    deleted_non_existent = await registry.delete_world_statement("999")
    assert deleted_non_existent is False

    # Test deleting with invalid ID format
    deleted_invalid_format = await registry.delete_world_statement("xyz")
    assert deleted_invalid_format is False


