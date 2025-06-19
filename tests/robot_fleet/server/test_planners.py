"""
Tests for the different planner implementations.

This test suite validates that all three planner types (Monolithic, DAG, and BigDAG)
generate plans with the expected structure and dependencies.

Before running tests, set your OpenAI API key in the .env file or export it as an environment variable:
export OPENAI_API_KEY=your_api_key
"""

import os
import sys
import json
import asyncio
import pytest
import pytest_asyncio
from typing import List, AsyncGenerator
from robot_fleet.robots.registry.models import Base


# Import environment variables from .env
import dotenv
dotenv.load_dotenv()

from robot_fleet.server.planner.planner import get_planner, BasePlanner
from robot_fleet.server.planner.formats.formats import Plan, DAGPlan, TaskPlanItem, DAGNode
from robot_fleet.proto.fleet_manager_pb2 import PlanningStrategy
from robot_fleet.robots.registry.instance_registry import RobotInstanceRegistry
from robot_fleet.robots.registry.models import Base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

# Use in-memory SQLite database for testing
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
async def registry(engine) -> AsyncGenerator[RobotInstanceRegistry, None]:
    """Create a registry instance with the test database."""
    registry = RobotInstanceRegistry(db_url=DATABASE_URL, engine=engine)
    await registry.initialize()
    yield registry

@pytest_asyncio.fixture(scope="function")
async def test_goals(registry) -> AsyncGenerator[List[int], None]:
    """Create test goals in the database and return their IDs."""
    # Create goals and extract the goal_id integers
    goal1_proto = await registry.create_goal(description="Test Goal 1")
    goal2_proto = await registry.create_goal(description="Test Goal 2")
    
    # Return the IDs as plain integers
    yield [goal1_proto.goal_id, goal2_proto.goal_id]

@pytest_asyncio.fixture(scope="function")
async def test_robots(registry) -> AsyncGenerator[List[str], None]:
    """Create test robots in the database and return their IDs."""
    robot1_id = "robot1"
    robot2_id = "robot2"
    
    # Register robots with proper capabilities
    await registry.register_robot(
        robot_id=robot1_id,
        robot_type="test_robot",
        description="Test Robot 1",
        capabilities=["navigation", "manipulation", "sensing", "grasping"]
    )
    
    await registry.register_robot(
        robot_id=robot2_id,
        robot_type="test_robot",
        description="Test Robot 2",
        capabilities=["navigation", "pick and place", "camera", "arm control"]
    )
    
    # Verify robots were registered properly
    robots = await registry.list_robots()
    assert len(robots) >= 2, "Test robots must be registered properly"
    
    yield [robot1_id, robot2_id]

@pytest_asyncio.fixture(scope="function")
async def monolithic_planner(engine) -> AsyncGenerator[BasePlanner, None]:
    """Create a MonolithicPlanner with the test database."""
    # Create the planner with the URL string
    planner = get_planner(PlanningStrategy.MONOLITHIC, DATABASE_URL)
    
    # Replace the registry with one that uses our test engine
    planner.registry = RobotInstanceRegistry(db_url=DATABASE_URL, engine=engine)
    
    # Initialize the registry
    await planner.registry.initialize()
    
    yield planner

@pytest_asyncio.fixture(scope="function")
async def dag_planner(engine) -> AsyncGenerator[BasePlanner, None]:
    """Create a DAGPlanner with the test database."""
    # Create the planner with the URL string
    planner = get_planner(PlanningStrategy.DAG, DATABASE_URL)
    
    # Replace the registry with one that uses our test engine
    planner.registry = RobotInstanceRegistry(db_url=DATABASE_URL, engine=engine)
    
    # Initialize the registry
    await planner.registry.initialize()
    
    yield planner

@pytest_asyncio.fixture(scope="function")
async def big_dag_planner(engine) -> AsyncGenerator[BasePlanner, None]:
    """Create a BigDAGPlanner with the test database."""
    # Create the planner with the URL string
    planner = get_planner(PlanningStrategy.BIG_DAG, DATABASE_URL)
    
    # Replace the registry with one that uses our test engine
    planner.registry = RobotInstanceRegistry(db_url=DATABASE_URL, engine=engine)
    
    # Initialize the registry
    await planner.registry.initialize()
    
    yield planner

# Skip all tests if OpenAI API key is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set. Set it to run these tests."
)

# Helper functions
def assert_valid_plan_format(plan_json: str):
    """Check that the plan JSON has the expected structure using Pydantic validation"""
    # Parse the JSON into a Plan model
    try:
        plan_dict = json.loads(plan_json)
        # Validate using Pydantic model
        plan = Plan.model_validate(plan_dict)
        
        # Additional checks
        assert len(plan.tasks) > 0, "Plan should contain at least one task"
        
        # Verify each task has the required fields (Pydantic would raise ValidationError if missing)
        for task in plan.tasks:
            assert isinstance(task.description, str), "Task description should be a string"
            assert isinstance(task.goal_id, int), "Goal ID should be an integer"
            assert isinstance(task.dependency_task_ids, list), "Dependencies should be a list"
        
        return plan
    except Exception as e:
        pytest.fail(f"Failed to validate plan format: {e}")

def assert_plan_dependencies(plan_json: str, expected_dependency_structure: str):
    """
    Validate that the plan has appropriate dependencies.
    
    expected_dependency_structure can be:
    - "sequential" - Each task depends only on the previous task
    - "dag_separated" - Separate DAGs for each goal with potential within-goal dependencies
    - "dag_unified" - Single unified DAG with potential cross-goal dependencies
    """
    try:
        plan_dict = json.loads(plan_json)
        plan = Plan.model_validate(plan_dict)
        
        # Group tasks by goal
        tasks_by_goal = {}
        for i, task in enumerate(plan.tasks):
            goal_id = task.goal_id
            if goal_id not in tasks_by_goal:
                tasks_by_goal[goal_id] = []
            tasks_by_goal[goal_id].append((i, task))
        
        # Validate dependency structure
        if expected_dependency_structure == "sequential":
            # For monolithic planner, each goal should have tasks that depend only on the previous task
            for goal_id, goal_tasks in tasks_by_goal.items():
                for i, (idx, task) in enumerate(goal_tasks):
                    if i == 0:
                        # First task should have no dependencies
                        assert len(task.dependency_task_ids) == 0, \
                            f"First task should have no dependencies, found {task.dependency_task_ids}"
                    else:
                        # Other tasks should depend only on the previous task
                        assert len(task.dependency_task_ids) == 1, \
                            f"Task should have exactly one dependency, found {task.dependency_task_ids}"
                        assert task.dependency_task_ids[0] == goal_tasks[i-1][0], \
                            f"Task should depend on previous task {goal_tasks[i-1][0]}, found {task.dependency_task_ids}"
        
        elif expected_dependency_structure == "dag_separated":
            # For DAG planner, tasks should only depend on tasks within the same goal
            for goal_id, goal_tasks in tasks_by_goal.items():
                goal_task_indices = {idx for idx, _ in goal_tasks}
                for _, task in goal_tasks:
                    for dep_idx in task.dependency_task_ids:
                        # All dependencies should be from the same goal
                        assert dep_idx in goal_task_indices, \
                            f"Task dependency {dep_idx} not found in goal group {goal_id}"
        
        elif expected_dependency_structure == "dag_unified":
            # For Big DAG planner, we should have at least one cross-goal dependency
            # if we have multiple goals
            if len(tasks_by_goal) > 1:
                cross_goal_deps_found = False
                for goal_id, goal_tasks in tasks_by_goal.items():
                    goal_task_indices = {idx for idx, _ in goal_tasks}
                    for idx, task in goal_tasks:
                        for dep_idx in task.dependency_task_ids:
                            if dep_idx not in goal_task_indices:
                                cross_goal_deps_found = True
                                break
                        if cross_goal_deps_found:
                            break
                    if cross_goal_deps_found:
                        break
                
                # If we have multiple goals, big DAG should optimize with cross-goal dependencies
                # But this is not guaranteed by the LLM, so just print a warning
                if not cross_goal_deps_found:
                    print("Warning: BigDAG planner did not create any cross-goal dependencies")
            
    except Exception as e:
        pytest.fail(f"Failed to validate plan dependencies: {e}")

# Tests for each planner type
@pytest.mark.asyncio
async def test_monolithic_planner(monolithic_planner, test_goals, test_robots):
    """Test MonolithicPlanner generates sequential tasks for each goal"""
    # Monolithic planner only works with a single goal
    single_goal_id = [test_goals[0]]  # Use first goal only, as a list with one item
    
    # Ensure robots are available for planning
    assert len(test_robots) > 0, "Test robots must be available for planning"
    
    # Run the planner
    plan_json = await monolithic_planner.plan(single_goal_id)
    
    # Validate the plan format
    assert_valid_plan_format(plan_json)
    
    # Validate dependencies are sequential
    assert_plan_dependencies(plan_json, "sequential")
    
    # Test plan is saved to database
    await monolithic_planner.save_plan_to_db(
        plan_json, 
        planning_strategy=PlanningStrategy.MONOLITHIC,
        goal_ids=single_goal_id
    )

@pytest.mark.asyncio
async def test_dag_planner(dag_planner, test_goals, test_robots):
    """Test DAGPlanner generates separate DAGs for each goal"""
    # Ensure robots are available for planning
    assert len(test_robots) > 0, "Test robots must be available for planning"
    
    # Run the planner
    plan_json = await dag_planner.plan(test_goals)
    
    # Validate the plan format
    assert_valid_plan_format(plan_json)
    
    # Validate dependencies follow DAG structure with no cross-goal dependencies
    assert_plan_dependencies(plan_json, "dag_separated")
    
    # Test plan is saved to database
    await dag_planner.save_plan_to_db(
        plan_json,
        planning_strategy=PlanningStrategy.DAG,
        goal_ids=test_goals
    )

@pytest.mark.asyncio
async def test_big_dag_planner(big_dag_planner, test_goals, test_robots):
    """Test BigDAGPlanner generates a unified DAG across goals"""
    # Ensure robots are available for planning
    assert len(test_robots) > 0, "Test robots must be available for planning"
    
    # Run the planner
    plan_json = await big_dag_planner.plan(test_goals)
    
    # Validate the plan format
    assert_valid_plan_format(plan_json)
    
    # Validate dependencies follow unified DAG structure
    assert_plan_dependencies(plan_json, "dag_unified")
    
    # Test plan is saved to database
    await big_dag_planner.save_plan_to_db(
        plan_json,
        planning_strategy=PlanningStrategy.BIG_DAG,
        goal_ids=test_goals
    )

# Test get_planner returns the right planner type
@pytest.mark.asyncio
async def test_get_planner_types():
    """Test get_planner creates the correct planner types"""
    monolithic = get_planner(PlanningStrategy.MONOLITHIC, DATABASE_URL)
    assert monolithic.__class__.__name__ == "MonolithicPlanner"
    
    dag = get_planner(PlanningStrategy.DAG, DATABASE_URL)
    assert dag.__class__.__name__ == "DAGPlanner"
    
    big_dag = get_planner(PlanningStrategy.BIG_DAG, DATABASE_URL)  
    assert big_dag.__class__.__name__ == "BigDAGPlanner"

# Test error cases
@pytest.mark.asyncio
async def test_invalid_strategy():
    """Test get_planner raises an error for invalid strategy"""
    with pytest.raises(ValueError):
        get_planner(999, DATABASE_URL)  # Invalid strategy
