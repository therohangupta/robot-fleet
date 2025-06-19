import os
from openai import OpenAI
import json
import asyncio
import re
from enum import Enum
from typing import List, Dict, Optional, Any, Set
from abc import ABC, abstractmethod
from robot_fleet.robots.registry.instance_registry import RobotInstanceRegistry
from .formats.formats import Plan, DAGPlan, TaskPlanItem, Allocation, RobotTask
import logging
from ..allocator.allocator import AllocatorBase, LLMAllocator, LPAllocator
from dotenv import load_dotenv
from pathlib import Path

# Determine the base directory for prompts
# PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts") # Old path
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]  # Adjust depth as needed
PROMPT_DIR = os.path.join(WORKSPACE_ROOT, "fleet_rules/prompts") # New path at workspace root

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)

class BasePlanner(ABC):
    """Abstract base class for all planners"""
    
    def __init__(self, db_url: Optional[str] = None):

        if not db_url:
            db_url = "postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet"
            print(f"Using default PostgreSQL database: {db_url}")
        else:
             print(f"Using database: {db_url}")
        self.registry = RobotInstanceRegistry(db_url)
        # Load world statements on initialization
        logger.info(f"Initialized {self.__class__.__name__}.")

    async def _load_world_statements(self) -> List[str]:
        world_statements = await self.registry.list_world_statements()
        return [ws.statement for ws in world_statements]

    async def _load_capabilities(self) -> Set[str]:
        """
        Load the union of all robot capabilities in the system.
        Returns a set of unique capabilities across all robots.
        """
        robots = await self.registry.list_robots()
        if not robots:
            raise ValueError("No robots found in the registry.")
        
        # Create a set of all unique capabilities across all robots
        return set().union(*(set(robot.capabilities) for robot in robots))
    
    def _load_prompt_template(self, filename: str) -> str:
        """Loads a prompt template from the prompts directory."""
        filepath = os.path.join(PROMPT_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt file {filepath}: {e}")
            raise

    async def _get_robot_context_string(self) -> str:
        """
        Fetches robot details and formats them into a string for the LLM context.
        Groups robots by type and lists their count and capabilities.
        """
        robots = await self.registry.list_robots()
        if not robots:
            return "No robots available in the fleet."

        robots_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for robot in robots:
            if robot.robot_type not in robots_by_type:
                robots_by_type[robot.robot_type] = []
            robots_by_type[robot.robot_type].append({
                "id": robot.robot_id,
                "capabilities": robot.capabilities
            })

        context_lines = ["AVAILABLE ROBOTS AND CAPABILITIES:"]
        for robot_type, instances in robots_by_type.items():
            count = len(instances)
            # Assuming all robots of the same type have the same capabilities for simplicity in the summary
            # If capabilities can vary within a type, this might need adjustment
            capabilities_str = ", ".join(sorted(list(set(instances[0]['capabilities'])))) if instances else "None"
            context_lines.append(f"- {count} {robot_type} robot(s) with capabilities: [{capabilities_str}]")
        
        return "\\n".join(context_lines)

    @abstractmethod
    async def plan(self, goal_ids: List[int]) -> str:
        """
        Generate a plan for the given goals. Returns a JSON string 
        conforming to the Plan schema.
        
        Args:
            goal_ids: List of goal IDs to include in the plan
            
        Returns:
            JSON string representation of the plan
        """
        pass

    # @abstractmethod
    # async def replan(self, plan_id: int, failed_task_id: int, failure_message: str, robot_task_assignments: Dict[int, int]) -> str:
    #     """
    #     Generate a new plan segment to recover from a failure in an existing plan.

    #     Args:
    #         plan_id: The ID of the plan that failed.
    #         failed_task_id: The ID of the task that failed.
    #         failure_message: A string describing the failure.
    #         robot_task_assignments: A dictionary mapping robot_id to the task_id they were assigned at the time of failure.

    #     Returns:
    #         JSON string representation of the new plan segment (in Plan format).
    #     """
    #     pass
    
    async def save_plan_to_db(self, plan_json: str, planning_strategy: int, allocation_strategy: int, goal_ids: List[int]) -> int:
        """
        Parse the plan JSON and save the plan and tasks to the database.
        
        Args:
            plan_json: JSON string representation of the plan
            planning_strategy: The planning strategy enum value
            allocation_strategy: The allocation strategy enum value
            goal_ids: List of goal IDs included in the plan
            
        Returns:
            The ID of the created plan
        """
        try:
            # First create the plan with the correct strategy
            plan = await self.registry.create_plan(
                planning_strategy=planning_strategy,
                allocation_strategy=allocation_strategy,
                goal_ids=goal_ids,
                task_ids=[]  # Will be populated later
            )
            
            if not plan:
                raise ValueError(f"Failed to create plan for goals: {goal_ids}")
            # breakpoint()
            plan_id = plan.plan_id
            
            # Parse the JSON
            plan_data = json.loads(plan_json)
            
            # Extract tasks list
            tasks = plan_data.get("tasks", [])
            
            if not tasks:
                print("❌ No tasks found in the plan")
                return plan_id
                
            # First pass: Create all tasks and get their IDs
            task_ids = []
            task_index_to_id = {}  # Map from task index to DB ID
            
            for i, task_data in enumerate(tasks):
                description = task_data.get("description")
                task_goal_id = task_data.get("goal_id")
                robot_type = task_data.get("robot_type")
                
                if description and task_goal_id is not None:
                    # Create the task using instance_registry
                    task_proto = await self.registry.create_task(
                        description=description,
                        goal_id=task_goal_id,
                        plan_id=plan_id,  # Link to our plan
                        robot_type=robot_type # Pass robot_type to create_task
                    )
                    task_ids.append(task_proto.task_id)
                    task_index_to_id[i] = task_proto.task_id
                    print(f"✅ Added task: {description} for goal {task_goal_id} (type: {robot_type})")
                else:
                    print(f"Skipping invalid task: {task_data}")
            
            # Second pass: Update dependencies now that we have IDs
            for i, task_data in enumerate(tasks):
                dependency_indices = task_data.get("dependency_task_ids", [])
                if dependency_indices:
                    # Convert indices to actual task IDs
                    dependency_ids = [task_index_to_id[idx] for idx in dependency_indices if idx in task_index_to_id]
                    if dependency_ids and i in task_index_to_id:
                        task_id = task_index_to_id[i]
                        await self.registry.update_task(
                            task_id=task_id,
                            dependency_task_ids=dependency_ids
                        )
                        print(f"✅ Updated task {task_id} with dependencies: {dependency_ids}")
            
            # Update the plan with the task IDs
            await self.registry.update_plan(plan_id, task_ids=task_ids)
            
            return plan_id
        except Exception as e:
            print(f"❌ Error processing plan: {str(e)}")
            raise
    
    def _convert_dag_to_plan(self, dag_plan: DAGPlan) -> str:
        """
        Convert a DAG plan to the standard Plan format.
        
        Args:
            dag_plan: DAGPlan model containing nodes with dependencies
            
        Returns:
            JSON string in Plan format
        """
        # Map node IDs to array indices
        node_id_to_index = {node.id: i for i, node in enumerate(dag_plan.nodes)}
        
        # Create tasks with mapped dependencies
        tasks = []
        
        for node in dag_plan.nodes:
            # Convert node dependencies from IDs to indices
            dependency_indices = []
            for dep_id in node.depends_on:
                if dep_id in node_id_to_index:
                    dependency_indices.append(node_id_to_index[dep_id])
                else:
                    print(f"Warning: Dependency {dep_id} not found in node map")
            
            # Create the task item using Pydantic model
            task = TaskPlanItem(
                description=node.description,
                goal_id=node.goal_id,
                dependency_task_ids=dependency_indices,
                robot_type=node.robot_type  # Pass through the robot_type
            )
            tasks.append(task)
        
        # Return as JSON string using Pydantic's json() method
        plan = Plan(tasks=tasks)
        print(f"Converted DAG with {len(dag_plan.nodes)} nodes to Plan with {len(tasks)} tasks")
        return plan.model_dump_json()

# Import concrete planner implementations
from .types import MonolithicPlanner, DAGPlanner
from .types.big_dag_planner import BigDAGPlanner
from robot_fleet.proto.fleet_manager_pb2 import PlanningStrategy

def get_planner(strategy: int, db_url: str = None):
    """
    Get the appropriate planner based on the planning strategy enum value from protobuf.
    
    Args:
        strategy: Integer enum value from fleet_manager_pb2.PlanningStrategy
        db_url: Optional database URL
        
    Returns:
        A planner instance of the appropriate type
    """
    if strategy == PlanningStrategy.MONOLITHIC:
        return MonolithicPlanner(db_url)
    elif strategy == PlanningStrategy.DAG:
        return DAGPlanner(db_url)
    elif strategy == PlanningStrategy.BIG_DAG:
        return BigDAGPlanner(db_url)
    else:
        raise ValueError(f"Unknown planning strategy: {strategy}")
