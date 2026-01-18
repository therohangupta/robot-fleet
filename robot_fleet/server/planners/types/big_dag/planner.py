"""
BigDAG planner that creates a single DAG for all goals in the system.
"""

import json
from typing import List, Dict, Any
import os
from pathlib import Path
from openai import OpenAI
from robot_fleet.server.planners.base import BasePlanner
from robot_fleet.server.formats.formats import Plan, DAGPlan, DAGNode, TaskPlanItem

# Prompts are co-located with this planner
PROMPT_DIR = Path(__file__).parent


class BigDAGPlanner(BasePlanner):
    """
    Planner that creates a single DAG for all goals in the system.
    This allows for coordinated planning across multiple goals.
    """
    
    def __init__(self, registry):
        super().__init__(registry)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _load_prompt(self, prompt_type: str) -> str:
        """Load a prompt file from this planner's directory."""
        filepath = PROMPT_DIR / f"{prompt_type}.prompt"
        with open(filepath, 'r') as f:
            return f.read()
    
    async def plan(self, goal_ids: List[int]) -> str:
        """
        Generate a comprehensive DAG-based plan that addresses multiple goals
        with potential interdependencies between goals.
        
        Args:
            goal_ids: List of goal IDs to include in the plan
            
        Returns:
            JSON string representation of the plan in Plan format
        """
        if not goal_ids:
            raise ValueError("BigDAGPlanner requires at least one goal ID")

        # Get all goals first
        goals = []
        for goal_id in goal_ids:
            goal = await self.registry.get_goal(goal_id)
            if not goal:
                raise ValueError(f"Goal with ID {goal_id} not found")
            goals.append(goal)

        # Initialize planning artifacts with goals data
        self.planning_artifacts = {
            "goals": [{"goal_id": goal.goal_id, "description": goal.description} for goal in goals],
            "dag_structure": {
                "nodes": [],
                "edges": []
            }
        }
        
        capabilities = await self._load_capabilities()
        world_statements = await self._load_world_statements()
        robot_context = await self._get_robot_context_string()
        
        # Load prompt templates (co-located with this planner)
        system_prompt_template = self._load_prompt("system")
        user_prompt_template = self._load_prompt("user")
        
        # Build the goals context
        goals_context = "GOALS TO PLAN FOR:\n"
        for goal in goals:
            goals_context += f"GOAL ID: {goal.goal_id}\n"
            goals_context += f"DESCRIPTION: {goal.description}\n\n"
        
        # Format prompts
        system_prompt = system_prompt_template
        user_prompt = user_prompt_template.format(
            goals_context=goals_context,
            robot_context=robot_context,
            world_statements="\n".join(world_statements)
        )

        # Store prompts for later retrieval
        self.planning_prompts = {
            "system": system_prompt,
            "user": user_prompt
        }

        # Call the OpenAI API
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=4000,
            response_format=DAGPlan
        )

        # Parse the response
        big_dag_json = response.choices[0].message.content
        try:
            # Parse the JSON response into a DAGPlan model
            big_dag_dict = json.loads(big_dag_json)
            nodes_data = big_dag_dict.get("nodes", [])
            
            # Convert to Pydantic model for validation
            dag_nodes = [DAGNode(**node) for node in nodes_data]
            big_dag_plan = DAGPlan(nodes=dag_nodes)
            
            print(f"Generated BigDAG plan with {len(dag_nodes)} nodes for {len(goals)} goals")

            # Store DAG structure in artifacts (include robot_type for transparency)
            self.planning_artifacts["dag_structure"] = {
                "nodes": [{"id": node.id, "description": node.description, "goal_id": node.goal_id, "robot_type": node.robot_type} for node in dag_nodes],
                "edges": [{"from": node.id, "to": dep} for node in dag_nodes for dep in (node.depends_on or [])]
            }
        except Exception as e:
            print(f"Error parsing BigDAG response: {e}")
            raise ValueError(f"Failed to parse LLM response: {e}")

        # Convert DAG format to Plan format using the parent class method
        return super()._convert_dag_to_plan(big_dag_plan)
