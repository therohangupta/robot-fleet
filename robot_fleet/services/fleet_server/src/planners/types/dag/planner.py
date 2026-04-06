"""
DAG-based planning strategy implementation.
"""

import json
import logging
from openai import OpenAI
import os
from pathlib import Path
from typing import List, Dict, Any
from ...base import BasePlanner
from ....formats.formats import Plan, DAGPlan, DAGNode, TaskPlanItem

logger = logging.getLogger(__name__)

# Prompts are co-located with this planner
PROMPT_DIR = Path(__file__).parent


class DAGPlanner(BasePlanner):
    """Planner that uses a DAG-based approach to generate plans"""
    
    def __init__(self, registry=None):
        super().__init__(registry=registry)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _load_prompt(self, prompt_type: str) -> str:
        """Load a prompt file from this planner's directory."""
        filepath = PROMPT_DIR / f"{prompt_type}.prompt"
        with open(filepath, 'r') as f:
            return f.read()
    
    async def plan(self, goal_ids: List[int]) -> str:
        """
        Generate separate DAG-based plans for each goal, then combine them into a single plan.
        Each goal gets its own isolated DAG with no cross-goal dependencies.
        
        Args:
            goal_ids: List of goal IDs to generate plans for
            
        Returns:
            JSON string representation of the combined plan in Plan format
        """
        if not goal_ids:
            raise ValueError("DAGPlanner requires at least one goal ID")

        # Initialize planning artifacts
        self.planning_artifacts = {
            "goals": [],
            "dag_structure": {
                "nodes": [],
                "edges": []
            }
        }

        # Get all the goals
        goals = []
        for goal_id in goal_ids:
            goal = await self.registry.get_goal(goal_id)
            if not goal:
                raise ValueError(f"Goal with ID {goal_id} not found")
            goals.append(goal)

        # Update goals in planning artifacts
        self.planning_artifacts["goals"] = [{"goal_id": goal.goal_id, "description": goal.description} for goal in goals]
        
        capabilities = await self._load_capabilities()
        world_statements = await self._load_world_statements()
        robot_context = await self._get_robot_context_string()
        
        # Load prompt templates (co-located with this planner)
        system_prompt_template = self._load_prompt("system")
        user_prompt_template = self._load_prompt("user")
        
        # System prompt is the same for all goals in this planner
        system_prompt = system_prompt_template

        # Store prompts for later retrieval
        self.planning_prompts = {
            "system": system_prompt,
            "user_template": user_prompt_template  # Store template since it varies per goal
        }

        # Generate separate DAGs for each goal
        all_nodes = []
        
        for goal in goals:
            goal_id = goal.goal_id
            
            # Create a unique prefix for this goal's nodes
            goal_prefix = f"goal{goal_id}_"
            
            # Format the user prompt for this specific goal
            user_prompt = user_prompt_template.format(
                goal_id=goal_id,
                goal_description=goal.description,
                robot_context=robot_context,
                world_statements="\n".join(world_statements),
                goal_prefix=goal_prefix
            )
            
            # Call the OpenAI API for this goal
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
            
            # Parse the response as a DAGPlan
            goal_dag_plan = response.choices[0].message.content
            try:
                # Parse the JSON response into a DAGPlan model
                goal_dag_plan = json.loads(goal_dag_plan)
                logger.debug("Generated DAG plan for goal %s:\n%s", goal_id, goal_dag_plan)
                # Convert to Pydantic models for validation
                goal_dag_nodes = [DAGNode(**node) for node in goal_dag_plan['nodes']]
                
                # Add to our cumulative list of nodes
                all_nodes.extend(goal_dag_nodes)
                logger.info("Generated DAG plan for goal %s with %s nodes", goal_id, len(goal_dag_nodes))
            except Exception as e:
                logger.error("Error parsing response for goal %s: %s", goal_id, e)
                raise ValueError(f"Failed to parse LLM response: {e}")
        
        # Create a DAGPlan from all nodes
        combined_dag = DAGPlan(nodes=all_nodes)

        # Store DAG structure in artifacts (include robot_type for transparency)
        self.planning_artifacts["dag_structure"] = {
            "nodes": [{"id": node.id, "description": node.description, "goal_id": node.goal_id, "robot_type": node.robot_type} for node in all_nodes],
            "edges": [{"from": node.id, "to": dep} for node in all_nodes for dep in (node.depends_on or [])]
        }

        # Convert DAG format to Plan format using the parent class method
        return super()._convert_dag_to_plan(combined_dag)
