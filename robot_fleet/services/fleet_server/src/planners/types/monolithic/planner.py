"""
Monolithic planning strategy implementation.
"""

import json
import logging
from openai import OpenAI
import os
from pathlib import Path
from ...base import BasePlanner
from ....formats.formats import Plan
from typing import List

logger = logging.getLogger(__name__)

# Prompts are co-located with this planner
PROMPT_DIR = Path(__file__).parent


class MonolithicPlanner(BasePlanner):
    """Planner that uses a monolithic approach to generate plans"""
    
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
        Generate a sequential plan where tasks for all goals are planned in sequence.
        Tasks can depend on previous tasks, including from different goals.
        
        Args:
            goal_ids: List of goal IDs to plan for
            
        Returns:
            JSON string representation of the plan
        """
        if not goal_ids:
            raise ValueError("MonolithicPlanner requires at least one goal ID")
        
        # Fetch all goals
        goals = []
        for goal_id in goal_ids:
            goal = await self.registry.get_goal(goal_id)
            if not goal:
                raise ValueError(f"Goal with ID {goal_id} not found")
            goals.append(goal)
        
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

        # Store planning artifacts
        self.planning_artifacts = {
            "goals": [goal.description for goal in goals],
            "robot_context": robot_context,
            "world_statements": world_statements,
            "capabilities": capabilities,
            "dag_structure": {
                "nodes": [{"id": f"task_{i}", "description": task.get("description", "")} for i, task in enumerate(plan_json.get("tasks", []))],
                "edges": []
            }
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
            response_format=Plan
        )

        # Extract the response content
        plan_json = response.choices[0].message.content
        logger.info("Generated monolithic plan for goals %s:\n%s", goal_ids, plan_json)

        # Store the generated plan as an artifact
        self.planning_artifacts["generated_plan"] = plan_json

        return plan_json
