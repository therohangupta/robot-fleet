"""
Monolithic planning strategy implementation.
"""

import json
from openai import OpenAI
import os
from ..planner import BasePlanner
from robot_fleet.server.formats.formats import Plan
from typing import List

class MonolithicPlanner(BasePlanner):
    """Planner that uses a monolithic approach to generate plans"""
    
    def __init__(self, registry):
        super().__init__(registry)
        # Initialize OpenAI client at instance creation time
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
        
        # Load prompt templates
        # REMINDER: Update 'monolithic_system.prompt' and 'monolithic_user.prompt' if necessary.
        # 'monolithic_user.prompt' should be updated to instruct the LLM
        # to include a 'robot_type' field for each task in the Plan.
        system_prompt_template = self._load_prompt_template("monolithic_system.prompt")
        user_prompt_template = self._load_prompt_template("monolithic_user.prompt")
        
        # Build the goals context
        goals_context = "GOALS TO PLAN FOR:\n"
        for goal in goals:
            goals_context += f"GOAL ID: {goal.goal_id}\n"
            goals_context += f"DESCRIPTION: {goal.description}\n\n"
        
        # Format prompts
        system_prompt = system_prompt_template # System prompt might not need formatting
        user_prompt = user_prompt_template.format(
            goals_context=goals_context,
            robot_context=robot_context,
            world_statements="\n".join(world_statements) # Join statements for formatting
        )

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
        print(f"Generated monolithic plan for goals {goal_ids}:\n{plan_json}")
        
        return plan_json
