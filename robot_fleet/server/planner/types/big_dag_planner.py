"""
BigDAG planner that creates a single DAG for all goals in the system.
"""

import json
from typing import List, Dict, Any
import os
from openai import OpenAI
from ..planner import BasePlanner
from ..formats.formats import Plan, DAGPlan, DAGNode, TaskPlanItem

class BigDAGPlanner(BasePlanner):
    """
    Planner that creates a single DAG for all goals in the system.
    This allows for coordinated planning across multiple goals.
    """
    
    def __init__(self, registry):
        super().__init__(registry)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
        
        # Get all goals
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
        # REMINDER: Update 'big_dag_system.prompt' and 'big_dag_user.prompt' if necessary.
        # 'big_dag_user.prompt' should be updated to instruct the LLM
        # to include a 'robot_type' field for each node in the DAGPlan.
        system_prompt_template = self._load_prompt_template("big_dag_system.prompt")
        user_prompt_template = self._load_prompt_template("big_dag_user.prompt")
        
        # Build the goals context
        goals_context = "GOALS TO PLAN FOR:\n"
        for goal in goals:
            goals_context += f"GOAL ID: {goal.goal_id}\n"
            goals_context += f"DESCRIPTION: {goal.description}\n\n"
        
        # Format prompts
        system_prompt = system_prompt_template # System prompt might not need formatting here
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
        except Exception as e:
            print(f"Error parsing BigDAG response: {e}")
            raise ValueError(f"Failed to parse LLM response: {e}")
        
        # Convert DAG format to Plan format using the parent class method
        return super()._convert_dag_to_plan(big_dag_plan)
