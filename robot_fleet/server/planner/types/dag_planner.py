"""
DAG-based planning strategy implementation.
"""

import json
from openai import OpenAI
import os
from typing import List, Dict, Any
from ..planner import BasePlanner
from robot_fleet.server.formats.formats import Plan, DAGPlan, DAGNode, TaskPlanItem

class DAGPlanner(BasePlanner):
    """Planner that uses a DAG-based approach to generate plans"""
    
    def __init__(self, registry):
        super().__init__(registry)
        # Initialize OpenAI client at instance creation time
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
        
        # Get all the goals
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
        # REMINDER: Update 'dag_system.prompt' and 'dag_user.prompt' if necessary.
        # 'dag_user.prompt' should be updated to instruct the LLM
        # to include a 'robot_type' field for each node in the DAGPlan.
        system_prompt_template = self._load_prompt_template("dag_system.prompt")
        user_prompt_template = self._load_prompt_template("dag_user.prompt")
        
        # System prompt is the same for all goals in this planner
        system_prompt = system_prompt_template
        
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
                world_statements="\n".join(world_statements), # Join for formatting
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
                print(f"Generated DAG plan for goal {goal_id}:\n{goal_dag_plan}")
                # Convert to Pydantic models for validation
                goal_dag_nodes = [DAGNode(**node) for node in goal_dag_plan['nodes']]
                
                # Add to our cumulative list of nodes
                all_nodes.extend(goal_dag_nodes)
                print(f"Generated DAG plan for goal {goal_id} with {len(goal_dag_nodes)} nodes")
            except Exception as e:
                print(f"Error parsing response for goal {goal_id}: {e}")
                raise ValueError(f"Failed to parse LLM response: {e}")
        
        # Create a DAGPlan from all nodes
        combined_dag = DAGPlan(nodes=all_nodes)
        
        # Convert DAG format to Plan format using the parent class method
        return super()._convert_dag_to_plan(combined_dag)
