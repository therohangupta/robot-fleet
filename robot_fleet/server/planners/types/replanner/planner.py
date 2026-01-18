"""
Replanning strategy implementation.
"""
import json
from openai import OpenAI
import os
from pathlib import Path
from robot_fleet.server.planners.base import BasePlanner
from robot_fleet.server.formats.formats import Plan
from typing import List, Dict, Any, Optional
import logging
import asyncio
from robot_fleet.proto import fleet_manager_pb2
from robot_fleet.robots.registry.instance_registry import RobotInstanceRegistry

logger = logging.getLogger(__name__)

# Prompts are co-located with this planner
PROMPT_DIR = Path(__file__).parent


class Replanner(BasePlanner):
    """Planner that generates a recovery plan after a task failure."""

    def __init__(self, db_url: Optional[str] = None):
        super().__init__(db_url)
        self.db_url = db_url
        self.registry = RobotInstanceRegistry(db_url)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _load_prompt(self, prompt_type: str) -> str:
        """Load a prompt file from this planner's directory."""
        filepath = PROMPT_DIR / f"{prompt_type}.prompt"
        with open(filepath, 'r') as f:
            return f.read()

    async def plan(self, goal_ids: List[int]) -> str:
        """Replanner does not implement standard planning."""
        raise NotImplementedError("Replanner only supports the 'replan' method.")

    async def replan(self, plan_id: int, failed_task_id: int, failure_message: str, robot_task_assignments: Dict[str, List]) -> int:
        """
        Generate a new plan segment to recover from a failure and save it.
        Returns the ID of the newly created plan.
        """
        print(f"Starting replan for Plan ID: {plan_id}, Failed Task ID: {failed_task_id}")

        try:
            # 1. Fetch Original Plan Data
            original_plan_obj: Optional[fleet_manager_pb2.Plan] = await self.registry.get_plan(plan_id)
            if not original_plan_obj:
                raise ValueError(f"Original plan with ID {plan_id} not found.")
            
            original_goal_ids = list(original_plan_obj.goal_ids)
            original_planning_strategy = original_plan_obj.planning_strategy
            original_allocation_strategy = original_plan_obj.allocation_strategy

            ordered_task_ids_for_context = original_plan_obj.task_ids
            if not ordered_task_ids_for_context:
                logger.warning(f"Plan {plan_id} from registry does not directly list ordered task_ids. Context generation might be limited.")

            original_tasks_in_order = []
            for task_id_for_context in ordered_task_ids_for_context:
                task = await self.registry.get_task(task_id_for_context)
                if task:
                    original_tasks_in_order.append(task)
                else:
                    logger.warning(f"Task ID {task_id_for_context} listed in plan {plan_id} not found in registry for context.")

            if not original_tasks_in_order and ordered_task_ids_for_context:
                logger.warning(f"Could not retrieve detailed task objects for plan {plan_id} context, though task IDs were listed.")

            goal_ids_for_context = original_plan_obj.goal_ids
            goals = []
            goals_context_lines = []
            for gid in goal_ids_for_context:
                goal = await self.registry.get_goal(gid)
                if goal:
                    goals.append(goal)
                    goals_context_lines.append(f"- GOAL ID: {goal.goal_id}, DESCRIPTION: {goal.description}")
                else:
                    logger.warning(f"Goal ID {gid} for plan {plan_id} not found.")
            goals_context = "\n".join(goals_context_lines) if goals_context_lines else "No goal details found."

            # 2. Determine Task States based on ordered list and failed_task_id
            failed_task_index = -1
            for i, task in enumerate(original_tasks_in_order):
                if task.task_id == failed_task_id:
                    failed_task_index = i
                    break
            
            if failed_task_index == -1 and ordered_task_ids_for_context:
                if failed_task_id not in ordered_task_ids_for_context:
                    raise ValueError(f"Failed task ID {failed_task_id} not part of plan {plan_id}'s task list: {ordered_task_ids_for_context}")
                logger.error(f"Failed task ID {failed_task_id} was in plan's task list but corresponding task object not retrieved for context.")
                raise ValueError(f"Failed task ID {failed_task_id} could not be processed from plan {plan_id}.")
            elif not ordered_task_ids_for_context and failed_task_id:
                raise ValueError(f"Plan {plan_id} has no tasks, so cannot find failed_task_id {failed_task_id}")

            completed_tasks_objects = original_tasks_in_order[:failed_task_index]
            failed_task_object = original_tasks_in_order[failed_task_index]
            pending_tasks_objects_original = original_tasks_in_order[failed_task_index+1:]

            completed_tasks_summary = "\n".join([f"- Task ID {t.task_id}: '{t.description}' (Goal {t.goal_id})" for t in completed_tasks_objects]) or "None"
            pending_tasks_summary = "\n".join([f"- Task ID {t.task_id}: '{t.description}' (Goal {t.goal_id})" for t in pending_tasks_objects_original]) or "None"
            
            # 3. Format Robot Task Status Context
            robot_task_status_lines = []
            for robot_id, task_ids in robot_task_assignments.items():
                try:
                    if len(task_ids) == 0:
                        robot_details = await self.registry.get_robot(robot_id)
                        robot_name = robot_details.robot_id
                        robot_task_status_lines.append(f"- {robot_name} (ID: {robot_id}) had no tasks assigned.")
                    else:
                        robot_details = await self.registry.get_robot(robot_id)
                        current_task_id = task_ids[0]
                        task_details = await self.registry.get_task(current_task_id)
                        status = "FAILED OR NEW INFORMATION FOUND" if current_task_id == failed_task_id else "ACTIVE (at time of failure)"
                        robot_name = robot_details.robot_id if robot_details else f"Robot {robot_id}"
                        task_desc = task_details.description if task_details else f"Task {current_task_id}"
                        robot_task_status_lines.append(f"- {robot_name} (ID: {robot_id}) was assigned Task ID {current_task_id}: '{task_desc}'. Status: {status}")
                except Exception as e:
                    current_task_id = task_ids[0]
                    logger.warning(f"Could not get details for robot {robot_id} or task {current_task_id}: {e}")
                    robot_task_status_lines.append(f"- Robot ID {robot_id} was assigned Task ID {current_task_id} (Details unavailable). Status: UNKNOWN")
            robot_task_status_context = "\n".join(robot_task_status_lines) if robot_task_status_lines else "No robot assignments provided or details unavailable."

            # 4. Load Capabilities and World State
            robot_capabilities_context = await self._get_robot_context_string()
            world_statements_list = await self._load_world_statements()
            world_statements = "\n".join([f"- {s}" for s in world_statements_list]) if world_statements_list else "No world statements available."

            # 5. Load and Format Prompts (co-located with this planner)
            system_prompt = self._load_prompt("system")
            user_prompt_template = self._load_prompt("user")

            user_prompt = user_prompt_template.format(
                plan_id=plan_id,
                failed_task_id=failed_task_id,
                goals_context=goals_context,
                failure_message=failure_message,
                robot_task_status_context=robot_task_status_context,
                completed_tasks_summary=completed_tasks_summary,
                pending_tasks_summary=pending_tasks_summary,
                robot_capabilities_context=robot_capabilities_context,
                world_statements=world_statements
            )
            
            logger.debug(f"Replanning System Prompt:\n{system_prompt}")
            logger.debug(f"Replanning User Prompt:\n{user_prompt}")

            # 6. Call LLM
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

            llm_response_content = response.choices[0].message.content
            if not llm_response_content or not isinstance(llm_response_content, str):
                logger.error(f"LLM response content is not a valid string: {llm_response_content}")
                raise ValueError("LLM generated empty or invalid content for the replan.")

            replan_json_str = llm_response_content
            logger.info(f"Generated replan JSON for Plan ID {plan_id}:\n{replan_json_str}")

            # 7. Validate and Save Replanning LLM response
            try:
                validated_plan_data = Plan.model_validate_json(replan_json_str)
                logger.info(f"Replanning LLM response successfully validated against Plan schema.")
                
                new_plan_id = await self.save_plan_to_db(
                    plan_json=replan_json_str,
                    planning_strategy=original_planning_strategy,
                    allocation_strategy=original_allocation_strategy,
                    goal_ids=original_goal_ids 
                )
                logger.info(f"Successfully saved new replan with ID: {new_plan_id} to the database.")

                # Call the allocator to assign robots to the new tasks
                from ....allocators.types.llm import LLMAllocator
                allocator = LLMAllocator(db_url=self.db_url if self.db_url else "postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet")
                await allocator.registry.initialize()
                allocation_result = await allocator.allocate(new_plan_id)
                logger.info(f"Allocation result for new plan: {allocation_result}")

                return new_plan_id

            except json.JSONDecodeError as e:
                logger.error(f"Replanning LLM response is not valid JSON: {e}\nContent: {replan_json_str}")
                raise ValueError(f"Replanning failed: LLM response was not valid JSON. Error: {e}")
            except Exception as e:
                logger.error(f"Failed to validate or save replan: {e}\nContent: {replan_json_str}")
                raise ValueError(f"Replanning failed: Could not validate or save the plan. Error: {e}")

        except Exception as e:
            logger.error(f"Error during replanning for plan {plan_id}: {e}")
            raise


async def main():
    # Hardcoded values for testing
    db_url = "postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet"
    replanner = Replanner(db_url=db_url)

    sample_plan_id = 1
    sample_failed_task_id = 3
    sample_failure_message = "Navigation sensor malfunctioned on approach to trash bin."
    sample_robot_task_assignments = { 
        "nav-1": sample_failed_task_id, 
        "pick-place-1": 4 
    }

    logger.info(f"Attempting to replan with hardcoded values: plan_id={sample_plan_id}, failed_task_id={sample_failed_task_id}")
    
    try:
        new_plan_id = await replanner.replan(
            plan_id=sample_plan_id,
            failed_task_id=sample_failed_task_id,
            failure_message=sample_failure_message,
            robot_task_assignments=sample_robot_task_assignments
        )
        print(f"Replanning successful. New plan segment created with ID: {new_plan_id}")
    except ValueError as ve:
        print(f"Replanning validation error: {ve}")
    except NotImplementedError as nie:
        print(f"Replanning not implemented error: {nie}")
    except Exception as e:
        print(f"An unexpected error occurred during replanning: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(replanner.registry, 'close'):
            await replanner.registry.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set.")
        exit(1)
    asyncio.run(main())
