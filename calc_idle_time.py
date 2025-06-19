#!/usr/bin/env python3
import asyncio
import json
import sys
from dotenv import load_dotenv
import os
from typing import Dict, List, Set, Tuple
import argparse
from tabulate import tabulate

# Import from your existing codebase
from robot_fleet.robots.registry.instance_registry import RobotInstanceRegistry

async def calculate_idle_time(plan_id: int, registry=None):
    """
    Calculate idle time for robots executing a plan.
    
    Args:
        plan_id: The ID of the plan to analyze
        registry: Optional registry instance (will create one if not provided)
    """
    # Create or use the provided registry
    if registry is None:
        # Get database URL from environment or use default
        load_dotenv()
        db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://robot_user:secret@localhost:5432/robot_fleet")
        print(f"Using default PostgreSQL database: {db_url}")
        registry = RobotInstanceRegistry(db_url)
    
    # Get the plan
    plan = await registry.get_plan(plan_id)
    if not plan:
        print(f"Error: Plan with ID {plan_id} not found")
        return
    
    print(f"Analyzing Plan #{plan.plan_id}: {len(plan.task_ids)} tasks")
    
    # Get all tasks for this plan
    tasks = {} # This dictionary stores fleet_manager_pb2.Task objects
    print("\nFetching tasks:") # DEBUG print
    for task_id in plan.task_ids: # plan.task_ids is from the Plan proto
        task = await registry.get_task(task_id) # Fetches DB model, converts to Task proto
        if task:
            # DEBUG: Print the fetched task object immediately
            print(f"  Fetched Task {task_id}: {task}")
            tasks[task_id] = task # Store the Task proto
        else:
            print(f"Warning: Task with ID {task_id} not found")

    # Map tasks to robots
    task_to_robot = {}
    robot_tasks = {}
    unassigned_tasks = []
    
    for task_id, task in tasks.items():
        if task.robot_id:
            task_to_robot[task_id] = task.robot_id
            if task.robot_id not in robot_tasks:
                robot_tasks[task.robot_id] = []
            robot_tasks[task.robot_id].append(task_id)
        else:
            unassigned_tasks.append(task_id)
    
    if unassigned_tasks:
        print(f"Warning: {len(unassigned_tasks)} tasks are not assigned to any robot")
        
        # Get available robots
        robots = await registry.list_robots()
        robot_ids = [robot.robot_id for robot in robots]
        
        if not robot_ids:
            print("Error: No robots available for assignment")
            return
        
        # Assign unassigned tasks to robots round-robin
        for i, task_id in enumerate(unassigned_tasks):
            robot_id = robot_ids[i % len(robot_ids)]
            task_to_robot[task_id] = robot_id
            if robot_id not in robot_tasks:
                robot_tasks[robot_id] = []
            robot_tasks[robot_id].append(task_id)
            print(f"Assigned Task {task_id} to Robot {robot_id}")
    
    # Parse dependencies
    dependencies = {}
    for task_id, task in tasks.items(): # Iterate over the Task proto objects stored above
        deps = []
        # Check if the Task proto object has the attribute
        if hasattr(task, 'dependency_task_ids'):
            dep_ids_obj = task.dependency_task_ids
            try:
                # Directly iterate over the Protobuf container
                deps = [int(dep_id) for dep_id in dep_ids_obj]
            except TypeError:
                # Handle case where it might be a single non-iterable value unexpectedly
                # (though unlikely based on previous errors)
                try:
                    deps = [int(dep_ids_obj)] if dep_ids_obj else []
                except (TypeError, ValueError):
                    deps = [] # Fallback if conversion fails
            except Exception as e:
                print(f"    ERROR processing dependencies for task {task_id}: {e}") # Keep error print
                deps = [] # Fallback
        # else: attribute doesn't exist, deps remains []

        dependencies[int(task_id)] = deps # Assign the parsed list

    # Print dependencies for verification
    print("\nTask Dependencies:")
    for task_id, deps in dependencies.items():
        dep_str = ", ".join(str(d) for d in deps) if deps else "None"
        print(f"Task {task_id} depends on: {dep_str}")

    

    
    # Initialize a timeline for each robot
    timeline = {robot_id: [] for robot_id in robot_tasks}
    active_tasks = {robot_id: None for robot_id in robot_tasks}
    
    # Track which tasks have been completed
    completed_tasks = set()
    
    # Simulate execution
    current_time = 0
    
    # Continue until all tasks are completed
    while len(completed_tasks) < len(tasks):
        # Update the completed tasks
        new_completed = False
        for robot_id, task_id in list(active_tasks.items()):
            if task_id is not None:
                completed_tasks.add(int(task_id))  # Convert to integer for proper set handling
                print(f"Robot {robot_id} completed Task {task_id}")
                active_tasks[robot_id] = None
                new_completed = True
        
        # Reset active tasks for next time step
        active_tasks = {r: None for r in robot_tasks.keys()}
        
        # Assign tasks to robots for this time step
        for robot_id in robot_tasks:
            # Find the next unassigned task for this robot
            next_task = None
            for task_id in robot_tasks[robot_id]:
                if int(task_id) not in completed_tasks:
                    # Check if all dependencies are met
                    deps_met = True
                    for dep_id in dependencies.get(int(task_id), []):
                        if int(dep_id) not in completed_tasks:
                            deps_met = False
                            print(f"Robot {robot_id} waiting for dependency Task {dep_id} to complete")
                            break
                    
                    if deps_met:
                        next_task = task_id
                        break
            
            # Assign the task or idle state to this time step
            if next_task:
                active_tasks[robot_id] = next_task
                timeline[robot_id].append((current_time, next_task))
                print(f"Robot {robot_id} working on Task {next_task}")
            else:
                timeline[robot_id].append((current_time, "IDLE"))
                print(f"Robot {robot_id} is IDLE")
        
        # Exit loop if no more tasks can be processed
        if all(task is None for task in active_tasks.values()) and len(completed_tasks) < len(tasks):
            print("Error: No progress being made. Possible circular dependency or unreachable task.")
            break
        
        # Advance time step
        current_time += 1
        
        # Safety check
        if current_time > 100:
            print("Warning: Exceeded maximum time steps (100). Stopping simulation.")
            break
    
    # Print the timeline in a readable format
    print("\nTask Execution Timeline:")
    
    # Create a table header with time steps
    max_time = max([t for robot, times in timeline.items() for t, _ in times]) if any(timeline.values()) else 0

    # Format as a nice table using tabulate
    table_data = []
    headers = ["Robot/Time"] + [str(t) for t in range(max_time + 1)]
    
    for robot_id, times in timeline.items():
        row = [f"Robot {robot_id}"]
        # Create a map of time step to task
        time_to_task = {t: task_id for t, task_id in times}
        
        # Fill in the row with tasks at each time step
        for t in range(max_time + 1):
            if t in time_to_task:
                task_id = time_to_task[t]
                row.append(f"Task {task_id}" if task_id != "IDLE" else "IDLE")
            else:
                row.append("")
        
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Calculate idle time statistics
    idle_stats = {}
    for robot_id, times in timeline.items():
        idle_count = sum(1 for _, task in times if task == "IDLE")
        total_time = max_time + 1
        idle_pct = idle_count / total_time * 100
        idle_stats[robot_id] = idle_pct
        print(f"Robot {robot_id}: {idle_pct:.1f}% idle time")
    
    # Average idle time
    avg_idle = sum(idle_stats.values()) / len(idle_stats) if idle_stats else 0
    print(f"\nAverage idle time across all robots: {avg_idle:.1f}%")
    
    # Clean up resources
    try:
        await registry.close()
    except:
        # If registry doesn't have a close method, we can ignore
        pass

async def main():
    parser = argparse.ArgumentParser(description="Calculate robot idle time for a plan")
    parser.add_argument("plan_id", type=int, help="ID of the plan to analyze")
    args = parser.parse_args()
    
    await calculate_idle_time(args.plan_id)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calc_idle_time.py <plan_id>")
        sys.exit(1)
    
    asyncio.run(main())
