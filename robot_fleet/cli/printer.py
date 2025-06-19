"""Printer functions for CLI output"""
import click
import sys
from typing import List
from ..proto import fleet_manager_pb2

PLANNING_STRATEGY_ENUMS = {
    "monolithic": fleet_manager_pb2.PlanningStrategy.MONOLITHIC,
    "dag": fleet_manager_pb2.PlanningStrategy.DAG,
    "big_dag": fleet_manager_pb2.PlanningStrategy.BIG_DAG
}
PLANNING_STRATEGY_CHOICES = list(PLANNING_STRATEGY_ENUMS.keys())
PLANNING_STRATEGY_STRINGS = {v: k for k, v in PLANNING_STRATEGY_ENUMS.items()}

ALLOCATION_STRATEGY_ENUMS = {
    "lp": fleet_manager_pb2.AllocationStrategy.LP,
    "llm": fleet_manager_pb2.AllocationStrategy.LLM,
    "cost_based": fleet_manager_pb2.AllocationStrategy.COST_BASED
}
ALLOCATION_STRATEGY_CHOICES = list(ALLOCATION_STRATEGY_ENUMS.keys())
ALLOCATION_STRATEGY_STRINGS = {v: k for k, v in ALLOCATION_STRATEGY_ENUMS.items()}

def print_response(response, success_prefix: str = "Success", error_prefix: str = "Failed"):
    """Print a standard response message and exit with appropriate code if failed
    
    Args:
        response: Response object with success and message fields
        success_prefix: Prefix for success message (default: "Success")
        error_prefix: Prefix for error message (default: "Failed")
    """
    if response.success:
        click.echo(f"{success_prefix}: {response.message}")
    else:
        click.echo(f"{error_prefix}: {response.message}", err=True)
        sys.exit(1)

def print_task(task, verbose=False, include_newline: bool = True, tabs: int = 0):
    """Print details of a single task. In verbose, print all proto fields. Indent output by 'tabs' spaces."""
    indent = ' ' * tabs
    if include_newline:
        click.echo("")
    if not verbose:
        click.echo(f"{indent}Task {task.task_id}")
        return
    # Print all proto fields
    click.echo(f"{indent}Task {task.task_id}:")
    click.echo(f"{indent}  description: {task.description}")
    if task.goal_id != 0:
        click.echo(f"{indent}  goal_id: {task.goal_id}")
    if task.plan_id != 0:
        click.echo(f"{indent}  plan_id: {task.plan_id}")
    if task.dependency_task_ids:
        click.echo(f"{indent}  dependency_task_ids: {task.dependency_task_ids}")
    if task.robot_id:
        click.echo(f"{indent}  robot_id: {task.robot_id}")

def print_goal(goal, plans=None, tasks=None, verbose=False, include_newline: bool = True):
    """Print details of a single goal, with associated plans and their tasks in a tree format."""
    if include_newline:
        click.echo("")
    click.echo(f"Goal {goal.goal_id}:")
    click.echo(f"  Goal ID: {goal.goal_id}")
    click.echo(f"  Description: {goal.description}")

    if plans and verbose:
        click.echo("  Plans:")
        for plan in plans:
            click.echo(f"    Plan: {plan.plan_id}")
            strategy_value = getattr(plan, 'planning_strategy', None)
            strategy_str = fleet_manager_pb2.PlanningStrategy.Name(strategy_value)
            click.echo(f"      Strategy: {strategy_str}")
            if tasks:
                plan_tasks = [task for task in tasks if task.plan_id == plan.plan_id]
                if plan_tasks:
                    click.echo("      Tasks:")
                    for task in plan_tasks:
                        print_task(task, verbose=verbose, include_newline=False, tabs=8)

# def print_plan(plan, goals=None, tasks=None, verbose=False, include_newline: bool = True):
#     """Print details of a single plan, optionally with associated goals and tasks (tree style)"""
#     if include_newline:
#         click.echo("")
#     click.echo(f"Plan: {plan.plan_id}")
#     strategy_value = getattr(plan, 'planning_strategy', None)
#     strategy_str = PLANNING_STRATEGY_STRINGS.get(strategy_value, str(strategy_value))
#     click.echo(f"  Strategy: {strategy_str}")
#     if verbose and hasattr(plan, 'goal_ids') and plan.goal_ids:
#         click.echo(f"  Goals:")
#         for gid in plan.goal_ids:
#             if goals:
#                 goal = next((g for g in goals if g.goal_id == gid), None)
#                 if goal:
#                     click.echo(f"    Goal ID: {goal.goal_id}")
#                     click.echo(f"    Description: {goal.description}")
#                     if verbose and tasks:
#                         plan_tasks = [task for task in tasks if task.goal_id == goal.goal_id and task.plan_id == plan.plan_id]
#                         if plan_tasks:
#                             click.echo("        Tasks:")
#                             for task in plan_tasks:
#                                 print_task(task, verbose=verbose, include_newline=False, tabs=10)

import click
from collections import defaultdict, deque
from ..proto import fleet_manager_pb2

def print_plan(plan, goals=None, tasks=None, verbose=False, include_newline: bool = True):
    """Print details of a single plan as an ASCII DAG per goal."""
    if include_newline:
        click.echo("")
    click.echo(f"Plan: {plan.plan_id}")
    strategy_value = getattr(plan, 'planning_strategy', None)
    strategy_str = fleet_manager_pb2.PlanningStrategy.Name(strategy_value)
    allocation_value = getattr(plan, 'allocation_strategy', None)
    allocation_str = fleet_manager_pb2.AllocationStrategy.Name(allocation_value) if allocation_value is not None else "UNSPECIFIED"
    click.echo(f"  Strategy: {strategy_str}")
    click.echo(f"  Allocation: {allocation_str}")

    if not verbose or not hasattr(plan, 'goal_ids') or not plan.goal_ids:
        return

    click.echo("  Tasks (DAG View):")
    # Collect all tasks for this plan (regardless of goal)
    plan_tasks = [t for t in (tasks or []) if t.plan_id == plan.plan_id]
    if not plan_tasks:
        return

    # Build dependency graph for all tasks
    deps = {t.task_id: set(t.dependency_task_ids) for t in plan_tasks}
    graph = defaultdict(set)
    for tid, prereqs in deps.items():
        for d in prereqs:
            graph[d].add(tid)
    # initialize in-degrees
    indegree = {t.task_id: 0 for t in plan_tasks}
    for src, targets in graph.items():
        for tgt in targets:
            indegree[tgt] += 1

    # Kahn's algorithm for topological sort
    queue = deque([tid for tid, deg in indegree.items() if deg == 0])
    topo = []
    while queue:
        nid = queue.popleft()
        topo.append(nid)
        for m in graph[nid]:
            indegree[m] -= 1
            if indegree[m] == 0:
                queue.append(m)

    for idx, tid in enumerate(topo):
        task = next(t for t in plan_tasks if t.task_id == tid)
        bullet = "└─" if idx == len(topo) - 1 else "├─"
        dep_list = deps[tid]
        dep_str = ""
        if dep_list:
            dep_str = " (depends on " + ", ".join(f"T{d}" for d in sorted(dep_list)) + ")"
        click.echo(f"    {bullet} (G{task.goal_id}, T{task.task_id}, {task.robot_id}): {task.description}{dep_str}")

def print_robot(robot, verbose=False, tasks=None, include_newline: bool = True):
    """Print details of a single robot, with optional verbose task info."""
    if include_newline:
        click.echo("")
    click.echo(f"Robot: {robot.robot_id}")
    click.echo(f"Type: {robot.robot_type}")
    click.echo(f"Description: {robot.description}")
    click.echo(f"Host: {robot.task_server_info.host}")
    click.echo(f"Port: {robot.task_server_info.port}")
    
    status = getattr(robot, 'status', None)
    if status:
        state = getattr(status, 'state', None)
        if state is not None:
            from ..proto import fleet_manager_pb2
            state_name = fleet_manager_pb2.RobotStatus.State.Name(state)
            click.echo(f"Status: {state_name}")
        if getattr(status, 'message', None):
            click.echo(f"Status Message: {status.message}")
    if getattr(robot, 'capabilities', []):
        click.echo(f"Capabilities:")
        for cap in robot.capabilities:
            click.echo(f"  - {cap}")

    if verbose:
        if getattr(robot, 'task_ids', []) and tasks:
            click.echo(f"Tasks:")
            for task in tasks:
                print_task(task, verbose=verbose, include_newline=False, tabs=4)

def print_all_robots(robots, tasks_by_robot=None, verbose=False):
    """Print a list of robots, optionally with their tasks (tree style)"""
    if not robots:
        click.echo("No robots found")
        return
    for robot in robots:
        assigned_tasks = tasks_by_robot[robot.robot_id] if (verbose and tasks_by_robot and robot.robot_id in tasks_by_robot) else None
        print_robot(robot, verbose=verbose, tasks=assigned_tasks)

def print_all_tasks(tasks, verbose=False):
    """Print a list of tasks, optionally with details"""
    if not tasks:
        click.echo("No tasks found")
        return
    for task in tasks:
        print_task(task, verbose=verbose, include_newline=False)

def print_all_goals(goals: List[fleet_manager_pb2.Goal], tasks_by_goal=None, verbose=False):
    """Print a list of goals"""
    if not goals:
        click.echo("No goals found")
        return
    for goal in goals:
        print_goal(goal, tasks=tasks_by_goal, verbose=verbose, include_newline=False)

def print_all_plans(plans, goals=None, tasks_by_plan=None, verbose=False):
    """Print a list of plans, optionally with associated goals and tasks (tree style)"""
    if not plans:
        click.echo("No plans found")
        return
    for plan in plans:
        print_plan(plan, goals, tasks_by_plan, verbose=verbose, include_newline=False)

def print_world_statement(ws: fleet_manager_pb2.WorldStatement):
    """Helper to print a single world statement."""
    if not ws:
        click.echo("World statement not found", err=True)
        return
    click.echo(f"ID: {ws.id}")
    click.echo(f"Statement: {ws.statement}")
    click.echo(f"Created At: {ws.created_at.ToDatetime().strftime('%Y-%m-%d %H:%M:%S') if ws.HasField('created_at') else 'N/A'}")

def print_world_statement_list(ws_list: List[fleet_manager_pb2.WorldStatement]):
    """Helper to print a list of world statements."""
    if not ws_list:
        click.echo("No world statements found", err=True)
        return
    click.echo("ID\tStatement")
    for ws in ws_list:
        click.echo(f"{ws.id}\t{ws.statement}")