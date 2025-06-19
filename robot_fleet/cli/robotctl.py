#!/usr/bin/env python3
import click
import grpc
import sys
import os
import re
from typing import Optional, List
from ..proto import fleet_manager_pb2
from ..robots.schema.yaml_validator import YAMLValidator
from .printer import (
    print_task, print_all_tasks, 
    print_goal, print_all_goals, 
    print_robot, print_all_robots,
    print_response, print_plan, print_all_plans,
    print_world_statement,
    print_world_statement_list,
    PLANNING_STRATEGY_ENUMS,
    PLANNING_STRATEGY_CHOICES,
    PLANNING_STRATEGY_STRINGS,
    ALLOCATION_STRATEGY_ENUMS,
    ALLOCATION_STRATEGY_CHOICES,
    ALLOCATION_STRATEGY_STRINGS,
)
from .client import FleetManagerClient
import yaml
import asyncio

def load_robot_config(config_file: str) -> dict:
    """Load and validate robot configuration from YAML file"""
    try:
        validator = YAMLValidator()
        return validator.validate_file(config_file)
    except Exception as e:
        click.echo(f"Error loading config file: {str(e)}", err=True)
        sys.exit(1)

@click.group()
def cli():
    """Robot fleet management CLI"""
    pass

@cli.command()
@click.argument('config_file')
@click.argument('robot_id', required=False)
@click.option('--num', '-n', default=1, show_default=True, type=int, help="Number of robots to register")
def register(config_file, robot_id, num):
    """Register a new robot. Optionally register multiple robots with --num."""
    try:
        # Load robot config
        config = load_robot_config(config_file)
        client = FleetManagerClient()
        robot_type = config['metadata']['name']
        description = config['metadata'].get('description', '')
        capabilities = config.get('capabilities', [])
        task_server_host = config['taskServer']['host']
        task_server_port = config['taskServer']['port']
        docker_host = config['deployment']['docker_host']
        docker_port = config['deployment']['docker_port']
        container_image = config['container']['image']
        container_env = config['container'].get('environment', {})

        if num == 1:
            rid = robot_id if robot_id else f"{robot_type}-1"
            response = client.register_robot(
                robot_id=rid,
                robot_type=robot_type,
                description=description,
                capabilities=capabilities,
                task_server_host=task_server_host,
                task_server_port=task_server_port,
                docker_host=docker_host,
                docker_port=docker_port,
                container_image=container_image,
                container_env=container_env
            )
            print_response(response)
        else:
            # Fetch all existing robots and their IDs
            existing_robots = client.list_robots().robots
            existing_ids = [r.robot_id for r in existing_robots if r.robot_id.startswith(f"{robot_type}-")]
            # Extract the numeric suffix for this robot_type
            suffixes = [int(re.match(rf"{robot_type}-(\\d+)", rid).group(1)) for rid in existing_ids if re.match(rf"{robot_type}-(\\d+)", rid)]
            start_idx = max(suffixes) + 1 if suffixes else 1
            # Always skip IDs that already exist
            num_registered = 0
            i = start_idx
            while num_registered < num:
                rid = f"{robot_type}-{i}"
                if rid in existing_ids:
                    i += 1
                    continue
                response = client.register_robot(
                    robot_id=rid,
                    robot_type=robot_type,
                    description=description,
                    capabilities=capabilities,
                    task_server_host=task_server_host,
                    task_server_port=task_server_port,
                    docker_host=docker_host,
                    docker_port=docker_port,
                    container_image=container_image,
                    container_env=container_env
                )
                print_response(response)
                num_registered += 1
                i += 1
            
    except FileNotFoundError:
        click.echo(f"Config file {config_file} not found", err=True)
        sys.exit(1)
    except yaml.YAMLError as e:
        click.echo(f"Error parsing config file: {str(e)}", err=True)
        sys.exit(1)
    except grpc.RpcError as e:
        click.echo(f"gRPC error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@cli.command()
@click.argument('robot_id')
def deploy(robot_id):
    """Deploy a registered robot"""
    try:
        client = FleetManagerClient()
        response = client.deploy_robot(robot_id)
        print_response(response)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@cli.command()
@click.argument('robot_id')
def undeploy(robot_id):
    """Undeploy a robot"""
    try:
        client = FleetManagerClient()
        response = client.undeploy_robot(robot_id)
        print_response(response)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@cli.command()
@click.argument('robot_id')
def unregister(robot_id):
    """Unregister a robot"""
    try:
        client = FleetManagerClient()
        response = client.unregister_robot(robot_id)
        print_response(response)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@cli.command()
@click.option('--filter', 'filter_type', type=click.Choice(['all', 'deployed', 'registered'], case_sensitive=False),
              default='all', help="Filter robot list")
@click.option('-v', '--verbose', is_flag=True, help="Show detailed robot and task info")
def list(filter_type, verbose):
    """List all robots"""
    try:
        client = FleetManagerClient()
        response = client.list_robots(filter_type)
        robots = response.robots
        tasks_by_robot = {}
        if verbose:
            for robot in robots:
                tasks = []
                for tid in getattr(robot, 'task_ids', []):
                    try:
                        t = client.get_task(tid).task
                        tasks.append(t)
                    except Exception:
                        click.echo(f"[Warning] Task ID {tid} not found for robot {robot.robot_id}", err=True)
                tasks_by_robot[robot.robot_id] = tasks
        print_all_robots(robots, tasks_by_robot if verbose else None, verbose)
    except Exception as e:
        click.echo(f"Error listing robots: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@cli.command()
@click.argument('robot_id')
@click.option('-v', '--verbose', is_flag=True, help="Show detailed robot and task info")
def status(robot_id, verbose):
    """Get detailed status of a robot"""
    try:
        client = FleetManagerClient()
        status = client.get_robot_status(robot_id)
        state_name = fleet_manager_pb2.RobotStatus.State.Name(status.state)
        click.echo(f"\nRobot Status: {state_name}")
        if status.message:
            click.echo(f"Status Message: {status.message}")
        if verbose and getattr(status, 'task_ids', []):
            tasks = []
            for tid in status.task_ids:
                try:
                    t = client.get_task(tid).task
                    tasks.append(t)
                except Exception:
                    click.echo(f"[Warning] Task ID {tid} not found for robot {robot_id}", err=True)
            print_all_tasks(tasks, verbose=True)
    except Exception as e:
        click.echo(f"Error getting robot status: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@cli.group()
def goal():
    """Goal management commands"""
    pass

@goal.command()
@click.argument('description')
@click.option('--planning-strategy', '-p', type=click.Choice(['monolithic', 'dag', 'big_dag'], case_sensitive=False), 
              default='monolithic', help='Planning strategy to use (monolithic, dag, or big_dag)')
def add(description, planning_strategy):
    """Add a new goal with specified planning strategy"""
    try:
        with FleetManagerClient() as client:
            # Map the strategy string to the string expected by the client
            response = client.create_goal(description)
            if response.goal:
                print_goal(response.goal)
            else:
                click.echo(f"Error: {response.error}", err=True)
    except grpc.RpcError as e:
        status_code = e.code()
        status_details = e.details()
        click.echo(f"gRPC error: {status_code.name} - {status_details}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error adding goal: {str(e)}", err=True)
        sys.exit(1)

@goal.command()
@click.argument('goal_id', type=int)
@click.option('-v', '--verbose', is_flag=True, help="Show detailed goal, plan, and task info")
def get(goal_id, verbose):
    """Get details of a specific goal"""
    try:
        client = FleetManagerClient()
        goal_resp = client.get_goal(goal_id)
        goal = goal_resp.goal
        if not goal.goal_id:
            click.echo(f"[Warning] Goal ID {goal_id} not found", err=True)
            return
        # Load associated plans and tasks for this goal
        plans = client.list_plans().plans
        tasks = client.list_tasks(goal_ids=[goal.goal_id]).tasks
        print_goal(goal, plans=plans, tasks=tasks, verbose=verbose)
    except Exception as e:
        click.echo(f"Error getting goal: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@goal.command()
@click.option('-v', '--verbose', is_flag=True, help="Show detailed info for all goals")
def list(verbose):
    """List all goals"""
    try:
        client = FleetManagerClient()
        goals = client.list_goals().goals
        plans = client.list_plans().plans
        tasks = client.list_tasks().tasks
        for goal in goals:
            goal_plans = [p for p in plans if goal.goal_id in getattr(p, 'goal_ids', [])]
            goal_tasks = [t for t in tasks if t.goal_id == goal.goal_id]
            print_goal(goal, plans=goal_plans, tasks=goal_tasks, verbose=verbose)
    except Exception as e:
        click.echo(f"Error listing goals: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@goal.command()
@click.argument('goal_id', type=int)
def delete(goal_id):
    """Delete a goal"""
    try:
        # Validate goal_id is an integer
        try:
            goal_id_int = int(goal_id)
        except ValueError:
            click.echo(f"Error: Goal ID must be an integer, got '{goal_id}'", err=True)
            sys.exit(1)
            
        client = FleetManagerClient()
        response = client.delete_goal(goal_id_int)
        if response.goal:
            click.echo(f"Goal {goal_id_int} deleted successfully")
        else:
            click.echo(f"Error: {response.error}", err=True)
    except grpc.RpcError as e:
        status_code = e.code()
        status_details = e.details()
        click.echo(f"gRPC error: {status_code.name} - {status_details}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@cli.group()
def task():
    """Task management commands"""
    pass

@task.command()
@click.argument('description', type=str)
@click.option('--goal-id', type=int, required=True, help="Goal ID for the task")
@click.option('--plan-id', type=int, required=True, help="Plan ID for the task")
@click.option('--robot-id', type=str, required=True, default=None, help="Robot ID to assign (optional)")
@click.option('--robot-type', type=str, required=True, default=None, help="Robot type to assign (optional)")
@click.option('--dependencies', required=False, type=str, default=None, help="Comma-separated list of task IDs this task depends on (optional)")
def add(description, goal_id, plan_id, robot_id, robot_type, dependencies):
    """Add a new task to a goal and plan, with optional robot assignment and dependencies."""
    try:
        client = FleetManagerClient()
        # Parse dependencies string into a list of ints, if provided
        if dependencies:
            dependencies_list = [int(x.strip()) for x in dependencies.split(',') if x.strip()]
        else:
            dependencies_list = []
        resp = client.create_task(description, robot_id, robot_type, goal_id, plan_id, dependencies_list)
        if resp.task:
            click.echo(f"Task added: {resp.task.task_id}")
        else:
            click.echo(f"Error adding task: {resp.error}", err=True)
    except Exception as e:
        click.echo(f"Error adding task: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

# Example usage:
# r task add "Fry eggs" --goal-id 1 --plan-id 1
# r task add "Pour coffee" --goal-id 1 --plan-id 2 --robot-id rob
# r task add "Put plates" --goal-id 2 --plan-id 1 --dependencies 5 --dependencies 6

@task.command()
@click.option('--goal-id', help='Filter tasks by goal ID')
@click.option('--robot-id', help='Filter tasks by robot ID')
@click.option('-v', '--verbose', is_flag=True, help="Show detailed task info")
def list(goal_id, robot_id, verbose):
    """List tasks with optional filtering"""
    try:
        # Convert goal_id to int if provided
        goal_ids = None
        if goal_id:
            try:
                goal_ids = [int(goal_id)]
            except ValueError:
                click.echo(f"Error: Goal ID must be an integer, got '{goal_id}'", err=True)
                sys.exit(1)
                
        # Convert robot_id to list if provided
        robot_ids = None
        if robot_id:
            robot_ids = [robot_id]
                
        client = FleetManagerClient()
        response = client.list_tasks(goal_ids=goal_ids, robot_ids=robot_ids)
        tasks = response.tasks
        print_all_tasks(tasks, verbose=verbose)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@task.command()
@click.argument('task_id', type=int)
@click.option('-v', '--verbose', is_flag=True, help="Show detailed task info")
def get(task_id, verbose):
    """Get details of a specific task"""
    try:
        # Validate task_id is an integer
        try:
            task_id_int = int(task_id)
        except ValueError:
            click.echo(f"Error: Task ID must be an integer, got '{task_id}'", err=True)
            sys.exit(1)
            
        client = FleetManagerClient()
        response = client.get_task(task_id_int)
        
        if response.task:
            task = response.task
            if verbose:
                if getattr(task, 'goal_id', None):
                    try:
                        g = client.get_goal(task.goal_id).goal
                        task._goal = g
                    except Exception:
                        click.echo(f"[Warning] Goal ID {task.goal_id} not found for task {task.task_id}", err=True)
                if getattr(task, 'robot_id', None):
                    try:
                        r = client.get_robot_status(task.robot_id)
                        task._robot = r
                    except Exception:
                        click.echo(f"[Warning] Robot ID {task.robot_id} not found for task {task.task_id}", err=True)
                plans = [p for p in client.list_plans().plans if task.task_id in getattr(p, 'task_ids', [])]
                task._plans = plans
                task._tasks = []
                for plan in plans:
                    for tid in getattr(plan, 'task_ids', []):
                        try:
                            t = client.get_task(tid).task
                            task._tasks.append(t)
                        except Exception:
                            click.echo(f"[Warning] Task ID {tid} not found for plan {plan.plan_id}", err=True)
            print_task(task)
        else:
            click.echo(f"Error: {response.error}", err=True)
            sys.exit(1)
    except grpc.RpcError as e:
        status_code = e.code()
        status_details = e.details()
        click.echo(f"gRPC error: {status_code.name} - {status_details}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@task.command()
@click.argument('task_id', type=int)
def delete(task_id):
    """Delete a task"""
    try:
        # Validate task_id is an integer
        try:
            task_id_int = int(task_id)
        except ValueError:
            click.echo(f"Error: Task ID must be an integer, got '{task_id}'", err=True)
            sys.exit(1)
            
        client = FleetManagerClient()
        response = client.delete_task(task_id_int)
        if response.task:
            click.echo(f"Task {task_id_int} deleted successfully")
        else:
            click.echo(f"Error: {response.error}", err=True)
    except grpc.RpcError as e:
        status_code = e.code()
        status_details = e.details()
        click.echo(f"gRPC error: {status_code.name} - {status_details}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@cli.group()
def plan():
    """Plan management commands"""
    pass

@plan.command()
@click.argument('planning_strategy', type=click.Choice(PLANNING_STRATEGY_CHOICES, case_sensitive=False))
@click.argument('allocation_strategy', type=click.Choice(ALLOCATION_STRATEGY_CHOICES, case_sensitive=False))
@click.argument('goal_ids', required=True,type=str)
@click.option('-v', '--verbose', is_flag=True, help="Show detailed plan, goal, and task info")
def create(planning_strategy, allocation_strategy, goal_ids, verbose):
    """Create a new plan with planning strategy, allocator, and goal IDs"""
    client = None
    try:
        # Parse the comma-separated string into a list of ints
        try:
            goal_ids_list = [int(x.strip()) for x in goal_ids.split(',') if x.strip()]
        except Exception:
            click.echo(f"Error: Could not parse goal IDs from '{goal_ids}'", err=True)
            sys.exit(1)
        client = FleetManagerClient()
        # Add allocator info to the plan creation request if supported
        response = client.create_plan(planning_strategy, goal_ids_list, allocation_strategy)
        plan = response.plan
        if not plan.plan_id:
            click.echo(f"[Warning] Plan creation failed", err=True)
            return
        goals = [client.get_goal(gid).goal for gid in getattr(plan, 'goal_ids', [])]
        tasks = []
        for tid in getattr(plan, 'task_ids', []):
            try:
                t = client.get_task(tid).task
                tasks.append(t)
            except Exception:
                click.echo(f"[Warning] Task ID {tid} not found for plan {plan.plan_id}", err=True)
        print_plan(plan, goals, tasks, verbose)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        if client is not None:
            client.close()

@plan.command()
@click.argument('plan_id', type=int)
@click.option('-a', '--analyze-idle', is_flag=True, help="Analyze idle time and display task grid for the plan")
@click.option('-v', '--verbose', is_flag=True, help="Show detailed plan, goal, and task info")
def get(plan_id, analyze_idle, verbose):
    """Get details of a specific plan"""
    client = None
    try:
        client = FleetManagerClient()
        plan_resp = client.get_plan(plan_id)
        plan = plan_resp.plan
        if not plan.plan_id:
            click.echo(f"[Warning] Plan ID {plan_id} not found", err=True)
            return
        goals = [client.get_goal(gid).goal for gid in getattr(plan, 'goal_ids', [])]
        tasks = client.list_tasks(plan_ids=[plan.plan_id]).tasks
        print_plan(plan, goals=goals, tasks=tasks, verbose=verbose)
        if analyze_idle:
            # Perform idle analysis and print the grid
            report = client.analyze_idle_time(plan_id)
            click.echo(report)
            return
    except Exception as e:
        click.echo(f"Error getting plan: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@plan.command()
@click.option('-v', '--verbose', is_flag=True, help="Show detailed info for all plans")
def list(verbose):
    """List all plans"""
    try:
        client = FleetManagerClient()
        plans_resp = client.list_plans()
        plans = plans_resp.plans
        goals = client.list_goals().goals
        tasks = client.list_tasks().tasks
        print_all_plans(plans, goals=goals, tasks_by_plan=tasks, verbose=verbose)
    except Exception as e:
        click.echo(f"Error listing plans: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@plan.command()
@click.argument('plan_id', type=int)
def delete(plan_id):
    """Delete a plan"""
    try:
        client = FleetManagerClient()
        response = client.delete_plan(plan_id)
        if response.plan:
            click.echo(f"Plan {plan_id} deleted successfully")
        else:
            click.echo(f"Error: {response.error}", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@plan.command()
@click.argument('plan_id', type=int)
def start(plan_id):
    """Start a plan"""
    try:
        client = FleetManagerClient()
        response = client.start_plan(plan_id)
        if response.error == "":
            click.echo(f"Plan {plan_id} completed successfully")
        else:
            click.echo(f"Error: {response.error}", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@cli.group()
def world():
    """World statement management commands"""
    pass

@world.command()
@click.argument('statement', type=str)
def add(statement):
    """Add a new world statement."""
    try:
        client = FleetManagerClient()
        ws = client.add_world_statement(statement)
        if ws:
            click.echo(f"World statement {ws.id} added successfully")
            print_world_statement(ws)
        else:
            click.echo("Error: Failed to add world statement", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@world.command()
@click.argument('world_statement_id', type=str)
def delete(world_statement_id):
    """Remove a world statement by its ID."""
    try:
        client = FleetManagerClient()
        success = client.delete_world_statement(world_statement_id)
        if success:
            click.echo(f"World statement {world_statement_id} deleted successfully")
        else:
            click.echo(f"Error: Failed to delete world statement {world_statement_id}", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@world.command()
@click.argument('world_statement_id', type=str)
def get(world_statement_id):
    """Get a world statement by its ID."""
    try:
        client = FleetManagerClient()
        ws = client.get_world_statement(world_statement_id)
        if ws:
            print_world_statement(ws)
        else:
            click.echo(f"World statement {world_statement_id} not found", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

@world.command('list')
def list_world():
    """List all world statements."""
    try:
        client = FleetManagerClient()
        ws_list = client.list_world_statements()
        print_world_statement_list(ws_list)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    finally:
        client.close()

cli.add_command(goal)  # Add goal command group to main CLI
cli.add_command(task)  # Add task command group to main CLI
cli.add_command(plan)  # Add plan command group to main CLI
cli.add_command(world)  # Add world command group to main CLI

def main():
    """Entry point for the CLI"""
    try:
        cli()
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 