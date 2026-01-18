"""
Port management service.

Provides utilities for tracking and allocating ports for robot instances,
particularly for locally-hosted (fake) robots that share localhost.
"""

from typing import Set

from ..config import DEFAULT_BASE_PORT
from ..dependencies import get_bridge


def is_localhost(host: str) -> bool:
    """
    Check if a host string refers to localhost.
    
    Args:
        host: Hostname or IP address
        
    Returns:
        True if host is any localhost variant
    """
    return host.lower() in ('localhost', '127.0.0.1', '0.0.0.0', '::1')


def get_used_ports() -> Set[int]:
    """
    Get set of ports currently in use by registered robots.
    
    Queries the robot registry via gRPC to find all ports
    that are already assigned to robot instances.
    
    Returns:
        Set of port numbers currently in use
    """
    bridge = get_bridge()
    robots = bridge.list_robots("all")
    
    used = set()
    for robot in robots:
        task_server = robot.get("task_server_info")
        if task_server and task_server.get("port"):
            used.add(task_server["port"])
    
    return used


def suggest_next_port(base_port: int = DEFAULT_BASE_PORT) -> int:
    """
    Suggest the next available port for a new robot instance.
    
    Finds the lowest available port starting from base_port
    that isn't already in use by another robot.
    
    Args:
        base_port: Starting port number to search from
        
    Returns:
        First available port number >= base_port
    """
    used = get_used_ports()
    port = base_port
    while port in used:
        port += 1
    return port
