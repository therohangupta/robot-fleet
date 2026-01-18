"""
Business logic services for the Robot Fleet Dashboard.

Services contain reusable business logic that can be called from
multiple routers or other services. They abstract away the details
of data access and complex operations.
"""

from .robot_health import check_robot_health, check_all_robots_health
from .yaml_scanner import (
    scan_embodiments,
    scan_planner_types,
    scan_allocator_types,
    scan_all_method_types,
    load_planner_summary,
    find_yaml_for_robot,
)
from .port_manager import get_used_ports, suggest_next_port, is_localhost

__all__ = [
    # Robot health
    "check_robot_health",
    "check_all_robots_health",
    # YAML scanning
    "scan_embodiments",
    "scan_planner_types",
    "load_planner_summary",
    "find_yaml_for_robot",
    # Port management
    "get_used_ports",
    "suggest_next_port",
    "is_localhost",
]
