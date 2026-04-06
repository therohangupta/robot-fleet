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
    get_allocation_strategy_id,
    get_planning_strategy_id,
)
from .port_manager import get_used_ports, suggest_next_port, is_localhost
from .telemetry_client import (
    get_health_summary as get_telemetry_health_summary,
    get_robot_health as get_telemetry_robot_health,
    close_client as close_telemetry_client,
)

__all__ = [
    # Robot health (direct polling - legacy/fallback)
    "check_robot_health",
    "check_all_robots_health",
    # Robot health from Telemetry service (preferred)
    "get_telemetry_health_summary",
    "get_telemetry_robot_health",
    "close_telemetry_client",
    # YAML scanning
    "scan_embodiments",
    "scan_planner_types",
    "scan_allocator_types",
    "scan_all_method_types",
    "load_planner_summary",
    "find_yaml_for_robot",
    "get_allocation_strategy_id",
    "get_planning_strategy_id",
    # Port management
    "get_used_ports",
    "suggest_next_port",
    "is_localhost",
]
