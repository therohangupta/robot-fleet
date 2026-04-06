"""
Robot health checking service.

Provides utilities for checking if robots are reachable and responsive
by pinging their HTTP endpoints.
"""

from typing import List, Dict, Any
import httpx

from ..config import HEALTH_CHECK_TIMEOUT_SECS, DEFAULT_ROBOT_HOST, DEFAULT_ROBOT_BASE_PORT


async def check_robot_health(
    host: str, 
    port: int, 
    timeout: float = HEALTH_CHECK_TIMEOUT_SECS
) -> Dict[str, Any]:
    """
    Check if a robot is reachable by pinging its HTTP endpoints.
    
    Attempts to connect to common health endpoints (/health, /) and
    returns connectivity status with latency if successful.
    
    Args:
        host: Robot's hostname or IP address
        port: Robot's HTTP port
        timeout: Connection timeout in seconds
        
    Returns:
        Dict with keys:
            - reachable: bool - True if robot responded
            - latency_ms: float - Response time in milliseconds (if reachable)
            - error: str - Error message (if not reachable)
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Try common health endpoints
            for endpoint in ["/health", "/"]:
                try:
                    url = f"http://{host}:{port}{endpoint}"
                    response = await client.get(url)
                    # Accept any non-5xx response as "alive"
                    if response.status_code < 500:
                        return {
                            "reachable": True, 
                            "latency_ms": response.elapsed.total_seconds() * 1000
                        }
                except Exception:
                    continue
            return {"reachable": False, "error": "No valid endpoint responded"}
    except httpx.TimeoutException:
        return {"reachable": False, "error": "Connection timeout"}
    except httpx.ConnectError:
        return {"reachable": False, "error": "Connection refused"}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


async def check_all_robots_health(robots: List[Dict]) -> List[Dict[str, Any]]:
    """
    Check health of multiple robots.
    
    Args:
        robots: List of robot dicts with task_server_info containing host/port
        
    Returns:
        List of health check results, each containing:
            - robot_id: str
            - host: str
            - port: int
            - reachable: bool
            - latency_ms or error: depending on reachability
    """
    results = []
    for robot in robots:
        host = robot.get("task_server_info", {}).get("host", DEFAULT_ROBOT_HOST)
        port = robot.get("task_server_info", {}).get("port", DEFAULT_ROBOT_BASE_PORT)
        health = await check_robot_health(host, port)
        results.append({
            "robot_id": robot["robot_id"],
            "host": host,
            "port": port,
            **health
        })
    return results
