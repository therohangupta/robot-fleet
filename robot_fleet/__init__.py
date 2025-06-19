"""Robot Fleet Management System

This package provides a complete system for managing a fleet of heterogeneous robots.
It consists of two main components:
1. Robot Control Node - For managing and controlling the robot fleet
2. Robots - Base classes and implementations for different types of robots
"""

from .robots import RobotServerBase, TaskResult
from .robots.containers.manager import ContainerManager, ContainerInfo

__all__ = [
    # Robot base classes and implementations
    'RobotServerBase',
    'TaskResult',
    
    # Robot control and management
    'ContainerManager',
    'ContainerInfo'
] 