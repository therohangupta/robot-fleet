"""
Base allocator classes and allocation strategy management.

This module contains the abstract base class for allocators and the
factory function for creating allocator instances.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pathlib import Path

from packages.fleet_sdk.src.instance_registry import RobotInstanceRegistry
from packages.config import DATABASE_URL
from ..formats.formats import Allocation, RobotTask

logger = logging.getLogger(__name__)


class BaseAllocator(ABC):
    """Abstract base class for all allocators."""

    def __init__(self, db_url: Optional[str] = None, registry: Optional[RobotInstanceRegistry] = None):
        self.registry = registry or RobotInstanceRegistry(db_url or DATABASE_URL)
        # Load world statements on initialization
        logger.info(f"Initialized {self.__class__.__name__}.")

        # Storage for allocation artifacts and prompts
        self.allocation_prompts = {}
        self.allocation_artifacts = {}
        self.server_logs = []

    def _load_prompt(self, prompt_type: str) -> str:
        """Load a prompt file from the allocator's directory.

        Args:
            prompt_type: Type of prompt ('system' or 'user')

        Returns:
            The prompt content as a string
        """
        # Get the directory of the concrete allocator class
        allocator_dir = Path(self.__class__.__module__.replace('.', '/')).parent
        prompt_file = allocator_dir / f"{prompt_type}.prompt"

        try:
            with open(prompt_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt file {prompt_file}: {e}")
            raise

    @abstractmethod
    async def allocate(self, plan_id: int) -> Dict[int, str]:
        """Allocate tasks from the given plan_id to robots. Returns a mapping from task_id to assigned robot_id."""
        pass


def get_allocator(allocation_strategy: int, db_url: str = None, registry: Optional[RobotInstanceRegistry] = None):
    """
    Return the appropriate Allocator instance based on the allocation strategy enum value.

    Args:
        allocation_strategy: Integer enum value from fleet_manager_pb2.AllocationStrategy
        db_url: Optional database URL for the allocator
        registry: Optional existing RobotInstanceRegistry to reuse

    Returns:
        An allocator instance for the given strategy

    Raises:
        ValueError: If allocation_strategy is NONE or unknown
    """
    from packages.proto import fleet_manager_pb2

    if allocation_strategy == fleet_manager_pb2.AllocationStrategy.LP:
        from .types.lp import LPAllocator
        return LPAllocator(db_url, registry=registry)
    elif allocation_strategy == fleet_manager_pb2.AllocationStrategy.LLM:
        from .types.llm import LLMAllocator
        return LLMAllocator(db_url, registry=registry)
    elif allocation_strategy == fleet_manager_pb2.AllocationStrategy.COST_BASED:
        from .types.cost_based import CostBasedAllocator
        return CostBasedAllocator(db_url, registry=registry)
    elif allocation_strategy == fleet_manager_pb2.AllocationStrategy.NONE:
        raise ValueError(
            "NONE strategy means no allocation. "
            "Check for NONE before calling get_allocator() and skip allocation."
        )
    else:
        raise ValueError(f"Unknown allocation strategy: {allocation_strategy}")