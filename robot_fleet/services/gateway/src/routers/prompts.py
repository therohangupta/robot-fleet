"""
Method details viewing endpoints.

Provides read-only access to the prompts and details used by different
planning and allocation methods. Methods are discovered dynamically from
the planners and allocators types directories.
"""

import re
from typing import List
from fastapi import APIRouter, HTTPException

from ..config import PLANNER_TYPES_DIR, ALLOCATOR_TYPES_DIR
from ..services import scan_all_method_types, load_planner_summary

router = APIRouter(prefix="/methods")


@router.get("")
async def list_methods() -> List[dict]:
    """
    List all available planning and allocation methods with metadata.

    Scans both planner and allocator types directories and returns metadata
    from each method's summary.yaml file.
    """
    return scan_all_method_types()


@router.get("/{method_type}")
async def get_method(method_type: str) -> dict:
    """
    Get the details for a specific method (planner or allocator).

    Returns method metadata, prompts, and extracted template variables.

    Args:
        method_type: Method directory name (e.g., 'monolithic', 'lp')
    """
    # First try planners directory
    method_dir = PLANNER_TYPES_DIR / method_type
    if not method_dir.exists():
        # Try allocators directory
        method_dir = ALLOCATOR_TYPES_DIR / method_type
        if not method_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Method type not found: {method_type}"
            )

    summary = load_planner_summary(method_dir)
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"No summary.yaml found for: {method_type}"
        )

    # Read prompt files if they exist
    system_content = ""
    user_content = ""

    system_file = method_dir / "system.prompt"
    user_file = method_dir / "user.prompt"

    if system_file.exists():
        system_content = system_file.read_text()
    if user_file.exists():
        user_content = user_file.read_text()

    # Extract template variables from user prompt (e.g., {goal_id}, {robot_context})
    variables = list(set(re.findall(r'\{(\w+)\}', user_content)))

    return {
        "category": "planner" if method_dir.parent == PLANNER_TYPES_DIR else "allocator",
        "type": method_type,
        "name": summary.get("name", method_type),
        "description": summary.get("description", "").strip(),
        "method_type": summary.get("method_type", "unknown"),
        "output_format": summary.get("output_format", "").strip(),
        "example_output": summary.get("example_output", "").strip(),
        "example_behavior": summary.get("example_behavior", "").strip(),
        "prompts": summary.get("prompts", []),
        "system_prompt": system_content,
        "user_prompt": user_content,
        "variables": sorted(variables),
    }
