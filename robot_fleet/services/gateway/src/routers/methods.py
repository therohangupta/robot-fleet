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


@router.get("/planners")
async def list_planners() -> List[dict]:
    """List all available planning methods with metadata."""
    return scan_planner_types()


@router.get("/allocators")
async def list_allocators() -> List[dict]:
    """List all available allocation methods with metadata."""
    return scan_allocator_types()


@router.get("/{method_id}")
async def get_method(method_id: int, category: str = None) -> dict:
    """
    Get the details for a specific method (planner or allocator).

    Returns method metadata, prompts, and extracted template variables.

    Args:
        method_id: Method ID from summary.yaml (e.g., 1, 2, 3)
    """
    # Scan all methods to find the one with matching ID
    all_methods = scan_all_method_types()
    method_info = None

    for method in all_methods:
        if method.get("id") == method_id:
            # If category is specified, ensure it matches
            if category and method.get("category") != category:
                continue
            method_info = method
            break

    if not method_info:
        raise HTTPException(
            status_code=404,
            detail=f"Method with ID {method_id} not found"
        )

    method_type = method_info["type"]
    category = method_info["category"]

    # Find the method directory
    base_dir = PLANNER_TYPES_DIR if category == "planner" else ALLOCATOR_TYPES_DIR
    method_dir = base_dir / method_type

    if not method_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Method directory not found: {method_type}"
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
        "category": category,
        "type": method_type,
        "id": method_id,
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
