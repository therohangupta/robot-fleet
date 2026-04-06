"""
YAML configuration scanning service.

Provides utilities for discovering and loading YAML configurations
for robot embodiments and planner types.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)

from ..config import EMBODIMENTS_DIR, PLANNER_TYPES_DIR, ALLOCATOR_TYPES_DIR, DEFAULT_ROBOT_BASE_PORT, REPO_ROOT


# =============================================================================
# Embodiment Scanning (Robot Type Templates)
# =============================================================================

def scan_embodiments() -> List[Dict[str, Any]]:
    """
    Scan the embodiments directory for available robot type YAMLs.
    
    Looks for YAML files in subdirectories of EMBODIMENTS_DIR,
    skipping any files prefixed with 'fake_' (test configurations).
    
    Returns:
        List of embodiment dicts with:
            - name: str - Human-readable name from metadata
            - description: str - Description from metadata
            - capabilities: List[str] - Robot capabilities
            - default_port: int - Default task server port
            - config_path: str - Relative path to YAML file
            - container_image: str - Docker image if containerized
    """
    embodiments = []
    
    if not EMBODIMENTS_DIR.exists():
        return embodiments
    
    for subdir in EMBODIMENTS_DIR.iterdir():
        if not subdir.is_dir():
            continue
            
        # Look for YAML files in each subdirectory
        for yaml_file in subdir.glob("*.yaml"):
            # Skip fake_* variants (test configurations)
            if yaml_file.name.startswith("fake_"):
                continue
                
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                
                # Extract project-relative path for portability
                embodiments.append({
                    "name": config.get("metadata", {}).get("name", yaml_file.stem),
                    "description": config.get("metadata", {}).get("description", ""),
                    "capabilities": config.get("capabilities", []),
                    "default_port": config.get("taskServer", {}).get("port", DEFAULT_ROBOT_BASE_PORT),
                    "config_path": str(yaml_file.relative_to(REPO_ROOT)),
                    "container_image": config.get("container", {}).get("image", ""),
                })
            except Exception as e:
                logger.error("Error loading %s: %s", yaml_file, e)
    
    return embodiments


def find_yaml_for_robot(robot: Dict) -> Optional[str]:
    """
    Find the YAML config path for a robot based on its type.
    
    Attempts to locate the embodiment YAML by matching the robot's
    type name to known embodiment directories.
    
    Args:
        robot: Robot dict with 'robot_type' key
        
    Returns:
        Absolute path to YAML file if found, None otherwise
    """
    robot_type = robot.get("robot_type", "").lower()
    
    # Try common patterns
    possible_paths = [
        EMBODIMENTS_DIR / robot_type / f"{robot_type}.yaml",
    ]
    
    # Add specific mappings for known types
    if "moma" in robot_type:
        possible_paths.append(EMBODIMENTS_DIR / "moma" / "moma.yaml")
    if "nav" in robot_type:
        possible_paths.append(EMBODIMENTS_DIR / "nav" / "nav.yaml")
    if "pick" in robot_type:
        possible_paths.append(EMBODIMENTS_DIR / "pick_place" / "pick_place.yaml")
    
    for path in possible_paths:
        if path and path.exists():
            return str(path)
    
    return None


# =============================================================================
# Planner Type Scanning (LLM Prompts)
# =============================================================================

def load_planner_summary(planner_dir: Path) -> Dict[str, Any]:
    """
    Load summary.yaml metadata from a planner directory.
    
    Args:
        planner_dir: Path to planner type directory
        
    Returns:
        Dict with planner metadata (name, description, output_format)
        or empty dict if no summary.yaml exists
    """
    summary_file = planner_dir / "summary.yaml"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def _scan_method_types(base_dir: Path, method_category: str) -> List[Dict[str, Any]]:
    """
    Scan a method types directory (planners or allocators) and return metadata.

    Args:
        base_dir: Directory containing method type subdirectories
        method_category: "planner" or "allocator"

    Returns:
        List of method dicts with metadata from summary.yaml
    """
    methods = []

    if not base_dir.exists():
        return methods

    for item in base_dir.iterdir():
        # Skip __pycache__ and non-directories
        if not item.is_dir() or item.name.startswith('__'):
            continue

        summary = load_planner_summary(item)

        # Only include if summary.yaml exists
        if not summary:
            continue

        # Check for prompt files
        prompts = []
        if method_category == "planner":
            system_file = item / "system.prompt"
            user_file = item / "user.prompt"
            if system_file.exists():
                prompts.append({"type": "system", "description": "System prompt defining planning role"})
            if user_file.exists():
                prompts.append({"type": "user", "description": "Template with goals, robot context, and world statements"})
        else:  # allocator
            # Allocators may have different prompt structures
            system_file = item / "system.prompt"
            user_file = item / "user.prompt"
            if system_file.exists():
                prompts.append({"type": "system", "description": "System prompt defining allocation role"})
            if user_file.exists():
                prompts.append({"type": "user", "description": "Template with robot/task context and allocation instructions"})

        methods.append({
            "category": method_category,
            "type": item.name,
            "id": summary.get("id"),
            "name": summary.get("name", item.name),
            "description": summary.get("description", "").strip(),
            "method_type": summary.get("method_type", "unknown"),
            "output_format": summary.get("output_format", "").strip(),
            "example_output": summary.get("example_output", "").strip(),
            "example_behavior": summary.get("example_behavior", "").strip(),
            "prompts": summary.get("prompts", prompts),  # Use summary prompts or detected ones
        })

    return methods


def scan_planner_types() -> List[Dict[str, Any]]:
    """Scan planner types directory and return metadata for each planner."""
    return _scan_method_types(PLANNER_TYPES_DIR, "planner")


def scan_allocator_types() -> List[Dict[str, Any]]:
    """Scan allocator types directory and return metadata for each allocator."""
    return _scan_method_types(ALLOCATOR_TYPES_DIR, "allocator")


def scan_all_method_types() -> List[Dict[str, Any]]:
    """
    Scan both planner and allocator types and return combined metadata.

    Returns all method types (planners and allocators) in a single list.
    """
    methods = []
    methods.extend(scan_planner_types())
    methods.extend(scan_allocator_types())
    return methods
