"""
Strategy options endpoint.

Returns the available planning and allocation strategies
that can be used when creating plans.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/strategies")


@router.get("")
async def get_strategies():
    """
    Get available planning and allocation strategies.
    
    Returns descriptions of each strategy to help users
    choose the appropriate one for their use case.
    """
    return {
        "planning": [
            {
                "value": "monolithic",
                "label": "Monolithic",
                "description": "LLM generates a single sequential task list for all goals"
            },
            {
                "value": "dag",
                "label": "DAG",
                "description": "LLM generates parallel task DAGs per goal"
            },
            {
                "value": "big_dag",
                "label": "Big DAG",
                "description": "LLM generates one comprehensive DAG spanning all goals"
            },
        ],
        "allocation": [
            {
                "value": "lp",
                "label": "Linear Programming",
                "description": "Mathematical optimization for balanced load distribution"
            },
            {
                "value": "llm",
                "label": "LLM (GPT-4)",
                "description": "AI-powered allocation considering context and robot state"
            },
            {
                "value": "cost_based",
                "label": "Cost-Based",
                "description": "Iterative assignment minimizing task switching costs"
            },
            {
                "value": "none",
                "label": "None (Unallocated)",
                "description": "Skip allocation — assign robots later"
            },
        ]
    }
