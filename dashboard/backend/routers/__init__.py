"""
FastAPI routers for the Robot Fleet Dashboard API.

Each router handles a specific domain of endpoints:
- robots: Robot registration, management, and health checks
- goals: Goal CRUD operations
- plans: Plan creation, allocation, and execution
- tasks: Task management
- world: World state statements
- embodiments: Robot type templates
- prompts: LLM prompt viewing
- strategies: Available planning/allocation strategies
"""

from fastapi import APIRouter

from . import robots, goals, plans, tasks, world, embodiments, methods, strategies

# Create a combined router that includes all sub-routers
api_router = APIRouter(prefix="/api")

# Include all domain routers
api_router.include_router(robots.router, tags=["Robots"])
api_router.include_router(goals.router, tags=["Goals"])
api_router.include_router(plans.router, tags=["Plans"])
api_router.include_router(tasks.router, tags=["Tasks"])
api_router.include_router(world.router, tags=["World State"])
api_router.include_router(embodiments.router, tags=["Embodiments"])
api_router.include_router(methods.router, tags=["Methods"])
api_router.include_router(strategies.router, tags=["Strategies"])

__all__ = ["api_router"]
