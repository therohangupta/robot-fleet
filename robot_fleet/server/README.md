# Robot Fleet Server

Core orchestration component for multi-robot task planning and allocation.

## Components

### Planner
Generates task plans from high-level goals.

**Strategies:**
- **Monolithic**: Single sequential plan for all goals (tightly coupled scenarios)
- **Per-Goal DAG + Aggregation**: Separate DAG plans for each goal (independent, parallel execution)
- **Big DAG**: Single comprehensive DAG for multiple goals (complex interdependencies)

### Allocator
Assigns tasks from plans to specific robots.

**Strategies:**
- **LP**: Linear programming optimization (mathematical, load-balanced)
- **LLM**: OpenAI GPT-4 allocation (contextual understanding)

## Usage

```python
from robot_fleet.server.planner.planner import get_planner
from robot_fleet.server.allocator.allocator import get_allocator
from robot_fleet.proto.fleet_manager_pb2 import PlanningStrategy, AllocationStrategy

# Create plan
planner = get_planner(PlanningStrategy.DAG, db_url="your_db_url")
plan_json = await planner.plan(goal_ids=[1, 2, 3])
plan_id = await planner.save_plan_to_db(plan_json, PlanningStrategy.DAG, AllocationStrategy.LP, [1, 2, 3])

# Allocate tasks
allocator = get_allocator(AllocationStrategy.LP, db_url="your_db_url")
allocation = await allocator.allocate(plan_id)
```

## Configuration

```bash
# Database
export DATABASE_URL="postgresql+asyncpg://user:pass@host:port/db"

# OpenAI (for LLM strategies)
export OPENAI_API_KEY="your_key"
```

## Strategy Selection

| Scenario | Planning | Allocation |
|----------|----------|------------|
| Sequential goals | Monolithic | LP |
| Independent goals | DAG | LLM |
| Complex interdependencies | Big DAG | Cost-Based |
| Load balancing priority | Any | LP |
| Contextual reasoning | Any | LLM |
