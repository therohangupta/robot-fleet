#!/usr/bin/env python3
"""
Test script to verify the complete data flow for plan creation and retrieval.
This will test each step of the pipeline to identify where the data is lost.
"""
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_data_flow():
    print("🔍 Testing complete data flow pipeline...")

    # Test 1: Check if planner can store data
    print("\n1️⃣ Testing planner data storage...")
    try:
        from robot_fleet.server.planners.base import BasePlanner

        # Create a minimal concrete planner just for testing data storage
        class TestPlanner(BasePlanner):
            async def plan(self, goal_ids):
                return '{"tasks": []}'

        planner = TestPlanner()
        planner.planning_prompts = {"system": "Test system prompt", "user": "Test user prompt"}
        planner.planning_artifacts = {"goals": ["test goal"], "test_data": "test_value"}
        planner.server_logs = ["Test log entry 1", "Test log entry 2"]

        print(f"✅ Planner can store data: prompts={len(planner.planning_prompts)}, artifacts={len(planner.planning_artifacts)}, logs={len(planner.server_logs)}")

    except Exception as e:
        print(f"❌ Planner test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Check if database storage works
    print("\n2️⃣ Testing database storage...")
    try:
        from robot_fleet.robots.registry.instance_registry import RobotInstanceRegistry

        # Use a test database URL
        test_db_url = "sqlite+aiosqlite:///:memory:"

        registry = RobotInstanceRegistry(test_db_url)
        print("✅ Registry created")

        # Try to create a plan with data
        plan = await registry.create_plan(
            planning_strategy=1,  # MONOLITHIC
            allocation_strategy=0,  # NONE
            planning_prompts={"system": "Test system prompt", "user": "Test user prompt"},
            planning_artifacts={"goals": ["test goal"], "test_data": "test_value"},
            server_logs="Test log entry 1\nTest log entry 2"
        )
        print(f"✅ Plan created with ID: {plan.plan_id}")

        # Try to retrieve the plan
        retrieved_plan = await registry.get_plan(plan.plan_id)
        print(f"✅ Plan retrieved, checking additional data...")

        # Check if the additional data is there
        if hasattr(retrieved_plan, 'planning_prompts') and retrieved_plan.planning_prompts:
            print(f"✅ Found planning_prompts: {list(retrieved_plan.planning_prompts.keys())}")
        else:
            print("❌ No planning_prompts found")

        if hasattr(retrieved_plan, 'server_logs') and retrieved_plan.server_logs:
            print(f"✅ Found server_logs: {len(retrieved_plan.server_logs)} chars")
        else:
            print("❌ No server_logs found")

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎯 Data flow test completed!")
    print("If all tests passed, the issue is likely in the gRPC bridge or frontend.")

if __name__ == "__main__":
    asyncio.run(test_data_flow())