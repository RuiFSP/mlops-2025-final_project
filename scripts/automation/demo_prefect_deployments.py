#!/usr/bin/env python3
"""
Demo: Prefect Deployments for Automated Retraining

This script demonstrates the difference between calling retraining functions
directly vs. using Prefect deployments for remote, API-based triggering.

Key Concepts:
- Prefect Deployments: Convert flows from function calls to API objects
- Remote Triggering: Trigger flows via Prefect API instead of direct calls
- Production MLOps: Use deployments for better observability and management
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.automation.prefect_client import PrefectClient
from src.simulation.season_simulator import SeasonSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_prefect_deployment_integration():
    """Demonstrate how simulation triggers Prefect deployments."""

    print("🎯 Demo: Simulation with Prefect Deployment Integration")
    print("=" * 60)

    # Initialize simulator with Prefect enabled
    simulator = SeasonSimulator(
        simulation_data_path="data/simulation/premier_league_2023_24.parquet",
        match_calendar_path="data/simulation/match_calendar.csv",
        model_path="models/model.pkl",
        output_dir="data/simulation_demo",
        use_prefect=True,  # Enable Prefect integration
        prefect_deployment_name="simulation-triggered-retraining",
    )

    print(f"✅ Simulator initialized with Prefect integration")
    print(f"📊 Simulator state: {simulator.get_simulation_state()}")

    # Simulate a few weeks to trigger retraining
    print("\n🏟️  Simulating matches to trigger retraining...")

    for week in range(1, 4):  # Simulate 3 weeks
        print(f"\n📅 Week {week}")
        result = simulator.simulate_week()

        # Display week results
        print(f"  - Matches: {len(result.get('predictions', []))}")
        print(f"  - Performance: {result.get('performance', {}).get('accuracy', 'N/A')}")

        # Check if retraining was triggered
        retraining_count = simulator.retraining_orchestrator.get_retraining_count()
        if retraining_count > 0:
            print(f"  🤖 Retraining triggered! Total retrainings: {retraining_count}")
            print("  🚀 This used Prefect deployment, not direct function call!")
            break

    print(f"\n📈 Final state: {simulator.get_simulation_state()}")
    print(f"🔄 Total retrainings: {simulator.retraining_orchestrator.get_retraining_count()}")


async def demo_direct_prefect_triggering():
    """Demonstrate direct Prefect deployment triggering."""

    print("\n🎯 Demo: Direct Prefect Deployment Triggering")
    print("=" * 60)

    # Initialize Prefect client
    try:
        client = PrefectClient()
        print("✅ Prefect client initialized")

        # Trigger retraining deployment directly
        print("🚀 Triggering simulation-triggered-retraining deployment...")

        flow_run = await client.trigger_deployment_run(
            deployment_name="simulation-triggered-retraining",
            parameters={
                "config_path": "config/retraining_config.yaml",
                "triggers": ["demo_manual_trigger"],
                "force_retrain": True,
                "simulation_context": {
                    "week": 99,
                    "trigger_reasons": ["demo"],
                    "performance_before": 0.45,
                }
            },
            wait_for_completion=True,
            timeout_seconds=60,
        )

        if flow_run:
            print(f"✅ Flow run completed: {flow_run.id}")
            print(f"📊 Final state: {flow_run.state.type}")
        else:
            print("❌ Flow run failed or timed out")

    except Exception as e:
        print(f"⚠️  Prefect deployment triggering failed: {e}")
        print("💡 This is expected if Prefect server is not running or deployments not served")


def compare_approaches():
    """Compare old vs new approach."""

    print("\n🔄 Comparison: Function Calls vs Prefect Deployments")
    print("=" * 60)

    print("❌ OLD APPROACH (Function Calls):")
    print("   - Simulation calls retraining_flow() directly")
    print("   - No observability or remote management")
    print("   - Harder to scale and monitor")
    print("   - Tight coupling between components")
    print("   - Example: retraining_flow.automated_retraining_flow()")

    print("\n✅ NEW APPROACH (Prefect Deployments):")
    print("   - Simulation triggers deployments via Prefect API")
    print("   - Full observability in Prefect UI")
    print("   - Remote triggering and management")
    print("   - Loose coupling via API")
    print("   - Example: client.trigger_deployment_run('simulation-triggered-retraining')")

    print("\n🏆 Benefits of Deployments:")
    print("   ✅ API-based triggering")
    print("   ✅ Better observability")
    print("   ✅ Remote management")
    print("   ✅ Scheduling capabilities")
    print("   ✅ Retry and error handling")
    print("   ✅ Parameter validation")
    print("   ✅ Scalability")


async def main():
    """Main demo function."""

    print("🚀 Prefect Deployments Integration Demo")
    print("=" * 70)

    # Show conceptual comparison
    compare_approaches()

    # Demo direct Prefect triggering
    await demo_direct_prefect_triggering()

    # Demo simulation with Prefect integration (requires deployment server)
    print("\n💡 To run full simulation demo with Prefect:")
    print("   1. Start Prefect server: prefect server start")
    print("   2. Serve deployments: python deployments/deploy_retraining_flow.py")
    print("   3. Run: python scripts/automation/demo_prefect_deployments.py --full")

    if "--full" in sys.argv:
        await demo_prefect_deployment_integration()
    else:
        print("   (Skipping full demo - use --full flag to run with deployments)")

    print("\n🎉 Demo completed!")
    print("🌐 View flows in Prefect UI: http://localhost:4200")


if __name__ == "__main__":
    asyncio.run(main())
