#!/usr/bin/env python3
"""
Demo: Prefect Deployment Architecture

This script demonstrates the complete Prefect-based automated retraining
architecture, showing how deployments work instead of direct function calls.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.automation.prefect_client import PrefectClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_prefect_architecture():
    """Demonstrate the complete Prefect deployment architecture."""

    print("🏗️ Prefect Deployment Architecture Demo")
    print("=" * 60)

    print("\n📋 Architecture Overview:")
    print("┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐")
    print("│   Simulation    │───▶│  Prefect API     │───▶│  Retraining     │")
    print("│   Engine        │    │  Deployments     │    │  Flow           │")
    print("└─────────────────┘    └──────────────────┘    └─────────────────┘")
    print("        │                        │                        │")
    print("   Performance            HTTP Requests            Model Updates")
    print("   Monitoring              + Parameters             + MLflow")

    print("\n🎯 Key Components:")
    print("  1. 📊 Performance Monitoring: Tracks model accuracy and drift")
    print("  2. 🔄 Prefect Deployments: API objects for remote triggering")
    print("  3. 🤖 Retraining Flow: Automated model update pipeline")
    print("  4. 📈 MLflow Integration: Experiment tracking and model versioning")

    # Initialize Prefect client
    print("\n🚀 Initializing Prefect Client...")
    try:
        client = PrefectClient()
        print("✅ Prefect client ready")

        # Demonstrate different trigger scenarios
        scenarios = [
            {
                "name": "Performance Degradation",
                "reason": "performance_drop",
                "context": {
                    "accuracy_drop": 0.08,
                    "current_accuracy": 0.52,
                    "threshold": 0.60,
                    "matches_since_retrain": 150
                }
            },
            {
                "name": "Data Drift Detection",
                "reason": "data_drift",
                "context": {
                    "drift_score": 0.15,
                    "drift_threshold": 0.10,
                    "features_drifted": ["home_odds", "away_odds"],
                    "weeks_since_retrain": 8
                }
            },
            {
                "name": "Scheduled Maintenance",
                "reason": "scheduled_retrain",
                "context": {
                    "schedule": "monthly",
                    "last_retrain": "2025-06-09",
                    "days_elapsed": 30,
                    "new_data_points": 380
                }
            }
        ]

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['name']}")
            print("-" * 40)

            # Show trigger context
            print("🔍 Trigger Context:")
            for key, value in scenario['context'].items():
                print(f"  - {key}: {value}")

            print(f"\n🚀 Triggering deployment for: {scenario['reason']}")

            # Trigger the deployment
            flow_run = await client.trigger_deployment_run(
                deployment_name="automated-retraining-flow/simulation-triggered-retraining",
                parameters={
                    "config_path": "config/retraining_config.yaml",
                    "triggers": [scenario['reason']],
                    "force_retrain": True,
                    "scenario_context": scenario['context']
                },
                wait_for_completion=False,  # Don't wait for demo speed
                timeout_seconds=30,
            )

            if flow_run:
                print(f"✅ Flow run started: {flow_run.id}")
                print(f"📊 State: {flow_run.state.type}")
                print(f"🌐 View in UI: http://localhost:4200/flow-runs/flow-run/{flow_run.id}")
            else:
                print("❌ Failed to trigger deployment")
                print("💡 This is expected if Prefect server is not running")

            print(f"⏱️  Waiting 3 seconds before next scenario...")
            await asyncio.sleep(3)

    except Exception as e:
        print(f"⚠️  Client initialization failed: {e}")
        print("💡 Expected if Prefect server is not running")
        print("\n🛠️  To run with full Prefect server:")
        print("   1. Terminal 1: prefect server start")
        print("   2. Terminal 2: python deployments/deploy_retraining_flow.py")
        print("   3. Terminal 3: python scripts/automation/demo_architecture.py")


def show_architecture_comparison():
    """Show the architectural difference between approaches."""

    print("\n🏛️ Architecture Comparison")
    print("=" * 60)

    print("\n❌ OLD ARCHITECTURE (Direct Function Calls):")
    print("```")
    print("┌─────────────┐    ┌─────────────────┐")
    print("│ Simulation  │───▶│ Retraining      │")
    print("│ Engine      │    │ Function        │")
    print("└─────────────┘    └─────────────────┘")
    print("       │                    │")
    print("  Direct Call          Local Execution")
    print("   (Tight Coupling)    (No Observability)")
    print("```")

    print("\n✅ NEW ARCHITECTURE (Prefect Deployments):")
    print("```")
    print("┌─────────────┐    ┌─────────────┐    ┌─────────────────┐")
    print("│ Simulation  │───▶│ Prefect API │───▶│ Retraining      │")
    print("│ Engine      │    │ Server      │    │ Deployment      │")
    print("└─────────────┘    └─────────────┘    └─────────────────┘")
    print("       │                  │                    │")
    print("  HTTP Request       Workflow Queue        Flow Execution")
    print("  (Loose Coupling)   (Full Observability)  (Scalable)")
    print("```")

    print("\n🏆 Benefits of New Architecture:")
    print("  ✅ Remote Triggering: API-based workflow execution")
    print("  ✅ Observability: Full visibility in Prefect UI")
    print("  ✅ Scalability: Distributed execution on workers")
    print("  ✅ Monitoring: Built-in flow run tracking")
    print("  ✅ Error Handling: Automatic retries and failure management")
    print("  ✅ Scheduling: Cron, interval, and event-based triggers")
    print("  ✅ Parameter Validation: Schema validation for inputs")
    print("  ✅ Version Control: Deployment versioning and rollback")


def show_implementation_details():
    """Show key implementation details."""

    print("\n🔧 Implementation Details")
    print("=" * 60)

    print("\n📁 Key Files:")
    print("  src/automation/prefect_client.py      - Prefect API client")
    print("  src/automation/retraining_flow.py     - Prefect flow definition")
    print("  deployments/deploy_retraining_flow.py - Deployment configuration")
    print("  src/simulation/retraining_orchestrator.py - Integration layer")

    print("\n🔄 Flow Execution Process:")
    print("  1. 📊 Monitor performance in simulation")
    print("  2. 🚨 Detect trigger condition (performance drop, drift, etc.)")
    print("  3. 🌐 Send HTTP request to Prefect API")
    print("  4. 📋 Create flow run with parameters")
    print("  5. ⚙️  Execute retraining flow on worker")
    print("  6. 💾 Update model if performance improves")
    print("  7. 📈 Log results to MLflow")
    print("  8. 📊 Update monitoring dashboard")

    print("\n🎛️ Configuration:")
    print("  - Deployment names: 'automated-retraining', 'simulation-triggered-retraining'")
    print("  - Parameters: config_path, triggers, force_retrain, context")
    print("  - Work pools: Default agent pool for local execution")
    print("  - Tags: 'mlops', 'retraining', 'simulation', 'production'")


async def main():
    """Main demo function."""

    print("🚀 Prefect Deployment Architecture Demo")
    print("=" * 70)

    # Show architectural concepts
    show_architecture_comparison()
    show_implementation_details()

    # Demo the actual architecture
    await demo_prefect_architecture()

    print("\n🎉 Demo Complete!")
    print("\n🌐 Next Steps:")
    print("  1. Visit Prefect UI: http://localhost:4200")
    print("  2. View flow runs and deployments")
    print("  3. Trigger deployments manually via UI")
    print("  4. Monitor execution logs and results")

    print("\n💡 Production Usage:")
    print("  - Run `prefect server start` for persistent server")
    print("  - Deploy flows with `python deployments/deploy_retraining_flow.py`")
    print("  - Integrate with CI/CD for automatic deployment updates")
    print("  - Scale with Prefect Cloud for production workloads")


if __name__ == "__main__":
    asyncio.run(main())
