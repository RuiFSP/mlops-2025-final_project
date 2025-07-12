#!/usr/bin/env python3
"""
Demo Access Guide for End-to-End MLOps Monitoring
This script provides a complete guide to accessing and using the monitoring stack
"""

import time
import webbrowser


def print_header():
    """Print the demo header"""
    print("\n" + "=" * 80)
    print("🎯 MLOPS MONITORING STACK - END-TO-END DEMO ACCESS GUIDE")
    print("=" * 80)
    print("This demo shows you how to access both Prefect flows and Grafana metrics!")
    print("=" * 80)


def print_service_status():
    """Print the status of all services"""
    print("\n📊 MONITORING SERVICES STATUS:")
    print("-" * 50)

    services = [
        ("🔧 Prefect UI", "http://localhost:4200", "Flow orchestration and monitoring"),
        ("📈 Grafana", "http://localhost:3000", "Metrics visualization and dashboards"),
        ("🔍 MLflow", "http://localhost:5000", "Model tracking and experiments"),
        ("🚀 API Server", "http://localhost:8000", "Live predictions and health checks"),
        ("🗄️ PostgreSQL", "localhost:5432", "Metrics storage and monitoring data"),
    ]

    for name, url, description in services:
        print(f"{name:15} {url:25} - {description}")

    print("-" * 50)


def print_prefect_guide():
    """Print Prefect UI access guide"""
    print("\n🔧 PREFECT UI ACCESS GUIDE:")
    print("-" * 40)
    print("1. 🌐 Open: http://localhost:4200")
    print("2. 📊 You should see:")
    print("   - Flow runs dashboard")
    print("   - Recent 'MLOps Monitoring Demo' executions")
    print("   - Task execution details and logs")
    print("   - Flow run timeline and status")
    print("3. 🔍 Click on any flow run to see:")
    print("   - Individual task execution")
    print("   - Task logs and outputs")
    print("   - Flow execution graph")
    print("   - Performance metrics")
    print("4. 🎯 Look for flows named:")
    print("   - 'MLOps Monitoring Demo'")
    print("   - Recent executions with green (success) status")


def print_grafana_setup_guide():
    """Print Grafana setup guide"""
    print("\n📈 GRAFANA SETUP GUIDE:")
    print("-" * 40)
    print("1. 🌐 Open: http://localhost:3000")
    print("2. 🔐 Login with:")
    print("   - Username: admin")
    print("   - Password: admin")
    print("3. 🔗 Add PostgreSQL Data Source:")
    print("   - Go to Configuration > Data Sources")
    print("   - Click 'Add data source'")
    print("   - Select 'PostgreSQL'")
    print("   - Configure:")
    print("     • Host: localhost:5432")
    print("     • Database: mlops_db")
    print("     • User: mlops_user")
    print("     • Password: mlops_password")
    print("     • SSL Mode: disable")
    print("   - Click 'Save & Test'")
    print("4. 📊 Import Dashboard:")
    print("   - Go to Dashboard > Import")
    print("   - Upload: grafana/dashboards/mlops_dashboard.json")
    print("   - Select your PostgreSQL data source")
    print("   - Click 'Import'")
    print("5. 📈 View Metrics:")
    print("   - Model performance metrics")
    print("   - Prediction accuracy over time")
    print("   - System health indicators")
    print("   - Alert status")


def print_database_verification():
    """Print database verification guide"""
    print("\n🗄️ DATABASE VERIFICATION:")
    print("-" * 40)
    print("You can verify metrics are stored by connecting to PostgreSQL:")
    print("psql -h localhost -U mlops_user -d mlops_db")
    print("Password: mlops_password")
    print("")
    print("Then run these queries:")
    print("SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT 10;")
    print("SELECT * FROM model_predictions ORDER BY timestamp DESC LIMIT 5;")


def print_real_time_demo():
    """Print real-time demo instructions"""
    print("\n🎬 REAL-TIME MONITORING DEMO:")
    print("-" * 40)
    print("To see real-time monitoring in action:")
    print("1. 🔧 Keep Prefect UI open (http://localhost:4200)")
    print("2. 📈 Keep Grafana open (http://localhost:3000)")
    print("3. 🚀 Run monitoring flows:")
    print("   cd scripts && uv run python test_simple_monitoring.py")
    print("4. 👀 Watch in real-time:")
    print("   - Prefect UI: New flow runs appearing")
    print("   - Grafana: Metrics updating (refresh dashboard)")
    print("   - Database: New records being inserted")


def print_advanced_features():
    """Print advanced features guide"""
    print("\n🚀 ADVANCED FEATURES:")
    print("-" * 40)
    print("1. 📊 API Health Check:")
    print("   curl http://localhost:8000/health")
    print("2. 🔮 Make Predictions:")
    print("   curl -X POST http://localhost:8000/predict \\")
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"home_team": "Arsenal", "away_team": "Chelsea"}\'')
    print("3. 📈 Get Model Info:")
    print("   curl http://localhost:8000/model/info")
    print("4. 🎯 View Upcoming Matches:")
    print("   curl http://localhost:8000/matches/upcoming")


def open_services(open_browser=True):
    """Open all monitoring services in browser"""
    if not open_browser:
        return

    print("\n🌐 OPENING MONITORING SERVICES...")
    urls = [
        "http://localhost:4200",  # Prefect
        "http://localhost:3000",  # Grafana
        "http://localhost:5000",  # MLflow
        "http://localhost:8000",  # API
    ]

    for url in urls:
        try:
            webbrowser.open(url)
            print(f"✅ Opened: {url}")
            time.sleep(1)  # Small delay between opens
        except Exception as e:
            print(f"❌ Failed to open {url}: {e}")


def print_troubleshooting():
    """Print troubleshooting guide"""
    print("\n🔧 TROUBLESHOOTING:")
    print("-" * 40)
    print("If services are not accessible:")
    print("1. Check if services are running:")
    print("   ps aux | grep -E '(prefect|grafana|mlflow|uvicorn)'")
    print("2. Check ports:")
    print("   netstat -tlnp | grep -E '(4200|3000|5000|8000)'")
    print("3. Restart services if needed:")
    print("   - Prefect: uv run prefect server start --host 0.0.0.0 --port 4200")
    print("   - Grafana: sudo systemctl restart grafana-server")
    print("   - MLflow: uv run mlflow server --host 127.0.0.1 --port 5000")
    print("   - API: cd src/api && uv run uvicorn main:app --host 0.0.0.0 --port 8000")


def main():
    """Main demo function"""
    print_header()
    print_service_status()
    print_prefect_guide()
    print_grafana_setup_guide()
    print_database_verification()
    print_real_time_demo()
    print_advanced_features()
    print_troubleshooting()

    print("\n" + "=" * 80)
    print("🎯 NEXT STEPS:")
    print("=" * 80)
    print("1. 🔧 Open Prefect UI: http://localhost:4200")
    print("2. 📈 Set up Grafana: http://localhost:3000")
    print("3. 🎬 Run monitoring demo: uv run python test_simple_monitoring.py")
    print("4. 👀 Watch metrics update in real-time!")
    print("=" * 80)

    # Ask user if they want to open browsers
    try:
        response = input(
            "\n🌐 Would you like to open all monitoring services in your browser? (y/n): "
        )
        if response.lower() in ["y", "yes"]:
            open_services(True)
        else:
            print("✅ You can manually open the URLs listed above")
    except KeyboardInterrupt:
        print("\n✅ Demo guide completed")

    print("\n🎉 End-to-End Monitoring Demo Ready!")
    print("Enjoy exploring your MLOps monitoring stack! 🚀")


if __name__ == "__main__":
    main()
