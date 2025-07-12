#!/usr/bin/env python3
"""
Grafana Setup Script
Automatically configures Grafana with PostgreSQL data source and imports the MLOps dashboard
"""

import json
import time
from pathlib import Path

import requests

# Grafana configuration
GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASS = "admin"


def wait_for_grafana():
    """Wait for Grafana to be ready"""
    print("🔄 Waiting for Grafana to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Grafana is ready!")
                return True
        except requests.exceptions.RequestException:
            print(f"⏳ Attempt {i+1}/30: Grafana not ready yet...")
            time.sleep(2)
    return False


def create_postgresql_datasource():
    """Create PostgreSQL data source in Grafana"""
    print("🔗 Creating PostgreSQL data source...")

    datasource_config = {
        "name": "MLOps PostgreSQL",
        "type": "postgres",
        "url": "localhost:5432",
        "database": "mlops_db",
        "user": "mlops_user",
        "secureJsonData": {"password": "mlops_password"},
        "jsonData": {"sslmode": "disable", "postgresVersion": 1300, "timescaledb": False},
        "access": "proxy",
        "isDefault": True,
    }

    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/datasources",
            auth=(GRAFANA_USER, GRAFANA_PASS),
            headers={"Content-Type": "application/json"},
            json=datasource_config,
            timeout=10,
        )

        if response.status_code == 200:
            print("✅ PostgreSQL data source created successfully!")
            return True
        elif response.status_code == 409:
            print("ℹ️ PostgreSQL data source already exists")
            return True
        else:
            print(f"❌ Failed to create data source: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error creating data source: {e}")
        return False


def import_dashboard():
    """Import the MLOps dashboard"""
    print("📊 Importing MLOps dashboard...")

    dashboard_path = (
        Path(__file__).parent.parent / "grafana" / "dashboards" / "simple_mlops_dashboard.json"
    )

    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False

    try:
        with open(dashboard_path) as f:
            dashboard_config = json.load(f)

        # Prepare dashboard for import
        import_payload = {
            "dashboard": dashboard_config,
            "overwrite": True,
            "inputs": [
                {
                    "name": "DS_POSTGRES",
                    "type": "datasource",
                    "pluginId": "postgres",
                    "value": "MLOps PostgreSQL",
                }
            ],
        }

        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/import",
            auth=(GRAFANA_USER, GRAFANA_PASS),
            headers={"Content-Type": "application/json"},
            json=import_payload,
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            dashboard_url = f"{GRAFANA_URL}/d/{result['uid']}"
            print("✅ Dashboard imported successfully!")
            print(f"🌐 Dashboard URL: {dashboard_url}")
            return True
        else:
            print(f"❌ Failed to import dashboard: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error importing dashboard: {e}")
        return False


def test_data_connection():
    """Test the PostgreSQL connection"""
    print("🔍 Testing PostgreSQL connection...")

    try:
        import psycopg2

        conn = psycopg2.connect(
            host="localhost", database="mlops_db", user="mlops_user", password="mlops_password"
        )

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM model_metrics")
        metrics_result = cursor.fetchone()
        metrics_count = metrics_result[0] if metrics_result else 0

        cursor.execute("SELECT COUNT(*) FROM model_predictions")
        predictions_result = cursor.fetchone()
        predictions_count = predictions_result[0] if predictions_result else 0

        conn.close()

        print("✅ Database connection successful!")
        print(f"📊 Found {metrics_count} metrics and {predictions_count} predictions")
        return True

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def main():
    """Main setup function"""
    print("\n" + "=" * 60)
    print("🎯 GRAFANA SETUP FOR MLOPS MONITORING")
    print("=" * 60)

    # Step 1: Wait for Grafana
    if not wait_for_grafana():
        print("❌ Grafana is not accessible. Please check if it's running.")
        return False

    # Step 2: Test database
    if not test_data_connection():
        print("❌ Database connection failed. Please check PostgreSQL.")
        return False

    # Step 3: Create data source
    if not create_postgresql_datasource():
        print("❌ Failed to create PostgreSQL data source.")
        return False

    # Step 4: Import dashboard
    if not import_dashboard():
        print("❌ Failed to import dashboard.")
        return False

    print("\n" + "=" * 60)
    print("🎉 GRAFANA SETUP COMPLETE!")
    print("=" * 60)
    print(f"🌐 Grafana URL: {GRAFANA_URL}")
    print(f"🔐 Username: {GRAFANA_USER}")
    print(f"🔐 Password: {GRAFANA_PASS}")
    print("📊 Dashboard: Go to Dashboards -> MLOps Dashboard")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        exit(1)
    else:
        print("\n✅ You can now view your MLOps metrics in Grafana!")
