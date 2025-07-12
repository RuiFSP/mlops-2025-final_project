#!/usr/bin/env python3
"""
Script to set up Grafana data source and import basic dashboard.
"""

import json
import requests
import time
import os

# Grafana configuration
GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASSWORD = "admin"

# PostgreSQL data source configuration
POSTGRES_DS = {
    "name": "PostgreSQL",
    "type": "postgres",
    "access": "proxy",
    "isDefault": True,
    "url": "postgres:5432",
    "database": "mlops_db",
    "user": "mlops_user",
    "basicAuth": False,
    "secureJsonData": {
        "password": "mlops_password"
    },
    "jsonData": {
        "sslmode": "disable"
    }
}

def setup_grafana():
    """Set up Grafana data source and dashboard."""
    print("🚀 Setting up Grafana...")
    
    # Create session
    session = requests.Session()
    session.auth = (GRAFANA_USER, GRAFANA_PASSWORD)
    
    # Test connection
    try:
        response = session.get(f"{GRAFANA_URL}/api/health")
        if response.status_code == 200:
            print("✅ Grafana is accessible")
        else:
            print(f"❌ Grafana health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Grafana: {e}")
        return False
    
    # Add PostgreSQL data source
    print("📊 Adding PostgreSQL data source...")
    try:
        response = session.post(
            f"{GRAFANA_URL}/api/datasources",
            json=POSTGRES_DS,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ PostgreSQL data source added successfully")
        elif response.status_code == 409:
            print("ℹ️ PostgreSQL data source already exists")
        else:
            print(f"❌ Failed to add data source: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error adding data source: {e}")
        return False
    
    # Import dashboard
    print("📈 Importing basic dashboard...")
    try:
        with open("grafana/dashboards/basic_dashboard.json", "r") as f:
            dashboard_config = json.load(f)
        
        # Add data source reference
        dashboard_config["dashboard"]["__inputs"] = [
            {
                "name": "DS_POSTGRES",
                "label": "PostgreSQL",
                "description": "",
                "type": "datasource",
                "pluginId": "postgres",
                "pluginName": "PostgreSQL"
            }
        ]
        
        dashboard_config["dashboard"]["__requires"] = [
            {
                "type": "grafana",
                "id": "grafana",
                "name": "Grafana",
                "version": "8.0.0"
            },
            {
                "type": "datasource",
                "id": "postgres",
                "name": "PostgreSQL",
                "version": "1.0.0"
            }
        ]
        
        # Replace data source references
        for panel in dashboard_config["dashboard"]["panels"]:
            if "targets" in panel:
                for target in panel["targets"]:
                    if "datasource" in target:
                        target["datasource"] = {"type": "postgres", "uid": "postgres"}
        
        response = session.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            json={"dashboard": dashboard_config["dashboard"], "overwrite": True},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            dashboard_url = f"{GRAFANA_URL}{result['url']}"
            print(f"✅ Dashboard imported successfully!")
            print(f"🌐 Dashboard URL: {dashboard_url}")
            return True
        else:
            print(f"❌ Failed to import dashboard: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error importing dashboard: {e}")
        return False

if __name__ == "__main__":
    success = setup_grafana()
    if success:
        print("\n🎉 Grafana setup completed successfully!")
        print(f"📊 Access Grafana at: {GRAFANA_URL}")
        print("👤 Username: admin")
        print("🔑 Password: admin")
    else:
        print("\n❌ Grafana setup failed!")
        exit(1) 