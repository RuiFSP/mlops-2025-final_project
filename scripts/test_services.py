#!/usr/bin/env python3
"""
Script to test connectivity to all Docker services.
"""

import requests
import psycopg2
import mlflow
import os

def test_postgres():
    """Test PostgreSQL connectivity."""
    print("🔍 Testing PostgreSQL...")
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DB', 'mlops_db'),
            user=os.getenv('POSTGRES_USER', 'mlops_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ PostgreSQL: Connected successfully - {version[0]}")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ PostgreSQL: Connection failed - {e}")
        return False

def test_mlflow():
    """Test MLflow connectivity."""
    print("🔍 Testing MLflow...")
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"✅ MLflow: Connected successfully - {len(experiments)} experiments found")
        return True
    except Exception as e:
        print(f"❌ MLflow: Connection failed - {e}")
        return False

def test_grafana():
    """Test Grafana connectivity."""
    print("🔍 Testing Grafana...")
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Grafana: Connected successfully")
            return True
        else:
            print(f"❌ Grafana: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Grafana: Connection failed - {e}")
        return False

def test_prefect():
    """Test Prefect connectivity."""
    print("🔍 Testing Prefect...")
    try:
        response = requests.get("http://localhost:4200", timeout=5)
        if response.status_code == 200:
            print("✅ Prefect: Connected successfully")
            return True
        else:
            print(f"❌ Prefect: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prefect: Connection failed - {e}")
        return False

def main():
    """Test all services."""
    print("🚀 Testing Docker Service Connectivity\n")
    
    results = {
        'postgres': test_postgres(),
        'mlflow': test_mlflow(),
        'grafana': test_grafana(),
        'prefect': test_prefect()
    }
    
    print(f"\n📊 Results Summary:")
    for service, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {service.upper()}")
    
    all_good = all(results.values())
    if all_good:
        print("\n🎉 All services are accessible!")
    else:
        print("\n⚠️  Some services have connectivity issues.")

if __name__ == "__main__":
    main() 