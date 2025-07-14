#!/usr/bin/env python3
"""
Test script for checking the status of all MLOps services
"""

import sys
import requests
from urllib.parse import urlparse

def test_postgres():
    """Test PostgreSQL connectivity."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            dbname="mlops_db",
            user="mlops_user",
            password="mlops_password"
        )
        conn.close()
        print("✅ PostgreSQL: Connected successfully")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL: Connection failed - {e}")
        return False

def test_mlflow():
    """Test MLflow connectivity."""
    print("🔍 Testing MLflow...")
    try:
        response = requests.get("http://127.0.0.1:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow: Connected successfully")
            return True
        else:
            print(f"❌ MLflow: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MLflow: Connection failed - {e}")
        return False

def test_api():
    """Test FastAPI connectivity."""
    print("🔍 Testing FastAPI...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI: Connected successfully")
            return True
        else:
            print(f"❌ FastAPI: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FastAPI: Connection failed - {e}")
        return False

def test_prefect():
    """Test Prefect connectivity."""
    print("🔍 Testing Prefect...")
    try:
        response = requests.get("http://localhost:4200/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Prefect: Connected successfully")
            return True
        else:
            print(f"❌ Prefect: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prefect: Connection failed - {e}")
        return False

def test_streamlit():
    """Test Streamlit connectivity."""
    print("🔍 Testing Streamlit...")
    try:
        response = requests.get("http://localhost:8501/", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit: Connected successfully")
            return True
        else:
            print(f"❌ Streamlit: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Streamlit: Connection failed - {e}")
        return False

def main():
    """Run all service tests."""
    print("🔍 Testing MLOps services...")
    
    results = {
        "postgres": test_postgres(),
        "mlflow": test_mlflow(),
        "api": test_api(),
        "prefect": test_prefect(),
        "streamlit": test_streamlit(),
    }
    
    all_passed = all(results.values())
    
    print("\n📊 Summary:")
    for service, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {service}")
    
    if all_passed:
        print("\n🎉 All services are running correctly!")
        return 0
    else:
        print("\n⚠️ Some services are not running correctly.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
