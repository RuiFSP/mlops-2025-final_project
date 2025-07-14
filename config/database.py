"""
Database configuration for the MLOps system
"""

import os

# Database connection parameters
DB_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "database": os.environ.get("POSTGRES_DB", "mlops_db"),
    "user": os.environ.get("POSTGRES_USER", "mlops_user"),
    "password": os.environ.get("POSTGRES_PASSWORD", "mlops_password"),
}


def get_db_config():
    """Return database configuration dictionary"""
    return DB_CONFIG.copy()
