#!/usr/bin/env python3
"""
Clean all relevant tables in the mlops_db PostgreSQL database for a fresh start.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'mlops_password')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'mlops_db')}"
)


def clean_tables():
    tables = ["bets", "predictions", "wallet", "matches", "metrics"]
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        for table in tables:
            try:
                conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
                print(f"Truncated table: {table}")
            except Exception as e:
                print(f"Failed to truncate {table}: {e}")
        conn.commit()
    print("✅ All tables cleaned.")


if __name__ == "__main__":
    clean_tables()
