#!/usr/bin/env python3
import os

from sqlalchemy import create_engine, text

DB_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'mlops_password')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'mlops_db')}"
)


def print_table_counts():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        for table in ["predictions", "bets", "wallet"]:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f"{table}: {count} rows")
            except Exception as e:
                print(f"{table}: ERROR - {e}")


if __name__ == "__main__":
    print_table_counts()
