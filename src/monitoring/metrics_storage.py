"""Metrics storage using PostgreSQL for MLOps monitoring."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class PostgreSQLMetricsStorage:
    """Store and retrieve metrics using PostgreSQL."""

    def __init__(self):
        """Initialize the metrics storage."""
        self.connection_params = {
            'host': os.environ.get('POSTGRES_HOST', 'postgres'),
            'port': int(os.environ.get('POSTGRES_PORT', 5432)),
            'database': os.environ.get('POSTGRES_DB', 'mlflow'),
            'user': os.environ.get('POSTGRES_USER', 'mlflow'),
            'password': os.environ.get('POSTGRES_PASSWORD', 'mlflow_password'),
        }
        self._init_tables()

    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(**self.connection_params)

    def _init_tables(self):
        """Initialize the metrics tables."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Create metrics table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS metrics (
                            id SERIAL PRIMARY KEY,
                            metric_name VARCHAR(255) NOT NULL,
                            metric_value DOUBLE PRECISION NOT NULL,
                            metric_type VARCHAR(50) DEFAULT 'gauge',
                            labels JSONB,
                            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        )
                    """)
                    
                    # Create index for efficient queries
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                        ON metrics(metric_name, timestamp)
                    """)
                    
                    # Create index for labels
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_metrics_labels 
                        ON metrics USING GIN(labels)
                    """)
                    
                    conn.commit()
                    logger.info("✅ Metrics tables initialized successfully")
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize metrics tables: {e}")

    def store_metric(self, name: str, value: float, metric_type: str = 'gauge', 
                    labels: Optional[Dict[str, Any]] = None, timestamp: Optional[datetime] = None):
        """Store a metric in PostgreSQL."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO metrics (metric_name, metric_value, metric_type, labels, timestamp)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (name, value, metric_type, labels, timestamp or datetime.now()))
                    conn.commit()
                    logger.debug(f"✅ Stored metric {name}={value}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to store metric {name}: {e}")

    def store_counter(self, name: str, value: float, labels: Optional[Dict[str, Any]] = None):
        """Store a counter metric."""
        self.store_metric(name, value, 'counter', labels)

    def store_gauge(self, name: str, value: float, labels: Optional[Dict[str, Any]] = None):
        """Store a gauge metric."""
        self.store_metric(name, value, 'gauge', labels)

    def get_metric_history(self, name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for the last N hours."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT metric_name, metric_value, metric_type, labels, timestamp
                        FROM metrics 
                        WHERE metric_name = %s 
                        AND timestamp >= NOW() - INTERVAL '%s hours'
                        ORDER BY timestamp ASC
                    """, (name, hours))
                    return cursor.fetchall()
                    
        except Exception as e:
            logger.error(f"❌ Failed to get metric history for {name}: {e}")
            return []

    def get_latest_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the latest value for a metric."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT metric_name, metric_value, metric_type, labels, timestamp
                        FROM metrics 
                        WHERE metric_name = %s 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """, (name,))
                    result = cursor.fetchone()
                    return dict(result) if result else None
                    
        except Exception as e:
            logger.error(f"❌ Failed to get latest metric for {name}: {e}")
            return None

    def get_metrics_summary(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get a summary of all metrics for the last N hours."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT 
                            metric_name,
                            metric_type,
                            COUNT(*) as count,
                            AVG(metric_value) as avg_value,
                            MIN(metric_value) as min_value,
                            MAX(metric_value) as max_value,
                            MAX(timestamp) as last_seen
                        FROM metrics 
                        WHERE timestamp >= NOW() - INTERVAL '%s hours'
                        GROUP BY metric_name, metric_type
                        ORDER BY metric_name
                    """, (hours,))
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"❌ Failed to get metrics summary: {e}")
            return []

    def cleanup_old_metrics(self, days: int = 30):
        """Clean up metrics older than N days."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        DELETE FROM metrics 
                        WHERE timestamp < NOW() - INTERVAL '%s days'
                    """, (days,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    logger.info(f"✅ Cleaned up {deleted_count} old metrics")
                    
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old metrics: {e}")


# Global instance
metrics_storage = PostgreSQLMetricsStorage() 