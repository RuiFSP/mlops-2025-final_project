"""
Automation module for MLOps workflows.

This module provides automated retraining capabilities including:
- Automated retraining scheduler
- Prefect-based retraining flows
- Configuration management
- Notification systems
"""

from .retraining_scheduler import AutomatedRetrainingScheduler, RetrainingConfig

__all__ = [
    "AutomatedRetrainingScheduler",
    "RetrainingConfig",
]
