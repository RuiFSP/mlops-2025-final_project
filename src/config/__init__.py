"""
Environment Configuration Utilities

This module provides utilities for loading and managing environment variables
across the MLOps project.
"""

import os
from pathlib import Path


def load_environment(env_file: str | None = None) -> bool:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file. If None, looks for .env in project root.

    Returns:
        bool: True if .env file was loaded successfully, False otherwise.
    """
    try:
        from dotenv import load_dotenv

        if env_file is None:
            # Find project root (look for pyproject.toml)
            current_path = Path(__file__).parent
            while current_path != current_path.parent:
                if (current_path / "pyproject.toml").exists():
                    env_path = current_path / ".env"
                    break
                current_path = current_path.parent
            else:
                # Fallback to current directory
                env_path = Path(".env")
        else:
            env_path = Path(env_file)

        if env_path.exists():
            load_dotenv(env_path)
            return True
        else:
            return False

    except ImportError:
        return False


def get_env_var(key: str, default: str | None = None, required: bool = False) -> str:
    """
    Get environment variable with proper error handling.

    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether the variable is required

    Returns:
        str: Environment variable value

    Raises:
        ValueError: If required variable is not found
    """
    value = os.environ.get(key, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' not found")

    return value or ""


class MLOpsConfig:
    """Configuration class for MLOps environment variables."""

    def __init__(self) -> None:
        """Initialize configuration by loading .env file."""
        self.env_loaded = load_environment()

    @property
    def prefect_api_url(self) -> str:
        """Get Prefect API URL."""
        return get_env_var("PREFECT_API_URL", "http://127.0.0.1:4200/api")

    @property
    def prefect_server_host(self) -> str:
        """Get Prefect server host."""
        return get_env_var("PREFECT_SERVER_HOST", "127.0.0.1")

    @property
    def prefect_server_port(self) -> int:
        """Get Prefect server port."""
        return int(get_env_var("PREFECT_SERVER_PORT", "4200"))

    @property
    def prefect_work_pool(self) -> str:
        """Get Prefect work pool name."""
        return get_env_var("PREFECT_WORK_POOL", "mlops-pool")

    @property
    def mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI."""
        return get_env_var("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    @property
    def mlflow_server_host(self) -> str:
        """Get MLflow server host."""
        return get_env_var("MLFLOW_SERVER_HOST", "127.0.0.1")

    @property
    def mlflow_server_port(self) -> int:
        """Get MLflow server port."""
        return int(get_env_var("MLFLOW_SERVER_PORT", "5000"))

    @property
    def api_host(self) -> str:
        """Get API host."""
        return get_env_var("API_HOST", "127.0.0.1")

    @property
    def api_port(self) -> int:
        """Get API port."""
        return int(get_env_var("API_PORT", "8000"))

    @property
    def model_path(self) -> str:
        """Get model path."""
        return get_env_var("MODEL_PATH", "models/model.pkl")

    @property
    def training_data_path(self) -> str:
        """Get training data path."""
        return get_env_var("TRAINING_DATA_PATH", "data/real_data/premier_league_matches.parquet")

    @property
    def backup_dir(self) -> str:
        """Get backup directory."""
        return get_env_var("BACKUP_DIR", "models/backups")

    @property
    def simulation_speed(self) -> int:
        """Get simulation speed (seconds between weeks)."""
        return int(get_env_var("SIMULATION_SPEED", "5"))

    @property
    def retraining_threshold(self) -> float:
        """Get retraining threshold."""
        return float(get_env_var("RETRAINING_THRESHOLD", "0.05"))

    @property
    def retraining_frequency(self) -> int:
        """Get retraining frequency (weeks)."""
        return int(get_env_var("RETRAINING_FREQUENCY", "5"))

    @property
    def log_level(self) -> str:
        """Get log level."""
        return get_env_var("LOG_LEVEL", "INFO")

    @property
    def log_file(self) -> str:
        """Get log file path."""
        return get_env_var("LOG_FILE", "logs/mlops.log")

    def setup_environment(self) -> None:
        """Set up environment variables for external tools."""
        # Set Prefect API URL
        os.environ["PREFECT_API_URL"] = self.prefect_api_url

        # Set MLflow tracking URI
        os.environ["MLFLOW_TRACKING_URI"] = self.mlflow_tracking_uri

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"""MLOpsConfig(
    env_loaded={self.env_loaded},
    prefect_api_url={self.prefect_api_url},
    mlflow_tracking_uri={self.mlflow_tracking_uri},
    work_pool={self.prefect_work_pool}
)"""


# Global configuration instance
config = MLOpsConfig()
