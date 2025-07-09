# Automated Retraining System

This document describes the automated retraining system implemented for the Premier League Match Predictor. This enterprise-grade system monitors model performance and automatically triggers retraining when conditions warrant it.

## 🎯 Overview

The automated retraining system provides:

- **Intelligent Monitoring**: Continuous monitoring of model performance, data drift, and prediction volume
- **Multiple Trigger Types**: Performance degradation, data drift, time-based, and volume-based triggers
- **Safe Deployment**: Validation gates and approval workflows before model deployment
- **Full Automation**: Hands-off operation with comprehensive logging and notifications
- **Production Ready**: Robust error handling, concurrent operation prevention, and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Trigger        │    │  Retraining     │    │  Validation &   │
│  Detection      │───▶│  Execution      │───▶│  Deployment     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • Performance   │    │ • Data Prep     │    │ • A/B Testing   │
│ • Drift         │    │ • Training      │    │ • Rollback      │
│ • Time-based    │    │ • Validation    │    │ • Monitoring    │
│ • Volume        │    │ • Backup        │    │ • Reporting     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Configuration

Create or modify `config/retraining_config.yaml`:

```yaml
# Performance thresholds
performance_threshold: 0.05  # Trigger if accuracy drops by 5%
drift_threshold: 0.1         # Trigger if drift score exceeds 10%

# Time-based triggers
max_days_without_retraining: 30  # Maximum days without retraining
min_days_between_retraining: 7   # Minimum days between attempts

# Data volume triggers
min_new_predictions: 100
max_predictions_without_retraining: 1000

# Validation requirements
min_validation_accuracy: 0.45     # Don't deploy if worse than 45%
validation_improvement_required: 0.01  # Must improve by 1%

# Scheduler settings
check_interval_minutes: 60        # Check triggers every hour
enable_automatic_deployment: false  # Safety: manual approval by default
```

### 2. Start the Scheduler

```bash
# Using the management script
python scripts/automation/manage_retraining.py start

# Or programmatically
python -c "
from src.automation import AutomatedRetrainingScheduler
scheduler = AutomatedRetrainingScheduler()
scheduler.start_scheduler()
"
```

### 3. Monitor via API

```bash
# Check status
curl http://localhost:8000/retraining/status

# Trigger manual retraining
curl -X POST http://localhost:8000/retraining/trigger \
  -H "Content-Type: application/json" \
  -d '{"reason": "manual_test", "force": true}'

# View history
curl http://localhost:8000/retraining/history
```

## 📋 Components

### AutomatedRetrainingScheduler

The core component that orchestrates the entire retraining process.

**Key Features:**
- Background thread monitoring
- Multiple trigger condition evaluation
- Concurrent operation prevention
- Comprehensive status reporting
- Configuration hot-reloading

**Usage:**
```python
from src.automation import AutomatedRetrainingScheduler, RetrainingConfig

# Initialize with configuration
config = RetrainingConfig(
    performance_threshold=0.05,
    check_interval_minutes=60,
)
scheduler = AutomatedRetrainingScheduler(config=config)

# Start monitoring
scheduler.start_scheduler()

# Record predictions for volume tracking
scheduler.record_prediction({
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "prediction": "H"
})

# Force manual retraining
scheduler.force_retraining("performance_concern")
```

### Retraining Flow (Prefect)

Prefect-based workflow that handles the actual retraining process.

**Flow Steps:**
1. **Backup Current Model**: Creates timestamped backup
2. **Prepare Data**: Combines historical and new data
3. **Train New Model**: Trains with latest hyperparameters
4. **Validate Performance**: Compares against current model
5. **Deploy if Improved**: Atomic model replacement
6. **Generate Report**: Comprehensive retraining report

**Usage:**
```python
from src.automation.retraining_flow import automated_retraining_flow

# Execute retraining flow
result = automated_retraining_flow(
    triggers=["performance_degradation"],
    model_path="models/model.pkl",
    min_accuracy_threshold=0.45,
    improvement_threshold=0.02,
)

print(f"Retraining success: {result['success']}")
print(f"Model deployed: {result['deployed']}")
```

### API Endpoints

RESTful API for managing retraining operations.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/retraining/status` | GET | Get current scheduler status |
| `/retraining/start` | POST | Start the automated scheduler |
| `/retraining/stop` | POST | Stop the automated scheduler |
| `/retraining/trigger` | POST | Manually trigger retraining |
| `/retraining/history` | GET | Get retraining event history |
| `/retraining/config` | GET/POST | Manage configuration |
| `/retraining/export` | POST | Export detailed status report |

## 🔧 Configuration Reference

### RetrainingConfig

Complete configuration options:

```python
@dataclass
class RetrainingConfig:
    # Performance thresholds
    performance_threshold: float = 0.05
    drift_threshold: float = 0.1

    # Time-based triggers
    max_days_without_retraining: int = 30
    min_days_between_retraining: int = 7

    # Data volume triggers
    min_new_predictions: int = 100
    max_predictions_without_retraining: int = 1000

    # Validation requirements
    min_validation_accuracy: float = 0.45
    validation_improvement_required: float = 0.01

    # File paths
    model_path: str = "models/model.pkl"
    backup_model_dir: str = "models/backups"
    training_data_path: str = "data/real_data/premier_league_matches.parquet"
    monitoring_output_dir: str = "evaluation_reports"

    # Scheduler settings
    check_interval_minutes: int = 60
    enable_automatic_deployment: bool = False
    max_concurrent_retraining: int = 1
```

## 🎮 Management Commands

### manage_retraining.py

Production management script with comprehensive commands:

```bash
# Start scheduler daemon
python scripts/automation/manage_retraining.py start

# Check current status
python scripts/automation/manage_retraining.py status

# Manually trigger retraining
python scripts/automation/manage_retraining.py trigger --reason "performance_drop"

# Export detailed report
python scripts/automation/manage_retraining.py export --output reports/retraining_analysis.json

# Validate configuration
python scripts/automation/manage_retraining.py validate

# Create default configuration file
python scripts/automation/manage_retraining.py create-config
```

### demo_automated_retraining.py

Interactive demo script showcasing all features:

```bash
# Run complete demo
python scripts/automation/demo_automated_retraining.py --demo all

# Run specific demos
python scripts/automation/demo_automated_retraining.py --demo manual
python scripts/automation/demo_automated_retraining.py --demo lifecycle
python scripts/automation/demo_automated_retraining.py --demo config

# Enable verbose logging
python scripts/automation/demo_automated_retraining.py --demo all --verbose
```

## 📊 Monitoring & Observability

### Trigger Conditions

The system monitors multiple conditions:

1. **Performance Degradation**
   - Tracks accuracy over time
   - Compares against baseline
   - Configurable threshold

2. **Data Drift Detection**
   - Statistical distribution changes
   - Feature drift monitoring
   - Integration with Evidently AI

3. **Time-Based Triggers**
   - Maximum age without retraining
   - Regular maintenance schedules
   - Configurable intervals

4. **Volume-Based Triggers**
   - Prediction count tracking
   - Accumulated new data
   - Automatic threshold management

### Status Reporting

Comprehensive status information available:

```python
status = scheduler.get_status()
# Returns:
{
    "is_running": True,
    "retraining_in_progress": False,
    "last_check_time": "2025-07-09T15:30:00",
    "last_retraining_time": "2025-07-08T10:15:00",
    "prediction_count_since_retraining": 234,
    "days_since_last_retraining": 1,
    "total_trigger_events": 3,
    "config": {...}
}
```

### History Tracking

All events are logged with full context:

```python
# Trigger events
trigger_history = scheduler.get_trigger_history()
# Returns list of trigger events with timestamps, reasons, and context

# Retraining events
retraining_history = scheduler.retraining_orchestrator.get_retraining_history()
# Returns detailed retraining execution results
```

## 🔒 Safety Features

### Validation Gates

Multiple validation steps prevent bad model deployment:

- **Minimum Accuracy Threshold**: Models below threshold are rejected
- **Improvement Requirement**: New models must improve performance
- **A/B Testing Support**: Side-by-side comparison capabilities
- **Rollback Mechanisms**: Quick reversion to previous model

### Concurrency Protection

- **Single Operation Lock**: Prevents concurrent retraining
- **Resource Management**: Memory and compute limits
- **Graceful Shutdown**: Clean termination handling

### Error Handling

- **Comprehensive Logging**: All operations logged with context
- **Notification System**: Configurable alerts for failures
- **Automatic Recovery**: Resilient to transient failures
- **State Consistency**: Atomic operations prevent corruption

## 🧪 Testing

### Unit Tests

```bash
# Run scheduler tests
python -m pytest tests/unit/test_retraining_scheduler.py -v

# Run with coverage
python -m pytest tests/unit/test_retraining_scheduler.py --cov=src.automation
```

### Integration Tests

```bash
# Run full integration test suite
python -m pytest tests/integration/test_automated_retraining_integration.py -v

# Run specific test categories
python -m pytest tests/integration/test_automated_retraining_integration.py::TestAutomatedRetrainingIntegration -v
```

### Manual Testing

```bash
# Interactive demo mode
python scripts/automation/demo_automated_retraining.py --demo all --verbose

# API testing
python scripts/automation/test_api_endpoints.py
```

## 📈 Performance Considerations

### Resource Usage

- **Memory**: ~50MB for scheduler process
- **CPU**: Minimal during monitoring, intensive during retraining
- **Storage**: Model backups and logs require adequate space
- **Network**: API calls and data transfer during retraining

### Scalability

- **Horizontal**: Multiple schedulers for different models
- **Vertical**: Configurable resource limits per retraining
- **Cloud**: Compatible with container orchestration

### Optimization

- **Incremental Training**: Support for partial model updates
- **Caching**: Intelligent data and model caching
- **Parallel Processing**: Multi-threaded validation and evaluation

## 🚨 Troubleshooting

### Common Issues

1. **Scheduler Not Starting**
   ```bash
   # Check configuration
   python scripts/automation/manage_retraining.py validate

   # Check dependencies
   pip install -e .
   ```

2. **Retraining Failures**
   ```bash
   # Check logs
   tail -f logs/mlops_demo.log

   # Verify data paths
   ls -la data/real_data/premier_league_matches.parquet
   ```

3. **API Errors**
   ```bash
   # Check if scheduler is initialized
   curl http://localhost:8000/retraining/status

   # Restart API server
   python -m src.deployment.api --reload
   ```

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```python
import logging
logging.getLogger('src.automation').setLevel(logging.DEBUG)

# Or via environment
export PYTHONPATH="src:$PYTHONPATH"
export LOG_LEVEL=DEBUG
```

## 🔮 Future Enhancements

### Planned Features

- **Advanced Triggers**: Custom trigger conditions via plugins
- **Multi-Model Support**: Manage multiple models simultaneously
- **Cloud Integration**: Native AWS/GCP/Azure deployment
- **Advanced Validation**: Canary deployments and gradual rollouts
- **ML Pipeline Integration**: Integration with Kubeflow/Airflow
- **Real-time Monitoring**: Streaming performance metrics

### Extension Points

The system is designed for extensibility:

- **Custom Triggers**: Implement `TriggerInterface` for new conditions
- **Notification Backends**: Slack, email, webhook integrations
- **Storage Backends**: S3, GCS, Azure Blob support
- **Monitoring Integration**: Prometheus, Grafana, DataDog

## 📝 License

This automated retraining system is part of the MLOps 2025 Final Project and follows the same licensing terms as the main project.

## 🤝 Contributing

Contributions welcome! Please see the main project README for contribution guidelines.

Key areas for contribution:
- Additional trigger types
- Enhanced validation methods
- Cloud provider integrations
- Performance optimizations
- Documentation improvements
