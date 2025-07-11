# Enhanced ML Monitoring with Evidently and Grafana

This guide covers the enhanced monitoring capabilities using Evidently for ML-specific monitoring and Grafana for visualization and dashboards.

## 🎯 Overview

The enhanced monitoring system provides:

- **Advanced ML Monitoring**: Data drift detection, model performance tracking, and data quality assessment using Evidently
- **Real-time Dashboards**: Interactive Grafana dashboards with live metrics visualization
- **Metrics Export**: Integration with Prometheus and InfluxDB for comprehensive observability
- **Automated Reporting**: Scheduled reports and alerts for stakeholder communication
- **Production-Ready**: Scalable architecture suitable for production MLOps workflows

## 📊 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ML Pipeline   │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Training Data │───▶│ • Model         │───▶│ • Evidently     │
│ • Production    │    │ • Predictions   │    │ • Drift Detection│
│ • Real-time     │    │ • Features      │    │ • Quality Checks│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │◀───│   Prometheus    │◀───│   Metrics       │
│                 │    │                 │    │                 │
│ • Dashboards    │    │ • Metrics       │    │ • Export        │
│ • Alerting      │    │ • Alerting      │    │ • Transform     │
│ • Visualization │    │ • Time Series   │    │ • Store         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   InfluxDB      │    │   Reports       │    │   Notifications │
│                 │    │                 │    │                 │
│ • Time Series   │    │ • HTML Reports  │    │ • Email Alerts  │
│ • Long-term     │    │ • PDF Export    │    │ • Slack/Teams   │
│ • Analytics     │    │ • Stakeholder   │    │ • Webhooks      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

The enhanced monitoring dependencies are already included in `pyproject.toml`:

```toml
evidently = "^0.4.0"
prometheus-client = "^0.17.0"
influxdb-client = "^1.36.0"
```

### 2. Basic Usage

```python
from src.monitoring.evidently import EvidentlyMLMonitor
from src.monitoring.grafana import GrafanaDashboardConfig

# Initialize monitoring
monitor = EvidentlyMLMonitor(
    reference_data=training_data,
    output_dir="evidently_reports",
    enable_data_drift=True,
    enable_target_drift=True,
    enable_data_quality=True,
    drift_threshold=0.1
)

# Generate monitoring report
report = monitor.generate_comprehensive_report(
    current_data=production_data,
    include_predictions=True
)

# Create Grafana dashboard
dashboard_config = GrafanaDashboardConfig(
    dashboard_title="ML Monitoring Dashboard",
    dashboard_tags=["mlops", "monitoring"]
)
dashboard = dashboard_config.create_ml_monitoring_dashboard()
```

### 3. Run Demo

```bash
# Run the comprehensive demo
python scripts/evidently_grafana_demo.py
```

## 🔧 Configuration

### Evidently Configuration

```python
# Basic monitoring setup
monitor = EvidentlyMLMonitor(
    reference_data=reference_data,
    column_mapping=column_mapping,  # Optional: custom column mapping
    output_dir="evidently_reports",
    enable_data_drift=True,
    enable_target_drift=True,
    enable_data_quality=True,
    drift_threshold=0.1
)

# Advanced configuration
monitor = EvidentlyMLMonitor(
    reference_data=reference_data,
    column_mapping=ColumnMapping(
        target="result",
        prediction="predicted_result",
        numerical_features=["home_odds", "draw_odds", "away_odds"],
        categorical_features=["home_team", "away_team"]
    ),
    output_dir="evidently_reports",
    enable_data_drift=True,
    enable_target_drift=True,
    enable_data_quality=True,
    drift_threshold=0.05  # More sensitive drift detection
)
```

### Metrics Export Configuration

```python
# Prometheus and InfluxDB export
metrics_exporter = EvidentlyMetricsExporter(
    prometheus_gateway="http://localhost:9091",
    prometheus_job="evidently_ml_monitoring",
    influxdb_url="http://localhost:8086",
    influxdb_token="your-influxdb-token",
    influxdb_org="mlops",
    influxdb_bucket="ml_monitoring",
    export_interval=60
)

# Export metrics with tags
export_results = metrics_exporter.export_metrics(
    reference_data=reference_data,
    current_data=current_data,
    tags={
        "environment": "production",
        "model": "premier_league_predictor",
        "version": "v1.2.0"
    }
)
```

### Grafana Configuration

```python
# Dashboard configuration
dashboard_config = GrafanaDashboardConfig(
    dashboard_title="MLOps Monitoring Dashboard",
    dashboard_tags=["mlops", "monitoring", "evidently"],
    refresh_interval="30s",
    time_range="1h"
)

# Create specialized dashboards
ml_dashboard = dashboard_config.create_ml_monitoring_dashboard(
    prometheus_datasource="prometheus",
    influxdb_datasource="influxdb"
)

# Export dashboard
dashboard_config.export_dashboard("dashboards/ml_monitoring.json")
```

## 📈 Key Features

### 1. Data Drift Detection

- **Statistical Tests**: Comprehensive statistical tests for numerical and categorical features
- **Drift Scoring**: Quantitative drift scores with customizable thresholds
- **Feature-level Analysis**: Individual feature drift analysis and reporting
- **Trend Analysis**: Historical drift trends and pattern detection

```python
# Generate drift report
drift_report = monitor.generate_comprehensive_report(
    current_data=current_data,
    include_predictions=True
)

# Check drift alerts
for alert in drift_report.get("alerts", []):
    if alert["type"] == "data_drift":
        print(f"Drift Alert: {alert['message']}")
```

### 2. Data Quality Monitoring

- **Missing Values**: Detection and tracking of missing data patterns
- **Data Type Validation**: Ensuring data types match expectations
- **Range Validation**: Checking if values fall within expected ranges
- **Distribution Analysis**: Comparing data distributions over time

```python
# Quality metrics
quality_metrics = drift_report.get("data_quality", {})
missing_pct = quality_metrics.get("missing_values_percentage", 0)

if missing_pct > 5:
    print(f"High missing values detected: {missing_pct:.1f}%")
```

### 3. Automated Reporting

- **Scheduled Reports**: Daily, weekly, and monthly automated reports
- **Stakeholder Notifications**: Email and Slack integration for alerts
- **Report Archival**: Automatic archival and retention management
- **Comparative Analysis**: Period-over-period comparison reports

```python
# Daily report generation
report_generator = EvidentlyReportGenerator(
    output_dir="evidently_reports",
    archive_dir="evidently_reports/archive",
    max_reports=50
)

daily_report = report_generator.generate_daily_report(
    reference_data=reference_data,
    current_data=current_data,
    report_date=datetime.now()
)
```

### 4. Real-time Dashboards

- **Live Metrics**: Real-time monitoring dashboards with auto-refresh
- **Interactive Visualizations**: Drill-down capabilities and filtering
- **Alert Integration**: Visual alerts and notification systems
- **Multi-datasource**: Support for Prometheus, InfluxDB, and other sources

## 🎨 Dashboard Examples

### 1. Main Monitoring Dashboard

The main dashboard provides an overview of:
- Data drift status and trends
- Data quality metrics
- Model performance indicators
- Alert summary and notifications

**Location**: `dashboards/mlops_monitoring_dashboard.json`

**Key Panels**:
- Dataset drift gauge
- Drift share trend over time
- Missing values percentage
- Feature drift heatmap
- Monitoring performance metrics

### 2. Drift Analysis Dashboard

Specialized dashboard for deep-dive drift analysis:
- Feature-level drift analysis
- Statistical test results
- Drift trend visualization
- Comparative analysis

**Location**: `dashboards/drift_analysis_dashboard.json`

**Key Panels**:
- Current drift statistics
- Drift trend over time
- Feature drift heatmap
- Drift alert history

### 3. Custom Dashboards

Create custom dashboards using the panel factory:

```python
from src.monitoring.grafana import GrafanaPanelFactory

# Create custom panels
factory = GrafanaPanelFactory()

# Drift overview panel
drift_panel = factory.create_ml_drift_overview_panel(
    panel_id=1,
    datasource="prometheus"
)

# Quality overview panel
quality_panel = factory.create_ml_quality_overview_panel(
    panel_id=2,
    datasource="prometheus"
)

# Performance panel
performance_panel = factory.create_ml_performance_panel(
    panel_id=3,
    datasource="prometheus"
)
```

## 📊 Metrics Reference

### Prometheus Metrics

The system exports the following metrics to Prometheus:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `evidently_dataset_drift` | Gauge | Dataset drift status (0/1) |
| `evidently_drift_share` | Gauge | Share of features showing drift (0.0-1.0) |
| `evidently_drifted_features_count` | Gauge | Number of features with drift |
| `evidently_missing_values_percentage` | Gauge | Percentage of missing values |
| `evidently_missing_values_count` | Gauge | Total count of missing values |
| `evidently_report_generation_seconds` | Histogram | Time to generate reports |
| `evidently_last_export_timestamp` | Gauge | Last successful export timestamp |
| `evidently_export_errors_total` | Counter | Total export errors by type |

### InfluxDB Measurements

Data is stored in InfluxDB with the following structure:

```
ml_data_drift
├── dataset_drift (int)
├── drift_share (float)
├── drifted_features_count (int)
└── total_features (int)

ml_data_quality
├── missing_values_percentage (float)
├── missing_values_count (int)
└── total_values (int)

ml_monitoring_performance
├── export_duration (float)
└── report_generation_time (float)
```

## 🔔 Alerting

### Grafana Alerts

Configure alerts in Grafana based on metrics:

```json
{
  "alert": {
    "name": "High Data Drift",
    "message": "Data drift detected in ML model",
    "frequency": "10s",
    "conditions": [
      {
        "query": {
          "queryType": "",
          "refId": "A"
        },
        "reducer": {
          "params": [],
          "type": "last"
        },
        "evaluator": {
          "params": [0.1],
          "type": "gt"
        }
      }
    ]
  }
}
```

### Automated Notifications

Set up notifications through various channels:

```python
# Email notifications
alert_config = {
    "email": {
        "to": ["ml-team@company.com"],
        "subject": "ML Monitoring Alert",
        "template": "drift_alert.html"
    },
    "slack": {
        "webhook": "https://hooks.slack.com/...",
        "channel": "#ml-alerts"
    }
}
```

## 🛠️ Setup Instructions

### 1. Infrastructure Setup

#### Prometheus Setup

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'evidently-ml-monitoring'
    static_configs:
      - targets: ['localhost:9091']
```

#### InfluxDB Setup

```bash
# Create bucket
influx bucket create -n ml_monitoring -o mlops

# Create token
influx auth create -o mlops --read-buckets --write-buckets
```

#### Grafana Setup

```bash
# Copy configuration
cp config/grafana/grafana.ini /etc/grafana/
cp config/grafana/datasources.yaml /etc/grafana/provisioning/datasources/

# Start Grafana
systemctl start grafana-server
```

### 2. Configuration Files

Update the configuration files with your specific settings:

- `config/grafana/grafana.ini`: Main Grafana configuration
- `config/grafana/datasources.yaml`: Data source configurations
- `dashboards/*.json`: Pre-built dashboard configurations

### 3. Integration with Existing Pipeline

```python
# In your existing ML pipeline
from src.monitoring.evidently import EvidentlyMLMonitor, EvidentlyMetricsExporter

# Initialize monitoring
monitor = EvidentlyMLMonitor(reference_data=train_data)
exporter = EvidentlyMetricsExporter(
    prometheus_gateway="http://localhost:9091",
    influxdb_url="http://localhost:8086"
)

# In your prediction pipeline
def process_batch(data):
    # Your existing logic
    predictions = model.predict(data)

    # Add monitoring
    report = monitor.generate_comprehensive_report(data)
    exporter.export_metrics(train_data, data)

    return predictions
```

## 🚀 Production Deployment

### 1. Environment Variables

```bash
# Set environment variables
export PROMETHEUS_GATEWAY=http://prometheus-pushgateway:9091
export INFLUXDB_URL=http://influxdb:8086
export INFLUXDB_TOKEN=your-production-token
export INFLUXDB_ORG=mlops
export INFLUXDB_BUCKET=ml_monitoring
export GRAFANA_URL=http://grafana:3000
```

### 2. Docker Deployment

```dockerfile
FROM python:3.10-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

CMD ["python", "-m", "src.monitoring.scheduler"]
```

### 3. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-monitoring
  template:
    metadata:
      labels:
        app: ml-monitoring
    spec:
      containers:
      - name: ml-monitoring
        image: mlops/monitoring:latest
        env:
        - name: PROMETHEUS_GATEWAY
          value: "http://prometheus-pushgateway:9091"
        - name: INFLUXDB_URL
          value: "http://influxdb:8086"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 🔍 Troubleshooting

### Common Issues

1. **Connection Errors**
   - Check service URLs and ports
   - Verify network connectivity
   - Ensure authentication credentials are correct

2. **Missing Data**
   - Check data source configurations
   - Verify query syntax
   - Ensure proper time ranges

3. **Performance Issues**
   - Monitor resource usage
   - Optimize query frequency
   - Consider data sampling for large datasets

### Debug Commands

```bash
# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=evidently_dataset_drift

# Check InfluxDB data
influx query 'from(bucket:"ml_monitoring") |> range(start: -1h)'

# Check Grafana datasources
curl -H "Authorization: Bearer <token>" http://localhost:3000/api/datasources
```

## 📚 Further Reading

- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [InfluxDB Documentation](https://docs.influxdata.com/)
- [MLOps Best Practices](https://ml-ops.org/)

## 🤝 Contributing

To contribute to the monitoring system:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with clear description

## 📄 License

This monitoring system is part of the MLOps project and follows the same license terms.
