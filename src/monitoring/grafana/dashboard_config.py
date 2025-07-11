"""
Grafana dashboard configuration generator for ML monitoring.

This module provides utilities to generate Grafana dashboard JSON configurations
for comprehensive MLOps monitoring visualization.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GrafanaDashboardConfig:
    """
    Generate Grafana dashboard configurations for ML monitoring.

    Provides utilities to create comprehensive MLOps monitoring dashboards
    with panels for data drift, model performance, and system health.
    """

    def __init__(
        self,
        dashboard_title: str = "MLOps Monitoring Dashboard",
        dashboard_tags: Optional[List[str]] = None,
        refresh_interval: str = "30s",
        time_range: str = "1h",
    ):
        """
        Initialize Grafana Dashboard Config.

        Args:
            dashboard_title: Title for the dashboard
            dashboard_tags: Tags to categorize the dashboard
            refresh_interval: Auto-refresh interval
            time_range: Default time range for panels
        """
        self.dashboard_title = dashboard_title
        self.dashboard_tags = dashboard_tags or ["mlops", "monitoring", "evidently"]
        self.refresh_interval = refresh_interval
        self.time_range = time_range

        # Dashboard configuration
        self.dashboard_config = self._create_base_dashboard()

        logger.info(f"Initialized Grafana Dashboard Config: {dashboard_title}")

    def _create_base_dashboard(self) -> Dict[str, Any]:
        """Create base dashboard configuration."""
        return {
            "dashboard": {
                "annotations": {
                    "list": [
                        {
                            "builtIn": 1,
                            "datasource": {
                                "type": "grafana",
                                "uid": "-- Grafana --"
                            },
                            "enable": True,
                            "hide": True,
                            "iconColor": "rgba(0, 211, 255, 1)",
                            "name": "Annotations & Alerts",
                            "type": "dashboard"
                        }
                    ]
                },
                "editable": True,
                "fiscalYearStartMonth": 0,
                "graphTooltip": 0,
                "id": None,
                "links": [],
                "liveNow": False,
                "panels": [],
                "refresh": self.refresh_interval,
                "schemaVersion": 38,
                "style": "dark",
                "tags": self.dashboard_tags,
                "templating": {
                    "list": []
                },
                "time": {
                    "from": f"now-{self.time_range}",
                    "to": "now"
                },
                "timepicker": {},
                "timezone": "",
                "title": self.dashboard_title,
                "uid": f"mlops-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "version": 1,
                "weekStart": ""
            }
        }

    def create_ml_monitoring_dashboard(
        self,
        prometheus_datasource: str = "prometheus",
        influxdb_datasource: str = "influxdb",
    ) -> Dict[str, Any]:
        """
        Create comprehensive ML monitoring dashboard.

        Args:
            prometheus_datasource: Name of Prometheus datasource
            influxdb_datasource: Name of InfluxDB datasource

        Returns:
            Complete dashboard configuration
        """
        dashboard = self.dashboard_config.copy()

        # Create panels
        panels = []
        panel_id = 1

        # Row 1: Data Drift Overview
        panels.append(self._create_row_panel("Data Drift Monitoring", panel_id))
        panel_id += 1

        # Data drift gauge
        panels.append(self._create_drift_gauge_panel(
            panel_id=panel_id,
            title="Dataset Drift Status",
            datasource=prometheus_datasource,
            x=0, y=1, w=6, h=8
        ))
        panel_id += 1

        # Drift share over time
        panels.append(self._create_drift_timeseries_panel(
            panel_id=panel_id,
            title="Drift Share Trend",
            datasource=prometheus_datasource,
            x=6, y=1, w=18, h=8
        ))
        panel_id += 1

        # Row 2: Data Quality
        panels.append(self._create_row_panel("Data Quality Monitoring", panel_id))
        panel_id += 1

        # Missing values percentage
        panels.append(self._create_quality_gauge_panel(
            panel_id=panel_id,
            title="Missing Values %",
            datasource=prometheus_datasource,
            x=0, y=10, w=6, h=8
        ))
        panel_id += 1

        # Missing values trend
        panels.append(self._create_quality_timeseries_panel(
            panel_id=panel_id,
            title="Data Quality Trend",
            datasource=prometheus_datasource,
            x=6, y=10, w=18, h=8
        ))
        panel_id += 1

        # Row 3: Model Performance
        panels.append(self._create_row_panel("Model Performance", panel_id))
        panel_id += 1

        # Feature drift heatmap
        panels.append(self._create_feature_drift_heatmap(
            panel_id=panel_id,
            title="Feature Drift Heatmap",
            datasource=influxdb_datasource,
            x=0, y=19, w=12, h=8
        ))
        panel_id += 1

        # Monitoring performance
        panels.append(self._create_monitoring_performance_panel(
            panel_id=panel_id,
            title="Monitoring Performance",
            datasource=prometheus_datasource,
            x=12, y=19, w=12, h=8
        ))
        panel_id += 1

        # Row 4: Alerts and Summary
        panels.append(self._create_row_panel("Alerts and Summary", panel_id))
        panel_id += 1

        # Alert summary table
        panels.append(self._create_alert_summary_table(
            panel_id=panel_id,
            title="Recent Alerts",
            datasource=influxdb_datasource,
            x=0, y=28, w=24, h=8
        ))
        panel_id += 1

        # Add panels to dashboard
        dashboard["dashboard"]["panels"] = panels

        return dashboard

    def _create_row_panel(self, title: str, panel_id: int) -> Dict[str, Any]:
        """Create a row panel for organizing dashboard sections."""
        return {
            "collapsed": False,
            "gridPos": {
                "h": 1,
                "w": 24,
                "x": 0,
                "y": 0
            },
            "id": panel_id,
            "panels": [],
            "title": title,
            "type": "row"
        }

    def _create_drift_gauge_panel(
        self,
        panel_id: int,
        title: str,
        datasource: str,
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Create drift status gauge panel."""
        return {
            "datasource": {
                "type": "prometheus",
                "uid": datasource
            },
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "thresholds"
                    },
                    "custom": {
                        "hideFrom": {
                            "tooltip": False,
                            "vis": False,
                            "legend": False
                        },
                        "arc": {"sections": [{"color": {"fixedColor": "red", "mode": "fixed"}, "endAngle": 90, "startAngle": 0}, {"color": {"fixedColor": "green", "mode": "fixed"}, "endAngle": 180, "startAngle": 90}]}
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 0.5
                            }
                        ]
                    },
                    "unit": "short"
                },
                "overrides": []
            },
            "gridPos": {
                "h": h,
                "w": w,
                "x": x,
                "y": y
            },
            "id": panel_id,
            "options": {
                "orientation": "auto",
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "showThresholdLabels": True,
                "showThresholdMarkers": True,
                "text": {}
            },
            "pluginVersion": "10.0.0",
            "targets": [
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": datasource
                    },
                    "expr": "evidently_dataset_drift",
                    "interval": "",
                    "legendFormat": "Dataset Drift",
                    "refId": "A"
                }
            ],
            "title": title,
            "type": "gauge"
        }

    def _create_drift_timeseries_panel(
        self,
        panel_id: int,
        title: str,
        datasource: str,
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Create drift share timeseries panel."""
        return {
            "datasource": {
                "type": "prometheus",
                "uid": datasource
            },
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisCenteredZero": False,
                        "axisColorMode": "text",
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "vis": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "auto",
                        "spanNulls": False,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 80
                            }
                        ]
                    },
                    "unit": "percentunit"
                },
                "overrides": []
            },
            "gridPos": {
                "h": h,
                "w": w,
                "x": x,
                "y": y
            },
            "id": panel_id,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": True
                },
                "tooltip": {
                    "mode": "single",
                    "sort": "none"
                }
            },
            "targets": [
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": datasource
                    },
                    "expr": "evidently_drift_share",
                    "interval": "",
                    "legendFormat": "Drift Share",
                    "refId": "A"
                },
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": datasource
                    },
                    "expr": "evidently_drifted_features_count",
                    "interval": "",
                    "legendFormat": "Drifted Features",
                    "refId": "B"
                }
            ],
            "title": title,
            "type": "timeseries"
        }

    def _create_quality_gauge_panel(
        self,
        panel_id: int,
        title: str,
        datasource: str,
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Create data quality gauge panel."""
        return {
            "datasource": {
                "type": "prometheus",
                "uid": datasource
            },
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "thresholds"
                    },
                    "custom": {
                        "hideFrom": {
                            "tooltip": False,
                            "vis": False,
                            "legend": False
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "yellow",
                                "value": 5
                            },
                            {
                                "color": "red",
                                "value": 10
                            }
                        ]
                    },
                    "unit": "percent"
                },
                "overrides": []
            },
            "gridPos": {
                "h": h,
                "w": w,
                "x": x,
                "y": y
            },
            "id": panel_id,
            "options": {
                "orientation": "auto",
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "showThresholdLabels": True,
                "showThresholdMarkers": True,
                "text": {}
            },
            "pluginVersion": "10.0.0",
            "targets": [
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": datasource
                    },
                    "expr": "evidently_missing_values_percentage",
                    "interval": "",
                    "legendFormat": "Missing Values %",
                    "refId": "A"
                }
            ],
            "title": title,
            "type": "gauge"
        }

    def _create_quality_timeseries_panel(
        self,
        panel_id: int,
        title: str,
        datasource: str,
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Create data quality timeseries panel."""
        return {
            "datasource": {
                "type": "prometheus",
                "uid": datasource
            },
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisCenteredZero": False,
                        "axisColorMode": "text",
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "vis": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "auto",
                        "spanNulls": False,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 80
                            }
                        ]
                    },
                    "unit": "percent"
                },
                "overrides": []
            },
            "gridPos": {
                "h": h,
                "w": w,
                "x": x,
                "y": y
            },
            "id": panel_id,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": True
                },
                "tooltip": {
                    "mode": "single",
                    "sort": "none"
                }
            },
            "targets": [
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": datasource
                    },
                    "expr": "evidently_missing_values_percentage",
                    "interval": "",
                    "legendFormat": "Missing Values %",
                    "refId": "A"
                }
            ],
            "title": title,
            "type": "timeseries"
        }

    def _create_feature_drift_heatmap(
        self,
        panel_id: int,
        title: str,
        datasource: str,
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Create feature drift heatmap panel."""
        return {
            "datasource": {
                "type": "influxdb",
                "uid": datasource
            },
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "continuous-GrYlRd"
                    },
                    "custom": {
                        "hideFrom": {
                            "tooltip": False,
                            "vis": False,
                            "legend": False
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 80
                            }
                        ]
                    }
                },
                "overrides": []
            },
            "gridPos": {
                "h": h,
                "w": w,
                "x": x,
                "y": y
            },
            "id": panel_id,
            "options": {
                "calculate": False,
                "cellGap": 1,
                "cellValues": {
                    "unit": "percentunit"
                },
                "color": {
                    "exponent": 0.5,
                    "fill": "dark-orange",
                    "mode": "spectrum",
                    "reverse": False,
                    "scale": "exponential",
                    "scheme": "Oranges",
                    "steps": 64
                },
                "exemplars": {
                    "color": "rgba(255,0,255,0.7)"
                },
                "filterValues": {
                    "le": 1e-9
                },
                "legend": {
                    "show": False
                },
                "rowsFrame": {
                    "layout": "auto"
                },
                "tooltip": {
                    "show": True,
                    "yHistogram": False
                },
                "yAxis": {
                    "axisPlacement": "left",
                    "reverse": False,
                    "unit": "short"
                }
            },
            "pluginVersion": "10.0.0",
            "targets": [
                {
                    "datasource": {
                        "type": "influxdb",
                        "uid": datasource
                    },
                    "query": "from(bucket: \"ml_monitoring\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ml_data_drift\")\n  |> filter(fn: (r) => r[\"_field\"] == \"drift_share\")\n  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)\n  |> yield(name: \"mean\")",
                    "refId": "A"
                }
            ],
            "title": title,
            "type": "heatmap"
        }

    def _create_monitoring_performance_panel(
        self,
        panel_id: int,
        title: str,
        datasource: str,
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Create monitoring performance panel."""
        return {
            "datasource": {
                "type": "prometheus",
                "uid": datasource
            },
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisCenteredZero": False,
                        "axisColorMode": "text",
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "vis": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "auto",
                        "spanNulls": False,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 80
                            }
                        ]
                    },
                    "unit": "s"
                },
                "overrides": []
            },
            "gridPos": {
                "h": h,
                "w": w,
                "x": x,
                "y": y
            },
            "id": panel_id,
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": True
                },
                "tooltip": {
                    "mode": "single",
                    "sort": "none"
                }
            },
            "targets": [
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": datasource
                    },
                    "expr": "histogram_quantile(0.95, evidently_report_generation_seconds_bucket)",
                    "interval": "",
                    "legendFormat": "95th Percentile",
                    "refId": "A"
                },
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": datasource
                    },
                    "expr": "histogram_quantile(0.50, evidently_report_generation_seconds_bucket)",
                    "interval": "",
                    "legendFormat": "50th Percentile",
                    "refId": "B"
                }
            ],
            "title": title,
            "type": "timeseries"
        }

    def _create_alert_summary_table(
        self,
        panel_id: int,
        title: str,
        datasource: str,
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Create alert summary table panel."""
        return {
            "datasource": {
                "type": "influxdb",
                "uid": datasource
            },
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "thresholds"
                    },
                    "custom": {
                        "align": "auto",
                        "cellOptions": {
                            "type": "auto"
                        },
                        "inspect": False
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 80
                            }
                        ]
                    }
                },
                "overrides": []
            },
            "gridPos": {
                "h": h,
                "w": w,
                "x": x,
                "y": y
            },
            "id": panel_id,
            "options": {
                "showHeader": True,
                "cellHeight": "sm",
                "footer": {
                    "show": False,
                    "reducer": ["sum"],
                    "countRows": False
                }
            },
            "pluginVersion": "10.0.0",
            "targets": [
                {
                    "datasource": {
                        "type": "influxdb",
                        "uid": datasource
                    },
                    "query": "from(bucket: \"ml_monitoring\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r[\"_measurement\"] == \"ml_data_drift\")\n  |> filter(fn: (r) => r[\"_field\"] == \"dataset_drift\")\n  |> filter(fn: (r) => r[\"_value\"] > 0)\n  |> sort(columns: [\"_time\"], desc: true)\n  |> limit(n: 10)",
                    "refId": "A"
                }
            ],
            "title": title,
            "type": "table"
        }

    def export_dashboard(self, output_path: str) -> bool:
        """
        Export dashboard configuration to JSON file.

        Args:
            output_path: Path to save the dashboard JSON

        Returns:
            True if successful, False otherwise
        """
        try:
            dashboard_path = Path(output_path)
            dashboard_path.parent.mkdir(parents=True, exist_ok=True)

            with open(dashboard_path, 'w') as f:
                json.dump(self.dashboard_config, f, indent=2)

            logger.info(f"Dashboard exported to: {dashboard_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting dashboard: {str(e)}")
            return False

    def get_dashboard_json(self) -> str:
        """Get dashboard configuration as JSON string."""
        return json.dumps(self.dashboard_config, indent=2)
