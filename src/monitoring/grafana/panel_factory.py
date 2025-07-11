"""
Grafana panel factory for creating specialized monitoring panels.

This module provides factory methods for creating common monitoring panels
used in MLOps dashboards.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GrafanaPanelFactory:
    """
    Factory for creating Grafana panels for ML monitoring.

    Provides methods to create commonly used panels for MLOps monitoring
    including drift detection, model performance, and data quality panels.
    """

    @staticmethod
    def create_stat_panel(
        panel_id: int,
        title: str,
        datasource: str,
        query: str,
        unit: str = "short",
        thresholds: Optional[List[Dict[str, Any]]] = None,
        x: int = 0, y: int = 0, w: int = 12, h: int = 8
    ) -> Dict[str, Any]:
        """
        Create a stat panel for displaying single values.

        Args:
            panel_id: Unique panel identifier
            title: Panel title
            datasource: Datasource name
            query: Query string
            unit: Value unit
            thresholds: Threshold configurations
            x, y, w, h: Panel position and dimensions

        Returns:
            Stat panel configuration
        """
        if thresholds is None:
            thresholds = [
                {"color": "green", "value": None},
                {"color": "red", "value": 80}
            ]

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
                        "align": "auto",
                        "cellOptions": {
                            "type": "auto"
                        },
                        "inspect": False
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds
                    },
                    "unit": unit
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
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "textMode": "auto"
            },
            "pluginVersion": "10.0.0",
            "targets": [
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": datasource
                    },
                    "expr": query,
                    "interval": "",
                    "legendFormat": "",
                    "refId": "A"
                }
            ],
            "title": title,
            "type": "stat"
        }

    @staticmethod
    def create_timeseries_panel(
        panel_id: int,
        title: str,
        datasource: str,
        targets: List[Dict[str, str]],
        unit: str = "short",
        fill_opacity: int = 10,
        x: int = 0, y: int = 0, w: int = 12, h: int = 8
    ) -> Dict[str, Any]:
        """
        Create a timeseries panel for showing trends over time.

        Args:
            panel_id: Unique panel identifier
            title: Panel title
            datasource: Datasource name
            targets: List of query targets with 'expr' and 'legendFormat' keys
            unit: Value unit
            fill_opacity: Fill opacity for the timeseries
            x, y, w, h: Panel position and dimensions

        Returns:
            Timeseries panel configuration
        """
        panel_targets = []
        for i, target in enumerate(targets):
            panel_targets.append({
                "datasource": {
                    "type": "prometheus",
                    "uid": datasource
                },
                "expr": target["expr"],
                "interval": "",
                "legendFormat": target.get("legendFormat", ""),
                "refId": chr(65 + i)  # A, B, C, etc.
            })

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
                        "fillOpacity": fill_opacity,
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
                    "unit": unit
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
            "targets": panel_targets,
            "title": title,
            "type": "timeseries"
        }

    @staticmethod
    def create_table_panel(
        panel_id: int,
        title: str,
        datasource: str,
        datasource_type: str,
        query: str,
        x: int = 0, y: int = 0, w: int = 12, h: int = 8
    ) -> Dict[str, Any]:
        """
        Create a table panel for tabular data display.

        Args:
            panel_id: Unique panel identifier
            title: Panel title
            datasource: Datasource name
            datasource_type: Type of datasource (prometheus, influxdb, etc.)
            query: Query string
            x, y, w, h: Panel position and dimensions

        Returns:
            Table panel configuration
        """
        return {
            "datasource": {
                "type": datasource_type,
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
                        "type": datasource_type,
                        "uid": datasource
                    },
                    "query" if datasource_type == "influxdb" else "expr": query,
                    "refId": "A"
                }
            ],
            "title": title,
            "type": "table"
        }

    @staticmethod
    def create_bargauge_panel(
        panel_id: int,
        title: str,
        datasource: str,
        query: str,
        unit: str = "short",
        max_value: Optional[float] = None,
        x: int = 0, y: int = 0, w: int = 12, h: int = 8
    ) -> Dict[str, Any]:
        """
        Create a bar gauge panel for displaying values as bars.

        Args:
            panel_id: Unique panel identifier
            title: Panel title
            datasource: Datasource name
            query: Query string
            unit: Value unit
            max_value: Maximum value for the gauge
            x, y, w, h: Panel position and dimensions

        Returns:
            Bar gauge panel configuration
        """
        field_config: Dict[str, Any] = {
            "defaults": {
                "color": {
                    "mode": "continuous-GrYlRd"
                },
                "custom": {
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "",
                    "axisPlacement": "auto",
                    "fillOpacity": 80,
                    "gradientMode": "none",
                    "hideFrom": {
                        "legend": False,
                        "tooltip": False,
                        "vis": False
                    },
                    "lineWidth": 1,
                    "scaleDistribution": {
                        "type": "linear"
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
                            "color": "yellow",
                            "value": 5
                        },
                        {
                            "color": "red",
                            "value": 10
                        }
                    ]
                },
                "unit": unit
            },
            "overrides": []
        }

        if max_value is not None:
            field_config["defaults"]["max"] = max_value

        return {
            "datasource": {
                "type": "prometheus",
                "uid": datasource
            },
            "fieldConfig": field_config,
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
                "showUnfilled": True,
                "valueMode": "color",
                "text": {}
            },
            "pluginVersion": "10.0.0",
            "targets": [
                {
                    "datasource": {
                        "type": "prometheus",
                        "uid": datasource
                    },
                    "expr": query,
                    "interval": "",
                    "legendFormat": "",
                    "refId": "A"
                }
            ],
            "title": title,
            "type": "bargauge"
        }

    @staticmethod
    def create_alertlist_panel(
        panel_id: int,
        title: str,
        datasource: str,
        x: int = 0, y: int = 0, w: int = 12, h: int = 8
    ) -> Dict[str, Any]:
        """
        Create an alert list panel for displaying active alerts.

        Args:
            panel_id: Unique panel identifier
            title: Panel title
            datasource: Datasource name
            x, y, w, h: Panel position and dimensions

        Returns:
            Alert list panel configuration
        """
        return {
            "datasource": {
                "type": "alertmanager",
                "uid": datasource
            },
            "fieldConfig": {
                "defaults": {},
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
                "showOptions": "current",
                "maxItems": 20,
                "sortOrder": 1,
                "dashboardAlerts": False,
                "alertName": "",
                "dashboardTitle": "",
                "folderId": None,
                "tags": []
            },
            "pluginVersion": "10.0.0",
            "title": title,
            "type": "alertlist"
        }

    @staticmethod
    def create_heatmap_panel(
        panel_id: int,
        title: str,
        datasource: str,
        datasource_type: str,
        query: str,
        x: int = 0, y: int = 0, w: int = 12, h: int = 8
    ) -> Dict[str, Any]:
        """
        Create a heatmap panel for correlation/distribution visualization.

        Args:
            panel_id: Unique panel identifier
            title: Panel title
            datasource: Datasource name
            datasource_type: Type of datasource (prometheus, influxdb, etc.)
            query: Query string
            x, y, w, h: Panel position and dimensions

        Returns:
            Heatmap panel configuration
        """
        return {
            "datasource": {
                "type": datasource_type,
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
                    "unit": "short"
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
                        "type": datasource_type,
                        "uid": datasource
                    },
                    "query" if datasource_type == "influxdb" else "expr": query,
                    "refId": "A"
                }
            ],
            "title": title,
            "type": "heatmap"
        }

    @staticmethod
    def create_ml_drift_overview_panel(
        panel_id: int,
        datasource: str,
        x: int = 0, y: int = 0, w: int = 24, h: int = 8
    ) -> Dict[str, Any]:
        """
        Create a specialized ML drift overview panel.

        Args:
            panel_id: Unique panel identifier
            datasource: Datasource name
            x, y, w, h: Panel position and dimensions

        Returns:
            ML drift overview panel configuration
        """
        return GrafanaPanelFactory.create_timeseries_panel(
            panel_id=panel_id,
            title="ML Data Drift Overview",
            datasource=datasource,
            targets=[
                {
                    "expr": "evidently_drift_share",
                    "legendFormat": "Drift Share (%)"
                },
                {
                    "expr": "evidently_drifted_features_count",
                    "legendFormat": "Drifted Features"
                },
                {
                    "expr": "evidently_dataset_drift",
                    "legendFormat": "Dataset Drift Alert"
                }
            ],
            unit="percentunit",
            x=x, y=y, w=w, h=h
        )

    @staticmethod
    def create_ml_quality_overview_panel(
        panel_id: int,
        datasource: str,
        x: int = 0, y: int = 0, w: int = 24, h: int = 8
    ) -> Dict[str, Any]:
        """
        Create a specialized ML data quality overview panel.

        Args:
            panel_id: Unique panel identifier
            datasource: Datasource name
            x, y, w, h: Panel position and dimensions

        Returns:
            ML data quality overview panel configuration
        """
        return GrafanaPanelFactory.create_timeseries_panel(
            panel_id=panel_id,
            title="ML Data Quality Overview",
            datasource=datasource,
            targets=[
                {
                    "expr": "evidently_missing_values_percentage",
                    "legendFormat": "Missing Values (%)"
                },
                {
                    "expr": "evidently_missing_values_count",
                    "legendFormat": "Missing Values Count"
                }
            ],
            unit="percent",
            x=x, y=y, w=w, h=h
        )

    @staticmethod
    def create_ml_performance_panel(
        panel_id: int,
        datasource: str,
        x: int = 0, y: int = 0, w: int = 24, h: int = 8
    ) -> Dict[str, Any]:
        """
        Create a specialized ML monitoring performance panel.

        Args:
            panel_id: Unique panel identifier
            datasource: Datasource name
            x, y, w, h: Panel position and dimensions

        Returns:
            ML monitoring performance panel configuration
        """
        return GrafanaPanelFactory.create_timeseries_panel(
            panel_id=panel_id,
            title="ML Monitoring Performance",
            datasource=datasource,
            targets=[
                {
                    "expr": "histogram_quantile(0.95, evidently_report_generation_seconds_bucket)",
                    "legendFormat": "95th Percentile (s)"
                },
                {
                    "expr": "histogram_quantile(0.50, evidently_report_generation_seconds_bucket)",
                    "legendFormat": "50th Percentile (s)"
                },
                {
                    "expr": "rate(evidently_export_errors_total[5m])",
                    "legendFormat": "Error Rate (/min)"
                }
            ],
            unit="s",
            x=x, y=y, w=w, h=h
        )
