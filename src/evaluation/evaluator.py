"""Model evaluation for Premier League match prediction."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles evaluation of Premier League match prediction models."""

    def __init__(self) -> None:
        """Initialize the model evaluator."""
        self.evaluation_results: dict[str, Any] = {}

    def evaluate(self, model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            model: Trained model or trainer instance
            test_data: Test dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting model evaluation")

        # Handle both trainer instance and raw model
        if hasattr(model, "predict") and hasattr(model, "prepare_features"):
            # Trainer instance
            predictions = model.predict(test_data)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(test_data)
                class_order = model.get_class_order()
            else:
                probabilities = None
                class_order = None
        elif hasattr(model, "predict"):
            # Direct model - need to prepare features first
            from src.model_training.trainer import ModelTrainer

            trainer = ModelTrainer()
            features = trainer.prepare_features(test_data)
            predictions = model.predict(features)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)
                class_order = model.classes_
            else:
                probabilities = None
                class_order = None
        else:
            logger.error("Model doesn't have predict method")
            return {}

        # Get true labels
        if "result" not in test_data.columns:
            logger.warning("No 'result' column found in test data")
            return {}

        y_true = test_data["result"]

        # Handle empty predictions
        if len(predictions) == 0:
            logger.warning("No predictions made")
            return {}

        # Calculate metrics
        metrics = self._calculate_metrics(y_true, predictions)

        # Calculate Brier score if probabilities are available
        if probabilities is not None and class_order is not None:
            brier_metrics = self._calculate_brier_score(
                y_true, probabilities, class_order
            )
            metrics.update(brier_metrics)

        # Compare with betting odds if available
        if probabilities is not None and self._has_odds_data(test_data):
            odds_comparison = self._compare_with_odds(
                y_true, probabilities, class_order, test_data
            )
            metrics.update(odds_comparison)

        # Log metrics to MLflow
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Generate evaluation report
        self._generate_evaluation_report(
            y_true, predictions, metrics, probabilities, class_order
        )

        logger.info(f"Evaluation completed. Accuracy: {metrics.get('accuracy', 0):.4f}")
        return metrics

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate various evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Class-specific metrics
        classes = ["H", "D", "A"]  # Home, Draw, Away
        for class_label in classes:
            if class_label in y_true.values:
                metrics[f"precision_{class_label}"] = precision_score(
                    y_true,
                    y_pred,
                    labels=[class_label],
                    average="macro",
                    zero_division=0,
                )
                metrics[f"recall_{class_label}"] = recall_score(
                    y_true,
                    y_pred,
                    labels=[class_label],
                    average="macro",
                    zero_division=0,
                )
                metrics[f"f1_{class_label}"] = f1_score(
                    y_true,
                    y_pred,
                    labels=[class_label],
                    average="macro",
                    zero_division=0,
                )

        return metrics

    def _generate_evaluation_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        metrics: Dict[str, float],
        probabilities: Optional[np.ndarray] = None,
        class_order: Optional[np.ndarray] = None,
    ) -> None:
        """Generate comprehensive evaluation report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metrics: Calculated metrics
            probabilities: Predicted probabilities (optional)
            class_order: Order of classes in probability array (optional)
        """
        # Create evaluation directory
        eval_dir = Path("evaluation_reports")
        eval_dir.mkdir(exist_ok=True)

        # Generate classification report
        report = classification_report(y_true, y_pred)

        # Save text report
        report_path = eval_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write("Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("Overall Metrics:\n")
            f.write("-" * 20 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
            f.write("Classification Report:\n")
            f.write("-" * 20 + "\n")
            # Ensure report is a string
            report_str = str(report) if not isinstance(report, str) else report
            f.write(report_str)

        logger.info(f"Evaluation report saved to {report_path}")

        # Generate confusion matrix plot
        self._plot_confusion_matrix(y_true, y_pred, eval_dir)

        # Generate metrics visualization
        self._plot_metrics(metrics, eval_dir)

    def _plot_confusion_matrix(
        self, y_true: pd.Series, y_pred: np.ndarray, save_dir: Path
    ) -> None:
        """Plot and save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_dir: Directory to save plots
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Away", "Draw", "Home"],
                yticklabels=["Away", "Draw", "Home"],
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            # Save plot
            plot_path = save_dir / "confusion_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Log to MLflow
            mlflow.log_artifact(str(plot_path))

            logger.info(f"Confusion matrix saved to {plot_path}")

        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")

    def _plot_metrics(self, metrics: Dict[str, float], save_dir: Path) -> None:
        """Plot evaluation metrics.

        Args:
            metrics: Dictionary of metrics
            save_dir: Directory to save plots
        """
        try:
            # Plot main metrics
            main_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            values = [metrics.get(metric, 0) for metric in main_metrics]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                main_metrics,
                values,
                color=["skyblue", "lightgreen", "lightcoral", "lightyellow"],
            )
            plt.title("Model Performance Metrics")
            plt.ylabel("Score")
            plt.ylim(0, 1)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plot_path = save_dir / "metrics_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Log to MLflow
            mlflow.log_artifact(str(plot_path))

            logger.info(f"Metrics plot saved to {plot_path}")

        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")

    def compare_models(
        self, model_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Compare multiple models' performance.

        Args:
            model_results: Dictionary mapping model names to their metrics

        Returns:
            DataFrame with comparison results
        """
        if not model_results:
            logger.warning("No model results provided for comparison")
            return pd.DataFrame()

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(model_results).T
        comparison_df = comparison_df.round(4)

        # Sort by accuracy
        if "accuracy" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("accuracy", ascending=False)

        # Save comparison
        comparison_path = Path("evaluation_reports/model_comparison.csv")
        comparison_path.parent.mkdir(exist_ok=True)
        comparison_df.to_csv(comparison_path)

        logger.info(f"Model comparison saved to {comparison_path}")

        return comparison_df

    def evaluate_prediction_confidence(
        self, model: Any, test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate prediction confidence and uncertainty.

        Args:
            model: Trained model
            test_data: Test dataset

        Returns:
            Dictionary containing confidence metrics
        """
        logger.info("Evaluating prediction confidence")

        try:
            # Get prediction probabilities if available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(test_data)
                max_probabilities = np.max(probabilities, axis=1)

                confidence_metrics = {
                    "mean_confidence": np.mean(max_probabilities),
                    "std_confidence": np.std(max_probabilities),
                    "min_confidence": np.min(max_probabilities),
                    "max_confidence": np.max(max_probabilities),
                }

                # Log confidence metrics
                for metric_name, value in confidence_metrics.items():
                    mlflow.log_metric(metric_name, value)

                return confidence_metrics

            else:
                logger.warning("Model does not support probability prediction")
                return {}

        except Exception as e:
            logger.error(f"Error evaluating prediction confidence: {e}")
            return {}

    def _calculate_brier_score(
        self, y_true: pd.Series, y_proba: np.ndarray, class_order: np.ndarray
    ) -> Dict[str, float]:
        """Calculate Brier score for probability predictions.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_order: Order of classes in probability array

        Returns:
            Dictionary with Brier score metrics
        """
        metrics = {}

        # Convert class order to ensure we have the right mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(class_order)}

        # Calculate Brier score for each class
        for i, class_name in enumerate(class_order):
            # Create binary indicator for this class
            y_binary = (y_true == class_name).astype(int)

            # Get probabilities for this class
            y_prob_class = y_proba[:, i]

            # Calculate Brier score (lower is better)
            brier_score = brier_score_loss(y_binary, y_prob_class)
            metrics[f"brier_score_{class_name}"] = brier_score

        # Calculate overall Brier score (average)
        metrics["brier_score_avg"] = np.mean(
            [metrics[f"brier_score_{cls}"] for cls in class_order]
        )

        return metrics

    def _has_odds_data(self, test_data: pd.DataFrame) -> bool:
        """Check if test data contains odds information.

        Args:
            test_data: Test dataset

        Returns:
            True if odds data is available
        """
        required_odds_cols = ["home_odds", "draw_odds", "away_odds"]
        return all(col in test_data.columns for col in required_odds_cols)

    def _compare_with_odds(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        class_order: np.ndarray,
        test_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compare model predictions with betting odds.

        Args:
            y_true: True labels
            y_proba: Model predicted probabilities
            class_order: Order of classes in probability array
            test_data: Test dataset with odds

        Returns:
            Dictionary with comparison metrics
        """
        metrics = {}

        # Remove margin from betting odds to get true probabilities
        odds_cols = ["home_odds", "draw_odds", "away_odds"]

        # Calculate implied probabilities
        implied_probs = {}
        for col in odds_cols:
            implied_probs[col.replace("_odds", "_prob")] = 1 / test_data[col]

        # Calculate total probability (includes margin)
        total_prob = sum(implied_probs.values())

        # Remove margin by normalizing
        for key in implied_probs:
            implied_probs[key] = implied_probs[key] / total_prob

        # Create odds probability array in same order as model classes
        odds_proba = np.zeros_like(y_proba)
        class_to_odds = {"H": "home_prob", "D": "draw_prob", "A": "away_prob"}

        for i, class_name in enumerate(class_order):
            if class_name in class_to_odds:
                odds_key = class_to_odds[class_name]
                odds_proba[:, i] = implied_probs[odds_key]

        # Calculate Brier score for odds
        odds_brier_scores = {}
        for i, class_name in enumerate(class_order):
            y_binary = (y_true == class_name).astype(int)
            odds_brier = brier_score_loss(y_binary, odds_proba[:, i])
            odds_brier_scores[f"odds_brier_{class_name}"] = odds_brier

        metrics.update(odds_brier_scores)
        metrics["odds_brier_avg"] = np.mean(list(odds_brier_scores.values()))

        # Calculate improvement over odds
        model_brier_avg = np.mean(
            [
                brier_score_loss((y_true == cls).astype(int), y_proba[:, i])
                for i, cls in enumerate(class_order)
            ]
        )

        metrics["brier_improvement"] = metrics["odds_brier_avg"] - model_brier_avg
        metrics["brier_improvement_pct"] = (
            metrics["brier_improvement"] / metrics["odds_brier_avg"]
        ) * 100

        return metrics
