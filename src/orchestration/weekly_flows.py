"""
Enhanced Prefect Flows for Weekly Premier League Data Processing
Realistic MLOps workflows using weekly batches of historical Premier League data
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.logging import get_run_logger

from ..monitoring.metrics_storage import metrics_storage
from ..pipelines.prediction_pipeline import PredictionPipeline

# Import our batch processor
from .batch_processor import WeeklyBatchProcessor
from .tasks import analyze_model_drift, check_model_performance, send_alerts

logger = logging.getLogger(__name__)


@task(name="process_weekly_batch", tags=["batch", "prediction"])
def process_weekly_batch(batch_data: dict[str, Any]) -> dict[str, Any]:
    """
    Process a single weekly batch of Premier League matches

    Args:
        batch_data: Weekly batch data as dictionary

    Returns:
        Processing results
    """
    logger = get_run_logger()
    logger.info(f"🏈 Processing weekly batch {batch_data['week_id']} with {batch_data['total_matches']} matches")

    try:
        pipeline = PredictionPipeline()

        batch_results = {
            "week_id": batch_data["week_id"],
            "total_matches": batch_data["total_matches"],
            "processed_matches": 0,
            "predictions": [],
            "accuracy_metrics": {},
            "processing_timestamp": datetime.now().isoformat(),
        }

        correct_predictions = 0
        total_predictions = 0

        # Process each match in the batch
        for match in batch_data["matches"]:
            try:
                # Generate prediction for the match
                prediction_result = pipeline.predict_match(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    home_odds=match.get("home_odds", 2.0),
                    away_odds=match.get("away_odds", 3.0),
                    draw_odds=match.get("draw_odds", 3.5),
                )

                # Compare prediction with actual result
                predicted_outcome = prediction_result["prediction"]
                actual_outcome = match["result"]
                is_correct = predicted_outcome == actual_outcome

                if is_correct:
                    correct_predictions += 1
                total_predictions += 1

                # Store prediction result
                prediction_record = {
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "predicted": predicted_outcome,
                    "actual": actual_outcome,
                    "correct": is_correct,
                    "confidence": prediction_result["confidence"],
                    "match_date": match["date"],
                }
                batch_results["predictions"].append(prediction_record)

                # Store in metrics database
                prediction_data = {
                    "match_id": f"{match['home_team']}vs{match['away_team']}_{match['date']}",
                    "prediction": predicted_outcome,
                    "confidence": prediction_result["confidence"],
                    "actual_result": actual_outcome,
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "match_date": match["date"],
                }
                metrics_storage.store_prediction(prediction_data)

                batch_results["processed_matches"] += 1

            except Exception as e:
                logger.warning(f"Error processing match {match['home_team']} vs {match['away_team']}: {e}")
                continue

        # Calculate batch accuracy metrics
        if total_predictions > 0:
            batch_accuracy = correct_predictions / total_predictions
            batch_results["accuracy_metrics"] = {
                "accuracy": batch_accuracy,
                "correct_predictions": correct_predictions,
                "total_predictions": total_predictions,
            }

            # Store batch metrics
            metrics_storage.store_model_metric(
                model_name="premier_league_predictor",
                metrics={
                    "batch_accuracy": batch_accuracy,
                    "batch_week": batch_data["week_id"],
                    "batch_matches": total_predictions,
                },
            )

        logger.info(
            f"✅ Processed batch {batch_data['week_id']}: {correct_predictions}/{total_predictions} correct ({batch_accuracy:.2%})"
        )
        return batch_results

    except Exception as e:
        logger.error(f"❌ Error processing weekly batch: {e}")
        return {
            "week_id": batch_data["week_id"],
            "error": str(e),
            "processing_timestamp": datetime.now().isoformat(),
        }


@task(name="analyze_weekly_performance", tags=["monitoring", "weekly"])
def analyze_weekly_performance(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze performance across multiple weekly batches

    Args:
        batch_results: List of weekly batch processing results

    Returns:
        Performance analysis results
    """
    logger = get_run_logger()
    logger.info(f"📊 Analyzing performance across {len(batch_results)} weekly batches")

    try:
        total_matches = 0
        total_correct = 0
        weekly_accuracies = []
        all_predictions = []

        for batch in batch_results:
            if "accuracy_metrics" in batch and batch["accuracy_metrics"]:
                metrics = batch["accuracy_metrics"]
                total_matches += metrics["total_predictions"]
                total_correct += metrics["correct_predictions"]
                weekly_accuracies.append(metrics["accuracy"])

            if "predictions" in batch:
                all_predictions.extend(batch["predictions"])

        # Calculate overall performance
        overall_accuracy = total_correct / total_matches if total_matches > 0 else 0
        avg_weekly_accuracy = sum(weekly_accuracies) / len(weekly_accuracies) if weekly_accuracies else 0

        # Analyze prediction patterns
        outcome_analysis = {
            "H": {"predicted": 0, "correct": 0},
            "D": {"predicted": 0, "correct": 0},
            "A": {"predicted": 0, "correct": 0},
        }

        for pred in all_predictions:
            outcome = pred["predicted"]
            if outcome in outcome_analysis:
                outcome_analysis[outcome]["predicted"] += 1
                if pred["correct"]:
                    outcome_analysis[outcome]["correct"] += 1

        # Calculate accuracy by outcome
        for outcome in outcome_analysis:
            predicted_count = outcome_analysis[outcome]["predicted"]
            if predicted_count > 0:
                outcome_analysis[outcome]["accuracy"] = outcome_analysis[outcome]["correct"] / predicted_count
            else:
                outcome_analysis[outcome]["accuracy"] = 0

        performance_analysis = {
            "overall_accuracy": overall_accuracy,
            "avg_weekly_accuracy": avg_weekly_accuracy,
            "total_matches": total_matches,
            "total_correct": total_correct,
            "weekly_count": len(batch_results),
            "outcome_analysis": outcome_analysis,
            "weekly_accuracies": weekly_accuracies,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"📈 Weekly performance analysis complete: {overall_accuracy:.2%} overall accuracy across {total_matches} matches"
        )
        return performance_analysis

    except Exception as e:
        logger.error(f"❌ Error analyzing weekly performance: {e}")
        return {"error": str(e), "analysis_timestamp": datetime.now().isoformat()}


@flow(name="weekly_batch_processing_flow", tags=["weekly", "batch", "prediction"])
def weekly_batch_processing_flow(num_weeks: int = 4, simulate_live: bool = True) -> dict[str, Any]:
    """
    Process multiple weeks of Premier League data in batches

    Args:
        num_weeks: Number of recent weeks to process
        simulate_live: Whether to simulate live processing scenario

    Returns:
        Flow execution results
    """
    logger = get_run_logger()
    logger.info(f"🚀 Starting weekly batch processing flow for {num_weeks} weeks")

    try:
        # Initialize batch processor
        processor = WeeklyBatchProcessor()

        # Get weekly batches
        if simulate_live:
            simulation = processor.simulate_live_processing()
            batches = simulation["upcoming"]  # Process "upcoming" matches as if they're current
            logger.info(f"🔄 Live simulation: processing {len(batches)} upcoming weeks")
        else:
            batches = processor.get_recent_weeks(num_weeks)
            logger.info(f"📅 Processing {len(batches)} recent weeks")

        if not batches:
            logger.warning("No batches to process")
            return {"status": "no_data", "timestamp": datetime.now().isoformat()}

        # Process each weekly batch
        batch_results = []
        for batch in batches:
            # Convert batch to dictionary for Prefect task
            batch_dict = {
                "week_id": batch.week_id,
                "week_start": batch.week_start.isoformat(),
                "week_end": batch.week_end.isoformat(),
                "season": batch.season,
                "total_matches": batch.total_matches,
                "matches": batch.matches,
            }

            # Process the batch
            result = process_weekly_batch(batch_dict)
            batch_results.append(result)

        # Analyze overall performance
        performance_analysis = analyze_weekly_performance(batch_results)

        # Check if performance monitoring is needed
        overall_accuracy = performance_analysis.get("overall_accuracy", 0)
        if overall_accuracy < 0.55:  # Performance threshold
            send_alerts(
                alert_type="performance",
                message=f"Weekly batch processing accuracy below threshold: {overall_accuracy:.2%}",
                severity="warning",
                data=performance_analysis,
            )
            logger.warning(f"🚨 Performance alert sent: {overall_accuracy:.2%} accuracy")

        # Save results
        results_dir = Path("data/weekly_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        flow_results = {
            "flow_type": "weekly_batch_processing",
            "execution_timestamp": datetime.now().isoformat(),
            "batches_processed": len(batch_results),
            "performance_analysis": performance_analysis,
            "batch_results": batch_results,
            "simulation_mode": simulate_live,
        }

        results_file = results_dir / f"weekly_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(flow_results, f, indent=2, default=str)

        logger.info(
            f"✅ Weekly batch processing flow completed: {overall_accuracy:.2%} accuracy across {performance_analysis['total_matches']} matches"
        )
        return flow_results

    except Exception as e:
        logger.error(f"❌ Error in weekly batch processing flow: {e}")
        return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}


@flow(name="historical_replay_flow", tags=["historical", "simulation", "batch"])
def historical_replay_flow(start_weeks_back: int = 12, num_weeks: int = 8) -> dict[str, Any]:
    """
    Replay historical Premier League data to simulate continuous monitoring

    Args:
        start_weeks_back: How many weeks back to start the replay
        num_weeks: Number of weeks to replay

    Returns:
        Historical replay results
    """
    logger = get_run_logger()
    logger.info(f"⏮️ Starting historical replay: {num_weeks} weeks starting {start_weeks_back} weeks back")

    try:
        processor = WeeklyBatchProcessor()

        # Get historical date range
        latest_date = processor.df.Date.max()
        start_date = latest_date - timedelta(weeks=start_weeks_back)
        end_date = start_date + timedelta(weeks=num_weeks)

        # Get historical batches
        historical_batches = processor.get_weekly_batches(start_date=start_date, end_date=end_date, max_weeks=num_weeks)

        logger.info(
            f"📚 Replaying {len(historical_batches)} historical weeks from {start_date.date()} to {end_date.date()}"
        )

        # Process historical batches in sequence (simulating time progression)
        all_results = []
        weekly_performance = []

        for i, batch in enumerate(historical_batches):
            logger.info(f"🔄 Processing historical week {i + 1}/{len(historical_batches)}: {batch.week_id}")

            # Convert to dict and process
            batch_dict = {
                "week_id": batch.week_id,
                "week_start": batch.week_start.isoformat(),
                "week_end": batch.week_end.isoformat(),
                "season": batch.season,
                "total_matches": batch.total_matches,
                "matches": batch.matches,
            }

            result = process_weekly_batch(batch_dict)
            all_results.append(result)

            # Track performance over time
            if "accuracy_metrics" in result:
                weekly_performance.append(
                    {
                        "week": batch.week_id,
                        "accuracy": result["accuracy_metrics"]["accuracy"],
                        "matches": result["accuracy_metrics"]["total_predictions"],
                    }
                )

        # Analyze historical performance trends
        performance_analysis = analyze_weekly_performance(all_results)

        # Detect performance drift over time
        if len(weekly_performance) >= 4:
            recent_weeks = weekly_performance[-4:]
            early_weeks = weekly_performance[:4]

            recent_avg = sum(w["accuracy"] for w in recent_weeks) / len(recent_weeks)
            early_avg = sum(w["accuracy"] for w in early_weeks) / len(early_weeks)

            drift_score = abs(recent_avg - early_avg)
            if drift_score > 0.05:  # 5% drift threshold
                send_alerts(
                    alert_type="drift",
                    message=f"Historical replay detected performance drift: {drift_score:.2%} change",
                    severity="warning",
                    data={
                        "recent_avg": recent_avg,
                        "early_avg": early_avg,
                        "drift_score": drift_score,
                    },
                )

        replay_results = {
            "flow_type": "historical_replay",
            "execution_timestamp": datetime.now().isoformat(),
            "weeks_replayed": len(historical_batches),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "performance_analysis": performance_analysis,
            "weekly_performance": weekly_performance,
            "batch_results": all_results,
        }

        # Save historical replay results
        results_dir = Path("data/historical_replay")
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / f"historical_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(replay_results, f, indent=2, default=str)

        logger.info(f"✅ Historical replay completed: {performance_analysis['overall_accuracy']:.2%} average accuracy")
        return replay_results

    except Exception as e:
        logger.error(f"❌ Error in historical replay flow: {e}")
        return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}


@flow(name="comprehensive_weekly_monitoring_flow", tags=["monitoring", "comprehensive"])
def comprehensive_weekly_monitoring_flow() -> dict[str, Any]:
    """
    Comprehensive weekly monitoring combining batch processing, performance analysis, and drift detection

    Returns:
        Comprehensive monitoring results
    """
    logger = get_run_logger()
    logger.info("🎯 Starting comprehensive weekly monitoring flow")

    try:
        # Run weekly batch processing
        batch_results = weekly_batch_processing_flow(num_weeks=4, simulate_live=True)

        # Run performance monitoring
        performance_results = check_model_performance()

        # Run drift analysis
        drift_results = analyze_model_drift()

        # Combine all results
        monitoring_results = {
            "flow_type": "comprehensive_weekly_monitoring",
            "execution_timestamp": datetime.now().isoformat(),
            "batch_processing": batch_results,
            "performance_monitoring": performance_results,
            "drift_analysis": drift_results,
            "overall_status": "healthy",
        }

        # Determine overall system health
        batch_accuracy = batch_results.get("performance_analysis", {}).get("overall_accuracy", 0)
        model_accuracy = performance_results.get("current_accuracy", 0)
        drift_detected = drift_results.get("drift_detected", False)

        if batch_accuracy < 0.55 or model_accuracy < 0.55:
            monitoring_results["overall_status"] = "performance_warning"
        elif drift_detected:
            monitoring_results["overall_status"] = "drift_detected"

        # Send comprehensive alert if needed
        if monitoring_results["overall_status"] != "healthy":
            send_alerts(
                alert_type="comprehensive_monitoring",
                message=f"Weekly monitoring status: {monitoring_results['overall_status']}",
                severity="warning",
                data=monitoring_results,
            )

        logger.info(f"✅ Comprehensive weekly monitoring completed: {monitoring_results['overall_status']}")
        return monitoring_results

    except Exception as e:
        logger.error(f"❌ Error in comprehensive weekly monitoring: {e}")
        return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
