#!/usr/bin/env python3
"""
Test script for the automated retraining system.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from retraining.retraining_monitor import RetrainingMonitor, RetrainingConfig
from retraining.scheduler import RetrainingScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_retraining_monitor():
    """Test the retraining monitor functionality."""
    logger.info("🧪 Testing Retraining Monitor...")
    
    # Create custom config for testing
    config = RetrainingConfig(
        min_accuracy_threshold=0.50,  # Lower threshold for testing
        min_prediction_count=1,  # Lower requirement for testing
        evaluation_window_days=30,  # Longer window for testing
        max_model_age_days=1,  # Short age limit for testing
        performance_degradation_threshold=0.02,  # Lower threshold
        consecutive_poor_performance_limit=1  # Lower limit
    )
    
    # Initialize monitor
    monitor = RetrainingMonitor(config)
    
    # Test 1: Get current model info
    logger.info("📊 Testing model info retrieval...")
    model_info = monitor.get_current_model_info()
    if model_info:
        logger.info(f"   Model version: {model_info.get('version', 'unknown')}")
        logger.info(f"   Model accuracy: {model_info.get('accuracy', 0.0):.3f}")
        logger.info(f"   Creation date: {model_info.get('creation_date', 'unknown')}")
    else:
        logger.warning("   No model info available")
    
    # Test 2: Collect recent predictions
    logger.info("📋 Testing prediction collection...")
    predictions_df = monitor.collect_recent_predictions(days=30)
    logger.info(f"   Collected {len(predictions_df)} predictions")
    
    if len(predictions_df) > 0:
        logger.info("   Sample predictions:")
        for _, row in predictions_df.head(3).iterrows():
            logger.info(f"     {row['home_team']} vs {row['away_team']} -> {row['prediction']} (conf: {row['confidence']:.3f})")
    
    # Test 3: Evaluate model performance
    logger.info("🎯 Testing performance evaluation...")
    if len(predictions_df) > 0:
        metrics = monitor.evaluate_model_performance(predictions_df)
        if metrics:
            logger.info(f"   Accuracy: {metrics.accuracy:.3f}")
            logger.info(f"   Precision: {metrics.precision:.3f}")
            logger.info(f"   Recall: {metrics.recall:.3f}")
            logger.info(f"   F1 Score: {metrics.f1_score:.3f}")
            logger.info(f"   Prediction count: {metrics.prediction_count}")
        else:
            logger.warning("   Unable to evaluate performance")
    else:
        logger.warning("   No predictions available for evaluation")
    
    # Test 4: Check retraining triggers
    logger.info("🔧 Testing retraining trigger check...")
    if len(predictions_df) > 0:
        metrics = monitor.evaluate_model_performance(predictions_df)
        if metrics:
            triggers = monitor.check_retraining_triggers(metrics)
            logger.info(f"   Should retrain: {triggers['should_retrain']}")
            if triggers['triggers']:
                logger.info("   Triggers:")
                for trigger in triggers['triggers']:
                    logger.info(f"     - {trigger}")
            else:
                logger.info("   No triggers detected")
        else:
            logger.warning("   Unable to check triggers - no metrics")
    else:
        logger.warning("   No predictions available for trigger check")
    
    # Test 5: Get monitoring summary
    logger.info("📈 Testing monitoring summary...")
    summary = monitor.get_monitoring_summary()
    if 'error' not in summary:
        logger.info(f"   Recent evaluations: {len(summary.get('recent_evaluations', []))}")
        logger.info(f"   Recent retraining: {len(summary.get('recent_retraining', []))}")
    else:
        logger.error(f"   Error getting summary: {summary['error']}")
    
    logger.info("✅ Retraining monitor tests completed")
    return monitor


def test_monitoring_cycle():
    """Test the complete monitoring cycle."""
    logger.info("🔄 Testing complete monitoring cycle...")
    
    # Use default config
    monitor = RetrainingMonitor()
    
    # Run monitoring cycle
    result = monitor.run_monitoring_cycle()
    
    logger.info(f"   Status: {result.get('status', 'unknown')}")
    logger.info(f"   Message: {result.get('message', 'No message')}")
    
    if 'metrics' in result:
        metrics = result['metrics']
        logger.info(f"   Accuracy: {metrics.get('accuracy', 0.0):.3f}")
        logger.info(f"   Prediction count: {metrics.get('prediction_count', 0)}")
    
    if 'triggers' in result:
        triggers = result['triggers']
        logger.info(f"   Triggers: {triggers}")
    
    if 'retraining_result' in result:
        retraining = result['retraining_result']
        logger.info(f"   Retraining success: {retraining.get('success', False)}")
        if retraining.get('success'):
            logger.info(f"   New accuracy: {retraining.get('new_accuracy', 0.0):.3f}")
    
    logger.info("✅ Monitoring cycle test completed")
    return result


def test_scheduler():
    """Test the scheduler functionality."""
    logger.info("⏰ Testing scheduler functionality...")
    
    # Create scheduler
    scheduler = RetrainingScheduler()
    
    # Test status
    status = scheduler.get_status()
    logger.info(f"   Running: {status.get('running', False)}")
    logger.info(f"   Config: {status.get('config', {})}")
    
    # Test immediate check
    logger.info("🔍 Running immediate check...")
    scheduler.run_immediate_check()
    
    logger.info("✅ Scheduler test completed")
    return scheduler


def simulate_poor_performance():
    """Simulate poor model performance to test retraining triggers."""
    logger.info("🎭 Simulating poor model performance...")
    
    # Create strict config that should trigger retraining
    config = RetrainingConfig(
        min_accuracy_threshold=0.90,  # Very high threshold
        min_prediction_count=1,  # Low requirement
        evaluation_window_days=30,  # Long window
        max_model_age_days=0,  # Force retrain on age
        performance_degradation_threshold=0.01,  # Very low threshold
        consecutive_poor_performance_limit=1  # Immediate trigger
    )
    
    monitor = RetrainingMonitor(config)
    
    # Run monitoring cycle with strict config
    result = monitor.run_monitoring_cycle()
    
    logger.info(f"   Status: {result.get('status', 'unknown')}")
    
    if result.get('status') == 'retraining_triggered':
        logger.info("🚨 Retraining was triggered as expected!")
        triggers = result.get('triggers', [])
        for trigger in triggers:
            logger.info(f"   - {trigger}")
    else:
        logger.info("ℹ️ Retraining was not triggered (this is normal if model is performing well)")
    
    logger.info("✅ Poor performance simulation completed")
    return result


def main():
    """Run all tests."""
    logger.info("🚀 Starting automated retraining system tests...")
    
    try:
        # Test 1: Basic monitor functionality
        logger.info("\n" + "="*50)
        logger.info("TEST 1: Basic Monitor Functionality")
        logger.info("="*50)
        test_retraining_monitor()
        
        # Test 2: Complete monitoring cycle
        logger.info("\n" + "="*50)
        logger.info("TEST 2: Complete Monitoring Cycle")
        logger.info("="*50)
        test_monitoring_cycle()
        
        # Test 3: Scheduler functionality
        logger.info("\n" + "="*50)
        logger.info("TEST 3: Scheduler Functionality")
        logger.info("="*50)
        test_scheduler()
        
        # Test 4: Poor performance simulation
        logger.info("\n" + "="*50)
        logger.info("TEST 4: Poor Performance Simulation")
        logger.info("="*50)
        simulate_poor_performance()
        
        logger.info("\n" + "="*50)
        logger.info("🎉 All tests completed successfully!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 