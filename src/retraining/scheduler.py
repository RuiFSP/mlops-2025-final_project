"""
Scheduler for automated retraining monitoring.
"""

import logging
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Any
import schedule

from dotenv import load_dotenv
load_dotenv()

from .retraining_monitor import RetrainingMonitor, RetrainingConfig

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """Scheduler for automated retraining monitoring."""
    
    def __init__(self, config: RetrainingConfig = None):
        """Initialize the scheduler."""
        self.config = config or RetrainingConfig()
        self.monitor = RetrainingMonitor(self.config)
        self.running = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ Retraining scheduler initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🔄 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run_monitoring_job(self):
        """Run a monitoring job."""
        logger.info("🔍 Running scheduled monitoring job...")
        
        try:
            result = self.monitor.run_monitoring_cycle()
            
            status = result.get('status', 'unknown')
            
            if status == 'retraining_triggered':
                logger.info("🚨 Retraining was triggered!")
                retraining_result = result.get('retraining_result', {})
                if retraining_result.get('success'):
                    logger.info(f"✅ New model accuracy: {retraining_result.get('new_accuracy', 0.0):.3f}")
                else:
                    logger.error(f"❌ Retraining failed: {retraining_result.get('error')}")
            
            elif status == 'performance_acceptable':
                metrics = result.get('metrics', {})
                logger.info(f"✅ Model performance is acceptable: {metrics.get('accuracy', 0.0):.3f}")
            
            elif status == 'no_data':
                logger.warning("⚠️ No data available for evaluation")
            
            else:
                logger.warning(f"⚠️ Unexpected monitoring status: {status}")
                
        except Exception as e:
            logger.error(f"❌ Monitoring job failed: {e}")
    
    def start(self, check_interval_hours: int = 24):
        """Start the scheduler."""
        logger.info(f"🚀 Starting retraining scheduler (check every {check_interval_hours} hours)")
        
        # Schedule monitoring job
        schedule.every(check_interval_hours).hours.do(self.run_monitoring_job)
        
        # Run initial monitoring job
        logger.info("🔄 Running initial monitoring job...")
        self.run_monitoring_job()
        
        # Start scheduling loop
        self.running = True
        logger.info("⏰ Scheduler started, waiting for next scheduled run...")
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        
        logger.info("🔄 Scheduler stopped")
    
    def run_immediate_check(self):
        """Run an immediate monitoring check."""
        logger.info("🔍 Running immediate monitoring check...")
        return self.run_monitoring_job()
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self.running,
            "config": {
                "min_accuracy_threshold": self.config.min_accuracy_threshold,
                "min_prediction_count": self.config.min_prediction_count,
                "evaluation_window_days": self.config.evaluation_window_days,
                "max_model_age_days": self.config.max_model_age_days,
                "performance_degradation_threshold": self.config.performance_degradation_threshold,
                "consecutive_poor_performance_limit": self.config.consecutive_poor_performance_limit
            },
            "next_run": schedule.next_run() if schedule.jobs else None,
            "monitoring_summary": self.monitor.get_monitoring_summary()
        }


def main():
    """Main function to run the scheduler."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start scheduler
    scheduler = RetrainingScheduler()
    
    try:
        # Start with daily checks
        scheduler.start(check_interval_hours=24)
    except KeyboardInterrupt:
        logger.info("🔄 Scheduler interrupted by user")
    except Exception as e:
        logger.error(f"❌ Scheduler failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 