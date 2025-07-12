#!/usr/bin/env python3
"""
Script to generate metrics and store them directly in PostgreSQL.
This replaces the Prometheus/Pushgateway approach.
"""

import sys
import os
sys.path.append('/app')

import time
import random
import logging
from datetime import datetime, timedelta
import requests

from src.monitoring.metrics_storage import metrics_storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Premier League teams
TEAMS = [
    "Arsenal", "Chelsea", "Manchester United", "Liverpool", "Manchester City",
    "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham",
    "Brentford", "Fulham", "Crystal Palace", "Wolves", "Leeds",
    "Everton", "Nottingham Forest", "Leicester", "Southampton", "Bournemouth"
]

def generate_prediction_metrics():
    """Generate prediction-related metrics."""
    logger.info("🎯 Generating prediction metrics...")
    
    # Generate prediction data
    home_team = random.choice(TEAMS)
    away_team = random.choice([t for t in TEAMS if t != home_team])
    home_odds = round(random.uniform(1.5, 4.0), 2)
    draw_odds = round(random.uniform(2.5, 4.5), 2)
    away_odds = round(random.uniform(1.8, 5.0), 2)
    match_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
    
    prediction_data = {
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "date": match_date
    }
    
    try:
        # Make API call
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction', 'Unknown')
            confidence = result.get('confidence', 0.5)
            
            logger.info(f"✅ Prediction: {home_team} vs {away_team} -> {prediction} (confidence: {confidence:.3f})")
            
            # Store metrics in PostgreSQL
            metrics_storage.store_counter("api_predictions_total", 1)
            metrics_storage.store_gauge("api_response_time_seconds", end_time - start_time)
            metrics_storage.store_gauge("prediction_confidence", confidence)
            metrics_storage.store_gauge("prediction_accuracy", random.uniform(0.6, 0.9))
            
            # Team-specific metrics
            metrics_storage.store_counter("team_predictions_total", 1, {"team": home_team})
            metrics_storage.store_counter("team_predictions_total", 1, {"team": away_team})
            
            # Prediction type metrics
            prediction_type = "home_win" if prediction == "H" else "away_win" if prediction == "A" else "draw"
            metrics_storage.store_counter("prediction_type_total", 1, {"type": prediction_type})
            
        else:
            logger.warning(f"⚠️ API request failed: {response.status_code}")
            metrics_storage.store_counter("api_errors_total", 1)
            
    except Exception as e:
        logger.error(f"❌ Error making prediction: {e}")
        metrics_storage.store_counter("api_errors_total", 1)

def generate_system_metrics():
    """Generate system and performance metrics."""
    logger.info("⚙️ Generating system metrics...")
    
    # Simulate system metrics
    cpu_usage = random.uniform(20, 80)
    memory_usage = random.uniform(30, 70)
    disk_usage = random.uniform(40, 90)
    network_throughput = random.uniform(100, 1000)
    
    metrics_storage.store_gauge("system_cpu_usage_percent", cpu_usage)
    metrics_storage.store_gauge("system_memory_usage_percent", memory_usage)
    metrics_storage.store_gauge("system_disk_usage_percent", disk_usage)
    metrics_storage.store_gauge("system_network_throughput_mbps", network_throughput)

def generate_model_metrics():
    """Generate model performance metrics."""
    logger.info("🤖 Generating model metrics...")
    
    # Simulate model performance metrics
    model_accuracy = random.uniform(0.65, 0.85)
    model_precision = random.uniform(0.60, 0.80)
    model_recall = random.uniform(0.60, 0.80)
    model_f1_score = random.uniform(0.60, 0.80)
    
    metrics_storage.store_gauge("model_accuracy", model_accuracy)
    metrics_storage.store_gauge("model_precision", model_precision)
    metrics_storage.store_gauge("model_recall", model_recall)
    metrics_storage.store_gauge("model_f1_score", model_f1_score)

def generate_simple_metrics():
    """Generate simple test metrics."""
    logger.info("📊 Generating simple metrics...")
    
    # Simple counter
    global _simple_counter
    _simple_counter += 1
    metrics_storage.store_counter("simple_counter_total", _simple_counter)
    
    # Simple gauges
    metrics_storage.store_gauge("simple_gauge", random.uniform(0, 100))
    metrics_storage.store_gauge("simple_temperature", random.uniform(20, 30))
    metrics_storage.store_gauge("simple_pressure", random.uniform(1000, 1100))

def main():
    """Main function to generate continuous metrics."""
    global _simple_counter
    _simple_counter = 0
    
    logger.info("🚀 Starting PostgreSQL Metrics Generator")
    logger.info("This will generate continuous metrics and store them in PostgreSQL")
    
    try:
        iteration = 1
        while True:
            logger.info(f"\n🔄 Iteration {iteration}")
            
            # Generate different types of metrics
            generate_simple_metrics()
            time.sleep(2)
            
            generate_system_metrics()
            time.sleep(1)
            
            generate_model_metrics()
            time.sleep(1)
            
            # Generate prediction metrics (less frequently)
            if iteration % 3 == 0:
                generate_prediction_metrics()
                time.sleep(2)
            
            # Wait before next iteration
            wait_time = random.randint(5, 10)
            logger.info(f"⏳ Waiting {wait_time} seconds before next iteration...")
            time.sleep(wait_time)
            iteration += 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping PostgreSQL Metrics Generator")
    except Exception as e:
        logger.error(f"❌ Error in main loop: {e}")

if __name__ == "__main__":
    main() 