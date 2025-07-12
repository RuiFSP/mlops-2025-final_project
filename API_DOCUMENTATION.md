# Premier League Match Predictor API Documentation

## Overview

The Premier League Match Predictor API provides a comprehensive REST interface for predicting match outcomes and simulating betting strategies. Built with FastAPI, it offers high-performance endpoints for real-time predictions and betting simulation.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for API access. In production, implement API keys or OAuth2.

## API Endpoints

### 1. Health & Status

#### GET `/health`
Check API health and component status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "prediction_pipeline": "operational",
    "betting_simulator": "operational",
    "real_data_fetcher": "operational"
  },
  "timestamp": "2025-07-12T15:35:55.894623"
}
```

#### GET `/`
Root endpoint with basic API information.

**Response:**
```json
{
  "message": "Premier League Match Predictor API",
  "version": "1.0.0",
  "docs": "/docs",
  "status": "operational"
}
```

### 2. Predictions

#### POST `/predict`
Predict the outcome of a single match.

**Request Body:**
```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "home_odds": 2.1,
  "away_odds": 3.2,
  "draw_odds": 3.0
}
```

**Response:**
```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "prediction": "A",
  "confidence": 0.4266627799292951,
  "probabilities": {
    "H": 0.35559673780499085,
    "D": 0.21774048226571383,
    "A": 0.4266627799292951
  },
  "created_at": "2025-07-12T15:36:03.870869"
}
```

**Response Fields:**
- `prediction`: "H" (Home win), "D" (Draw), "A" (Away win)
- `confidence`: Prediction confidence (0-1)
- `probabilities`: Individual outcome probabilities

#### POST `/predict/batch`
Predict outcomes for multiple matches.

**Request Body:**
```json
[
  {
    "home_team": "Manchester City",
    "away_team": "Liverpool",
    "home_odds": 1.8,
    "away_odds": 4.0,
    "draw_odds": 3.5
  },
  {
    "home_team": "Manchester United",
    "away_team": "Tottenham",
    "home_odds": 2.5,
    "away_odds": 2.8,
    "draw_odds": 3.2
  }
]
```

**Response:** Array of prediction objects (same format as single prediction)

#### GET `/predictions/upcoming`
Get predictions for upcoming Premier League matches.

**Response:** Array of prediction objects with real upcoming matches

### 3. Matches

#### GET `/matches/upcoming`
Get upcoming Premier League matches.

**Response:**
```json
[
  {
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "match_date": "2025-07-13T15:00:00",
    "home_odds": 2.1,
    "away_odds": 3.2,
    "draw_odds": 3.0
  }
]
```

### 4. Betting Simulation

#### POST `/betting/simulate`
Simulate betting on provided predictions.

**Request Body:**
```json
{
  "predictions": [
    {
      "match_id": "test_match_1",
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "prediction": "H",
      "confidence": 0.65,
      "home_odds": 2.1,
      "draw_odds": 3.0,
      "away_odds": 3.2
    }
  ]
}
```

**Response:**
```json
{
  "bets_placed": 1,
  "total_amount_bet": 50.0,
  "successful_bets": [
    {
      "match_id": "test_match_1",
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "prediction": "H",
      "bet_amount": 50.0,
      "potential_return": 105.0,
      "confidence": 0.65
    }
  ],
  "betting_statistics": {
    "total_bets": 7,
    "winning_bets": 0,
    "win_rate": 0.0,
    "total_amount_bet": 314.91,
    "total_winnings": 0.0,
    "overall_roi": -1.0,
    "current_balance": 685.09,
    "profit_loss": -314.91
  },
  "current_balance": 685.09
}
```

#### GET `/betting/statistics`
Get current betting statistics.

**Response:**
```json
{
  "statistics": {
    "total_bets": 7,
    "winning_bets": 0,
    "win_rate": 0.0,
    "total_amount_bet": 314.91,
    "total_winnings": 0.0,
    "overall_roi": -1.0,
    "current_balance": 685.09,
    "profit_loss": -314.91
  },
  "current_balance": 685.09,
  "initial_balance": 1000.0,
  "timestamp": "2025-07-12T15:36:03.870869"
}
```

### 5. Model Information

#### GET `/model/info`
Get information about the current prediction model.

**Response:**
```json
{
  "model_type": "Random Forest",
  "accuracy": "61.84%",
  "features": 15,
  "training_matches": 3040,
  "last_updated": "2025-07-12T15:36:03.870869",
  "version": "latest"
}
```

### 6. 🆕 Automated Retraining

#### GET `/retraining/status`
Get the current status of the automated retraining system.

**Response:**
```json
{
  "model_info": {
    "version": "1",
    "accuracy": 0.618,
    "creation_date": "2025-07-12T15:18:00.049000",
    "status": "READY"
  },
  "monitoring_summary": {
    "recent_evaluations": [],
    "recent_retraining": []
  },
  "config": {
    "min_accuracy_threshold": 0.55,
    "min_prediction_count": 100,
    "evaluation_window_days": 7,
    "max_model_age_days": 30,
    "performance_degradation_threshold": 0.05,
    "consecutive_poor_performance_limit": 3
  },
  "timestamp": "2025-07-12T15:50:50.123456"
}
```

#### POST `/retraining/check`
Run an immediate retraining check to evaluate model performance.

**Response:**
```json
{
  "check_result": {
    "status": "performance_acceptable",
    "metrics": {
      "accuracy": 0.618,
      "precision": 0.610,
      "recall": 0.605,
      "f1_score": 0.608,
      "prediction_count": 23
    },
    "message": "Model performance is acceptable"
  },
  "timestamp": "2025-07-12T15:50:50.123456"
}
```

#### POST `/retraining/force`
Force an immediate retraining of the model.

**Response:**
```json
{
  "retraining_result": {
    "success": true,
    "duration_seconds": 45.2,
    "new_model_version": "2",
    "new_accuracy": 0.625,
    "message": "Retraining completed successfully"
  },
  "timestamp": "2025-07-12T15:50:50.123456"
}
```

#### GET `/retraining/history`
Get the history of retraining activities and performance evaluations.

**Response:**
```json
{
  "recent_evaluations": [
    {
      "model_version": "1",
      "accuracy": 0.618,
      "precision": 0.610,
      "recall": 0.605,
      "f1_score": 0.608,
      "prediction_count": 23,
      "evaluation_date": "2025-07-12T15:50:50.123456",
      "requires_retraining": false
    }
  ],
  "recent_retraining": [
    {
      "trigger_reason": "Manual force retraining via API",
      "old_model_version": "1",
      "new_model_version": "2",
      "old_accuracy": 0.618,
      "new_accuracy": 0.625,
      "training_duration_seconds": 45,
      "retraining_date": "2025-07-12T15:50:50.123456",
      "success": true
    }
  ],
  "timestamp": "2025-07-12T15:50:50.123456"
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found
- `500`: Internal Server Error
- `503`: Service Unavailable (component not initialized)

**Error Response Format:**
```json
{
  "detail": "Error message",
  "timestamp": "2025-07-12T15:36:03.870869"
}
```

## Rate Limiting

Currently no rate limiting is implemented. In production, implement rate limiting to prevent abuse.

## Interactive Documentation

The API provides interactive documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Usage Examples

### Python Client Example

```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "home_odds": 2.1,
    "away_odds": 3.2,
    "draw_odds": 3.0
})
prediction = response.json()
print(f"Prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")

# Betting simulation
response = requests.post("http://localhost:8000/betting/simulate", json={
    "predictions": [prediction]
})
betting_result = response.json()
print(f"Bets placed: {betting_result['bets_placed']}")
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea", "home_odds": 2.1, "away_odds": 3.2, "draw_odds": 3.0}'

# Get upcoming matches
curl -X GET "http://localhost:8000/matches/upcoming"

# Betting statistics
curl -X GET "http://localhost:8000/betting/statistics"
```

## Production Considerations

1. **Authentication**: Implement API keys or OAuth2
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **CORS**: Configure CORS for specific origins
4. **Logging**: Enhanced logging for monitoring
5. **Monitoring**: Add metrics and health checks
6. **Caching**: Cache predictions and model results
7. **Load Balancing**: Use load balancers for high availability

## Running the API

### Development
```bash
cd src/api
uv run python main.py
```

### Production
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Testing

Run the API test suite:
```bash
uv run python scripts/test_api.py
```

## System Requirements

- Python 3.10+
- FastAPI
- uvicorn
- PostgreSQL
- MLflow
- All dependencies from `pyproject.toml`

## Support

For issues or questions, check the project documentation or raise an issue in the repository.
