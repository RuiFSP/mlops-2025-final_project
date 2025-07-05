#!/bin/bash

# Development setup script for Premier League Match Predictor
set -e

echo "🏈 Setting up Premier League Match Predictor development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv found"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

# Install development dependencies
echo "📦 Installing development dependencies..."
uv sync --extra dev

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
uv run pre-commit install

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p evaluation_reports
mkdir -p mlflow/artifacts

# Run a quick test to ensure everything works
echo "🧪 Running quick test..."
if uv run python -m src.main --help > /dev/null 2>&1; then
    echo "✅ CLI is working"
else
    echo "❌ CLI test failed"
    exit 1
fi

# Test with sample data
echo "🏆 Running sample training..."
if uv run python -m src.main train > /dev/null 2>&1; then
    echo "✅ Training pipeline works"
else
    echo "⚠️  Training pipeline test failed (this might be due to small dataset)"
fi

echo ""
echo "🎉 Setup complete! You can now:"
echo "   • Run training: uv run python -m src.main train"
echo "   • Run tests: uv run pytest"
echo "   • Start MLflow: make mlflow-server"
echo "   • Start Prefect: make prefect-server"
echo "   • View notebook: jupyter lab notebooks/premier_league_analysis.ipynb"
echo ""
echo "💡 Check the README.md for more commands and information!"
