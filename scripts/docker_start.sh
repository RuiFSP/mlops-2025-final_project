#!/bin/bash

# Docker Compose management script for Premier League MLOps system

# Set script to exit on error
set -e

# Default values
ACTION="start"
SERVICES=""

# Display help
function show_help {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -a, --action ACTION        Action to perform (start|stop|restart|logs|status)"
    echo "  -s, --services SERVICES    Specific services to target (comma-separated)"
    echo ""
    echo "Examples:"
    echo "  $0 --action start                   # Start all services"
    echo "  $0 --action stop                    # Stop all services"
    echo "  $0 --action restart --services api  # Restart only the API service"
    echo "  $0 --action logs --services mlflow  # View logs for MLflow service"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--action)
            ACTION="$2"
            shift
            shift
            ;;
        -s|--services)
            SERVICES="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate action
if [[ ! "$ACTION" =~ ^(start|stop|restart|logs|status)$ ]]; then
    echo "Error: Invalid action '$ACTION'"
    show_help
    exit 1
fi

# Create necessary directories
mkdir -p data/predictions data/real_data models logs mlflow/artifacts

# Execute the requested action
case $ACTION in
    start)
        echo "Starting services..."
        if [ -z "$SERVICES" ]; then
            docker-compose up -d
        else
            docker-compose up -d $(echo $SERVICES | tr ',' ' ')
        fi
        echo "Services started. Dashboard available at: http://localhost:8501"
        ;;
    stop)
        echo "Stopping services..."
        if [ -z "$SERVICES" ]; then
            docker-compose down
        else
            docker-compose stop $(echo $SERVICES | tr ',' ' ')
        fi
        echo "Services stopped."
        ;;
    restart)
        echo "Restarting services..."
        if [ -z "$SERVICES" ]; then
            docker-compose restart
        else
            docker-compose restart $(echo $SERVICES | tr ',' ' ')
        fi
        echo "Services restarted."
        ;;
    logs)
        echo "Showing logs..."
        if [ -z "$SERVICES" ]; then
            docker-compose logs -f
        else
            docker-compose logs -f $(echo $SERVICES | tr ',' ' ')
        fi
        ;;
    status)
        echo "Service status:"
        docker-compose ps
        ;;
esac

exit 0 