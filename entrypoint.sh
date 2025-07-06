#!/bin/bash
set -e

# If no arguments or just --help, show help for the API
if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
    exec uv run python -m src.deployment.api --help
fi

# If the first argument is a known module, run it with uv
case "$1" in
    "src.deployment.api"|"api")
        shift
        exec uv run python -m src.deployment.api "$@"
        ;;
    "src.model_training.trainer"|"train")
        shift
        exec uv run python -m src.model_training.trainer "$@"
        ;;
    "src.data_collection.data_collector"|"collect")
        shift
        exec uv run python -m src.data_collection.data_collector "$@"
        ;;
    *)
        # For other cases, try to run as a Python module
        exec uv run python -m "$@"
        ;;
esac
