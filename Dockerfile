FROM python:3.10-slim

# Install system dependencies for scientific Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libc-dev \
    libatlas-base-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the rest of the code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY data/ ./data/

# Create mlruns directory
RUN mkdir -p /mlruns

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["python", "-m", "src.main", "train", "--data-path", "data/real_data/"]
