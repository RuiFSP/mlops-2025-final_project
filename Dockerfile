FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    gcc \
    g++ \
    curl \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Install uv
RUN pip install --no-cache-dir uv

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Create home directory and set permissions first
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser

# Copy dependency files with correct ownership
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser README.md ./
COPY --chown=appuser:appuser entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/ ./data/

# Install Python dependencies
RUN uv sync --frozen

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Set the entrypoint to our custom script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command to run the API
CMD ["src.deployment.api"]
