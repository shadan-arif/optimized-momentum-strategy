# Backtest Dockerfile for Optimized Momentum Strategy
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (optional, for debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    yfinance \
    pandas \
    requests>=2.31

# Copy base bot infrastructure
COPY base-bot-template/ /app/base-bot-template/

# Copy optimized momentum strategy
COPY optimized-momentum-strategy/ /app/optimized-momentum-strategy/

# Copy reports folder (backtest runner)
COPY reports/ /app/reports/

# Set working directory to reports
WORKDIR /app/reports

# Run the backtest
CMD ["python3", "backtest_runner.py"]

