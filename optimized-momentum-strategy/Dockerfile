# Optimized Momentum Strategy - Dockerfile
FROM python:3.11-slim

LABEL description="Optimized Momentum Trading Bot"
LABEL strategies="optimized_momentum"
LABEL tier="optimized"

# Set working directory
WORKDIR /app

# Copy base bot infrastructure
COPY base-bot-template/ /app/base/

# Copy Optimized Momentum specific files
COPY optimized-momentum-strategy/ /app/

# Install dependencies
RUN pip install --no-cache-dir -r /app/base/requirements.txt

# Make startup script executable
RUN chmod +x /app/startup.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Expose ports
EXPOSE 8080 3010

# Default environment
ENV BOT_EXCHANGE=paper
ENV BOT_STRATEGY=optimized_momentum
ENV BOT_SYMBOL=BTC-USD
ENV BOT_STARTING_CASH=10000.0
ENV BOT_SLEEP=3600

# Start the Optimized Momentum bot
CMD ["python", "/app/startup.py"]

