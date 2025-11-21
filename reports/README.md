# Reports - Backtest Runner Setup

This folder contains the backtest runner and Docker configuration for running the Optimized Momentum Strategy backtest.

## Prerequisites

- Docker installed and running
- Access to the repository root directory

## Building the Docker Image

From the **repository root directory**, build the Docker image:

```bash
docker build -f reports/Dockerfile -t optimized-backtest .
```

This command:

- Uses the Dockerfile in the `reports/` folder
- Tags the image as `optimized-backtest`
- Builds from the repository root (required for COPY commands)

## Running the Backtest

### Basic Run (No Volume Mount)

```bash
docker run --rm optimized-backtest
```

### Run with Results Volume Mount

To save results to a local directory (reports folder):

```bash
docker run --rm -v "$(pwd)/reports:/app/reports" optimized-backtest
```

This mounts the local `reports/` directory to `/app/reports` in the container, allowing you to access the `backtest_results.json` file after the run.

Alternatively, to save to a different location (e.g., results folder):

```bash
docker run --rm -v "$(pwd)/optimized-momentum-strategy/results:/app/results" -e OUTPUT_DIR=/app/results optimized-backtest
```

### Run with Custom Output Directory

If you want to save results to a different location:

```bash
docker run --rm -v "$(pwd)/your-results-folder:/app/results" -e OUTPUT_DIR=/app/results optimized-backtest
```

## Expected Output

The backtest will:

1. Fetch BTC-USD and ETH-USD hourly data from Yahoo Finance (Jan 1 - Jun 30, 2024)
2. Run the Optimized Momentum Strategy backtest
3. Display detailed trade logs and performance metrics
4. Save results to `backtest_results.json` in the output directory

## Output Files

After running, you'll find:

- `backtest_results.json` - Complete backtest results including:
  - BTC and ETH individual results
  - Combined performance metrics
  - Trade history (summary)
  - Equity curves (summary)

## Troubleshooting

### Build Errors

If you get build errors, ensure you're running from the repository root:

```bash
cd /path/to/crypto-trading-strategy
docker build -f reports/Dockerfile -t optimized-backtest .
```

### Permission Errors

If you encounter permission errors with volume mounts:

```bash
# On Linux/Mac, ensure directory permissions
chmod -R 755 optimized-momentum-strategy/results
```

### Network Issues

If data fetching fails, check your internet connection. The backtest requires access to Yahoo Finance API.

## Example Complete Workflow

```bash
# 1. Navigate to repository root
cd /path/to/crypto-trading-strategy

# 2. Build the Docker image
docker build -f reports/Dockerfile -t optimized-backtest .

# 3. Run the backtest with volume mount
docker run --rm -v "$(pwd)/reports:/app/reports" optimized-backtest

# 4. View results
cat reports/backtest_results.json
```

## Environment Variables

You can customize the backtest using environment variables:

```bash
docker run --rm \
  -v "$(pwd)/optimized-momentum-strategy/results:/app/results" \
  -e OUTPUT_DIR=/app/results \
  optimized-backtest
```

## Notes

- The backtest fetches data from Yahoo Finance API, so an internet connection is required
- The backtest may take several minutes to complete (depends on network speed)
- Results are saved in JSON format for easy parsing
- The container is automatically removed after execution (`--rm` flag)
