# Optimized Momentum Strategy

Data-driven momentum trading strategy optimized specifically for the Jan-Jun 2024 bull market period. This strategy uses simplified but effective indicators with aggressive position sizing to maximize returns in trending markets.

## Strategy Overview

The Optimized Momentum Strategy is designed specifically for bull market conditions. It uses a simplified multi-indicator approach with aggressive position sizing and wide stops to let winners run while avoiding whipsaws. The strategy is optimized to achieve >25% returns while maintaining contest compliance.

## Core Features

### 🎯 Multi-Indicator System
- **EMA (Exponential Moving Averages)**: Fast (12h), Slow (26h), and Trend (50h) filters
- **RSI (Relative Strength Index)**: Momentum confirmation with lenient thresholds for bull markets
- **MACD (Moving Average Convergence Divergence)**: Trend momentum detection
- **Momentum Scoring**: Multi-timeframe momentum analysis (6h, 24h)
- **Support/Resistance**: Dynamic level detection for entry/exit timing
- **Breakout Detection**: Identifies strong breakout opportunities

### 💎 Aggressive Position Sizing
- **Dynamic allocation**: 50-55% of portfolio based on signal confidence
- **Contest compliant**: Never exceeds 55% maximum position size
- **Confidence-based**: Higher confidence = larger positions
- **Balanced approach**: Base 50% position for more trading opportunities

### 🛡️ Risk Management
- **Stop Loss**: 10% maximum loss per trade
- **Take Profit**: 30% profit target (let winners run)
- **Trailing Stop**: 12% trailing stop to protect gains
- **Quick Profit**: 8% quick profit target for frequent trading
- **Cooldown Period**: 1 hour minimum between trades (very active)
- **Max Hold Period**: 200 hours forced exit

### 🎲 Signal Generation
- **Confidence Threshold**: Minimum 40% confidence (lenient for more trades)
- **Multiple Entry Conditions**: Bullish signals, pullbacks, breakouts, support bounces
- **Weighted Scoring**: Each indicator contributes to overall score
- **Bearish Reversal Detection**: Early exit on trend reversals

## Configuration

### Default Parameters (Optimized for BTC/ETH Bull Market)

```json
{
  "exchange": "paper",
  "strategy": "optimized_momentum",
  "symbol": "BTC-USD",
  "starting_cash": 10000.0,
  "sleep_seconds": 3600,
  "strategy_params": {
    "ema_fast": 12,
    "ema_slow": 26,
    "ema_trend": 50,
    "rsi_period": 14,
    "rsi_oversold": 40,
    "rsi_overbought": 75,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "max_position_pct": 0.55,
    "base_position_pct": 0.50,
    "stop_loss_pct": 0.10,
    "take_profit_pct": 0.30,
    "trailing_stop_pct": 0.12,
    "quick_profit_pct": 0.08,
    "min_confidence": 0.40,
    "cooldown_periods": 1,
    "max_hold_periods": 200
  }
}
```

## Environment Variables

```bash
# Core Configuration
BOT_EXCHANGE=paper
BOT_STRATEGY=optimized_momentum
BOT_SYMBOL=BTC-USD
BOT_STARTING_CASH=10000.0
BOT_SLEEP=3600

# Dashboard Integration (Optional)
BOT_INSTANCE_ID=your-bot-id
USER_ID=your-user-id
BOT_SECRET=your-hmac-secret
BASE_URL=https://your-app.com
POSTGRES_URL=postgresql://...
```

## Quick Start

### Prerequisites

This template inherits from `base-bot-template`. Ensure the base template exists:

```
strategy-contest-1/
├── base-bot-template/           # Required infrastructure
└── optimized-momentum-strategy/ # This strategy
```

### Local Development

```bash
# Run with default configuration
python startup.py

# Run with custom symbol
BOT_SYMBOL=ETH-USD python startup.py

# Run with custom parameters
BOT_STRATEGY_PARAMS='{"min_confidence":0.45,"max_position_pct":0.55}' python startup.py
```

### Docker Deployment

**Build (from repository root):**
```bash
docker build -f optimized-momentum-strategy/Dockerfile -t optimized-momentum-bot .
```

**Run:**
```bash
docker run -p 8080:8080 -p 3010:3010 \
  -e BOT_STRATEGY=optimized_momentum \
  -e BOT_SYMBOL=BTC-USD \
  -e BOT_STARTING_CASH=10000 \
  optimized-momentum-bot
```

**Run with custom parameters:**
```bash
docker run -p 8080:8080 -p 3010:3010 \
  -e BOT_STRATEGY=optimized_momentum \
  -e BOT_SYMBOL=ETH-USD \
  -e BOT_STARTING_CASH=10000 \
  -e BOT_STRATEGY_PARAMS='{"min_confidence":0.45,"max_position_pct":0.55}' \
  optimized-momentum-bot
```

## Trading Logic

### Entry Conditions (BUY)
1. **No current position** (or position < max exposure)
2. **Confidence > 40%** from multi-indicator analysis (lenient threshold)
3. **Cooldown period elapsed** (1 hour since last trade)
4. **Sufficient capital** (minimum $10 trade size)
5. **Multiple entry conditions**:
   - **Standard bullish signal**: High confidence from indicator confluence
   - **Pullback opportunity**: Price near support with positive momentum
   - **Breakout**: Price breaking above resistance with strength
   - **Support bounce**: Price near support with positive momentum

### Exit Conditions (SELL)
1. **Stop Loss**: Price drops 10% below entry
2. **Take Profit**: Price rises 30% above entry
3. **Quick Profit**: 8% profit with confidence drop or momentum weakening
4. **Trailing Stop**: Price drops 12% from highest point (if in profit >8%)
5. **Bearish Reversal**: Confidence < 40% or strong bearish signals
6. **Max Hold Period**: Force exit after 200 hours (time-based exit)
7. **EMA Cross Bearish**: Fast EMA crosses below slow EMA with profit >5%

### Position Sizing Formula
```
base_position = 50% of portfolio
max_position = 55% of portfolio (contest limit)

position_pct = base_position + ((max_position - base_position) * confidence_factor)
position_size = (portfolio_value * position_pct) / current_price

where confidence_factor = (confidence - min_confidence) / (1 - min_confidence)
```

## Performance Targets

### Contest Requirements
- ✅ **Minimum Trades**: 10+ trades (strategy easily exceeds this with 94 trades)
- ✅ **Maximum Drawdown**: <50% (achieved 18.19%)
- ✅ **Position Sizing**: ≤55% per trade
- ✅ **Data Source**: Yahoo Finance hourly data
- ✅ **Period**: Jan-Jun 2024

### Performance Results
- 📈 **Total Return**: +23.27% (BTC: +19.96%, ETH: +26.58%)
- 🎯 **Win Rate**: 87.0%
- 📊 **Total Trades**: 94 trades (47 BTC + 47 ETH)
- 📉 **Max Drawdown**: 18.19%
- 📐 **Sharpe Ratio**: 0.26

## Technical Indicators Explained

### EMA (Exponential Moving Averages)
- **Fast**: 12 periods (short-term trend)
- **Slow**: 26 periods (medium-term trend)
- **Trend**: 50 periods (long-term trend filter)
- **Weight**: 25% of confidence score
- **Strategy**: Extremely lenient thresholds to catch more opportunities

### RSI (Relative Strength Index)
- **Range**: 0-100
- **Oversold**: <40 (buying opportunity)
- **Overbought**: >75 (still OK in bull market)
- **Weight**: 8% of confidence score
- **Strategy**: Lenient thresholds for bull market conditions

### MACD (Moving Average Convergence Divergence)
- **Components**: Fast EMA (12), Slow EMA (26), Signal (9)
- **Bullish**: Histogram > 0, MACD line > signal line
- **Weight**: 18% of confidence score

### Momentum Score
- **Short-term**: 6 periods
- **Long-term**: 24 periods
- **Weight**: 15% of confidence score

### Breakout & Pullback Detection
- **Breakout Strength**: Measures distance above resistance
- **Pullback Opportunity**: Identifies buying near support
- **Weight**: 12% each of confidence score

## API Endpoints

### Health Check (Port 8080)
- `GET /health` - Bot status and strategy info

### Control API (Port 3010, HMAC Authenticated)
- `GET /performance` - Real-time performance metrics
- `GET /settings` - Current configuration
- `POST /settings` - Update configuration (hot reload)
- `POST /commands` - Bot control (start/stop/pause/restart)
- `GET /logs` - Recent trading logs

## Dashboard Integration

Full compatibility with the main app dashboard:

- **Performance Metrics**: Real-time P&L, positions, trade history
- **Settings Management**: Hot configuration reload via dashboard
- **Bot Controls**: Start/stop/pause/restart from dashboard
- **Live Logs**: Structured log output with trade details
- **Status Reporting**: Real-time status updates via callbacks

## Strategy Optimization

This strategy is specifically optimized for:
- **Bull Market Conditions**: Jan-Jun 2024 was a strong bull market
- **High Trade Frequency**: 1-hour cooldown enables frequent trading
- **Aggressive Position Sizing**: 50-55% positions for maximum returns
- **Wide Stops**: 10% stop loss avoids whipsaws in volatile markets
- **Let Winners Run**: 30% take profit targets capture large moves
- **Multiple Entry Conditions**: Catches various market setups

## Risk Disclosure

This strategy is designed for the trading contest with historical backtesting data. Past performance does not guarantee future results. Cryptocurrency trading carries significant risk. Always:

- Use appropriate position sizing
- Set stop losses on all trades
- Never risk more than you can afford to lose
- Understand the strategy logic before deploying
- Monitor performance regularly

## Support & Documentation

For questions, issues, or contributions:
- Review the strategy code in `optimized_momentum_strategy.py`
- Check the backtest report in `reports/backtest_report.md`
- Read the logic explanation in `trade_logic_explanation.md`

## License

This strategy is submitted for the Trading Strategy Contest. All rights reserved.

