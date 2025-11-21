#!/usr/bin/env python3
"""Optimized Momentum Strategy Bot - Startup Script."""

from __future__ import annotations

import sys
import os

# Import base infrastructure from base-bot-template
# Handle both local development and Docker container paths
base_path = os.path.join(os.path.dirname(__file__), '..', 'base-bot-template')
if not os.path.exists(base_path):
    # In Docker container, base template is at /app/base/
    base_path = '/app/base'

sys.path.insert(0, base_path)

# Import Optimized Momentum strategy (this registers the strategy)
import optimized_momentum_strategy

# Import base bot infrastructure
from universal_bot import UniversalBot


def main() -> None:
    """Main entry point for Optimized Momentum Bot."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    bot = UniversalBot(config_path)

    # Print startup info with unique identifiers
    print("=" * 70)
    print("🚀 OPTIMIZED MOMENTUM TRADING BOT")
    print("=" * 70)
    print(f"🆔 Bot ID: {bot.config.bot_instance_id}")
    print(f"👤 User ID: {bot.config.user_id}")
    print(f"📈 Strategy: {bot.config.strategy}")
    print(f"💰 Symbol: {bot.config.symbol}")
    print(f"🏦 Exchange: {bot.config.exchange}")
    print(f"💵 Starting Cash: ${bot.config.starting_cash:,.2f}")
    print("=" * 70)
    print("🎯 STRATEGY: Data-Optimized Momentum System")
    print("📊 INDICATORS: EMA, RSI, MACD, Momentum")
    print("🎲 APPROACH: Trend-following with aggressive sizing")
    print("🛡️  RISK: Wide stops, high take profit targets")
    print("💎 TARGET: >25% return (optimized for bull markets)")
    print("=" * 70)

    bot.run()


if __name__ == "__main__":
    main()

