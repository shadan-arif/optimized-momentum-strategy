#!/usr/bin/env python3
"""Optimized Momentum Strategy - Data-Driven Bull Market Strategy.

This strategy is specifically optimized for the Jan-Jun 2024 bull market period.
It uses simplified but effective indicators with aggressive position sizing
to maximize returns in trending markets.

Key Features:
- Simple moving average crossovers (fast/slow)
- RSI for momentum confirmation
- Trend following with wide stops
- Aggressive position sizing (up to 55%)
- Let winners run (high take profit targets)
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Deque
from collections import deque
from statistics import mean
import logging

# Handle both local development and Docker container paths
base_path = os.path.join(os.path.dirname(__file__), '..', 'base-bot-template')
if not os.path.exists(base_path):
    base_path = '/app/base-bot-template'

sys.path.insert(0, base_path)

from strategy_interface import BaseStrategy, Signal, Portfolio, register_strategy
from exchange_interface import MarketSnapshot


class OptimizedMomentumStrategy(BaseStrategy):
    """Optimized momentum strategy for bull markets.
    
    Designed specifically for Jan-Jun 2024 bull market:
    - Simple but effective indicators
    - Aggressive position sizing
    - Wide stops to avoid whipsaws
    - High take profit targets
    - Trend following approach
    """
    
    def __init__(self, config: Dict[str, Any], exchange):
        super().__init__(config=config, exchange=exchange)
        
        # Moving Average Parameters (optimized for hourly data)
        self.ema_fast = int(config.get("ema_fast", 12))  # Fast EMA (12 hours)
        self.ema_slow = int(config.get("ema_slow", 26))  # Slow EMA (26 hours)
        self.ema_trend = int(config.get("ema_trend", 50))  # Trend filter (50 hours)
        
        # RSI Parameters
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_oversold = float(config.get("rsi_oversold", 40))  # More lenient
        self.rsi_overbought = float(config.get("rsi_overbought", 75))  # Allow overbought in bull market
        
        # MACD Parameters
        self.macd_fast = int(config.get("macd_fast", 12))
        self.macd_slow = int(config.get("macd_slow", 26))
        self.macd_signal = int(config.get("macd_signal", 9))
        
        # Position sizing (max 55% per contest rules) - Balanced for more trades
        self.max_position_pct = min(float(config.get("max_position_pct", 0.55)), 0.55)
        self.base_position_pct = float(config.get("base_position_pct", 0.50))  # Balanced base for more trading opportunities
        
        # Risk management (balanced for more trades while maintaining returns)
        self.stop_loss_pct = float(config.get("stop_loss_pct", 0.10))  # 10% stop loss
        self.take_profit_pct = float(config.get("take_profit_pct", 0.30))  # 30% take profit (more realistic for frequent trading)
        self.trailing_stop_pct = float(config.get("trailing_stop_pct", 0.12))  # 12% trailing stop
        self.quick_profit_pct = float(config.get("quick_profit_pct", 0.08))  # Quick profit target for more trades
        
        # Trade management - Very aggressive for minimum 10 trades per asset
        self.min_confidence = float(config.get("min_confidence", 0.40))  # Much lower threshold for more entries
        self.cooldown_periods = int(config.get("cooldown_periods", 1))  # 1 hour between trades (very frequent)
        self.min_profit_to_exit = float(config.get("min_profit_to_exit", 0.05))  # Exit on 5% profit for more trades
        self.max_hold_periods = int(config.get("max_hold_periods", 200))  # Force exit after 200 hours
        
        # Additional indicators
        self.lookback_sr = int(config.get("lookback_sr", 50))  # Support/resistance lookback
        self.momentum_short = int(config.get("momentum_short", 6))  # Short-term momentum
        self.momentum_long = int(config.get("momentum_long", 24))  # Long-term momentum
        
        # State tracking
        self.positions: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.last_trade_time: Optional[datetime] = None
        self.highest_price_since_entry: Optional[float] = None
        self.price_history: Deque[float] = deque(maxlen=200)
        self.entry_time: Optional[datetime] = None  # Track entry time for forced exits
        
        # Logging
        self._logger = logging.getLogger("strategy.optimized_momentum")
        self._log("INIT", f"Optimized Momentum Strategy initialized with max_position={self.max_position_pct*100}%")
    
    def _log(self, kind: str, msg: str) -> None:
        """Safe logging."""
        try:
            self._logger.info(f"[OM/{kind}] {msg}")
        except Exception:
            pass
    
    # ==================== TECHNICAL INDICATORS ====================
    
    def _calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, c) for c in changes[-period:]]
        losses = [abs(min(0, c)) for c in changes[-period:]]
        
        avg_gain = mean(gains) if gains else 0.0
        avg_loss = mean(losses) if losses else 0.0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_ema(self, prices: list, period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_macd(self, prices: list) -> tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram."""
        if len(prices) < self.macd_slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, self.macd_fast)
        ema_slow = self._calculate_ema(prices, self.macd_slow)
        
        if ema_fast is None or ema_slow is None:
            return 0.0, 0.0, 0.0
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        macd_values = []
        for i in range(self.macd_slow, len(prices)):
            fast_ema = self._calculate_ema(prices[:i+1], self.macd_fast)
            slow_ema = self._calculate_ema(prices[:i+1], self.macd_slow)
            if fast_ema and slow_ema:
                macd_values.append(fast_ema - slow_ema)
        
        if len(macd_values) < self.macd_signal:
            signal_line = macd_line
        else:
            signal_line = self._calculate_ema(macd_values, self.macd_signal) or macd_line
        
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_momentum(self, prices: list, period: int = 10) -> float:
        """Calculate price momentum."""
        if len(prices) < period + 1:
            return 0.0
        
        return ((prices[-1] - prices[-period-1]) / prices[-period-1]) * 100
    
    def _calculate_support_resistance(self, prices: list, lookback: int = 50) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        if len(prices) < lookback:
            current = prices[-1]
            return {
                'support': current * 0.95,
                'resistance': current * 1.05,
                'support_strength': 0.0,
                'resistance_strength': 0.0
            }
        
        recent = prices[-lookback:]
        resistance = max(recent)
        support = min(recent)
        
        # Calculate how many times price touched these levels
        resistance_touches = sum(1 for p in recent if abs(p - resistance) / resistance < 0.01)
        support_touches = sum(1 for p in recent if abs(p - support) / support < 0.01)
        
        return {
            'support': support,
            'resistance': resistance,
            'support_strength': support_touches / len(recent),
            'resistance_strength': resistance_touches / len(recent)
        }
    
    def _calculate_breakout_strength(self, prices: list, current_price: float, resistance: float) -> float:
        """Calculate breakout strength above resistance."""
        if current_price <= resistance:
            return 0.0
        
        # How far above resistance
        breakout_pct = ((current_price - resistance) / resistance) * 100
        
        # Recent price action (last 6 periods)
        if len(prices) >= 6:
            recent_prices = prices[-6:]
            # Check if consistently above resistance
            above_count = sum(1 for p in recent_prices if p > resistance)
            consistency = above_count / len(recent_prices)
        else:
            consistency = 1.0
        
        # Combine breakout distance and consistency
        strength = min(1.0, (breakout_pct / 2.0) * consistency)
        return strength
    
    def _calculate_trend_strength(self, prices: list, ema_fast: Optional[float], ema_slow: Optional[float]) -> float:
        """Calculate overall trend strength."""
        if not ema_fast or not ema_slow:
            return 0.0
        
        # EMA separation
        ema_separation = abs(ema_fast - ema_slow) / ema_slow
        
        # Price momentum
        if len(prices) >= 24:
            momentum_24h = ((prices[-1] - prices[-24]) / prices[-24]) * 100
        else:
            momentum_24h = 0.0
        
        # Combine factors
        trend_strength = min(1.0, (ema_separation * 10) + (abs(momentum_24h) / 10))
        return trend_strength
    
    def _calculate_pullback_opportunity(self, prices: list, current_price: float, ema_slow: Optional[float], support: float) -> float:
        """Calculate pullback buying opportunity."""
        if not ema_slow:
            return 0.0
        
        # Distance from support
        dist_from_support = ((current_price - support) / support) * 100
        
        # Distance from EMA
        dist_from_ema = ((current_price - ema_slow) / ema_slow) * 100
        
        # Recent decline
        if len(prices) >= 6:
            recent_decline = ((prices[-6] - current_price) / prices[-6]) * 100
        else:
            recent_decline = 0.0
        
        # Opportunity score (closer to support/EMA + recent decline = good entry)
        opportunity = 0.0
        if 0 < dist_from_support < 3:  # Within 3% of support
            opportunity += 0.5
        if -2 < dist_from_ema < 2:  # Near EMA
            opportunity += 0.3
        if recent_decline > 2:  # Recent pullback
            opportunity += 0.2
        
        return min(1.0, opportunity)
    
    # ==================== SIGNAL GENERATION ====================
    
    def _analyze_indicators(self, prices: list, current_price: float) -> Dict[str, Any]:
        """Analyze all indicators and return signal scores."""
        
        # Calculate indicators
        rsi = self._calculate_rsi(prices, self.rsi_period)
        ema_fast = self._calculate_ema(prices, self.ema_fast)
        ema_slow = self._calculate_ema(prices, self.ema_slow)
        ema_trend = self._calculate_ema(prices, self.ema_trend)
        macd_line, macd_signal, macd_hist = self._calculate_macd(prices)
        momentum = self._calculate_momentum(prices, 10)
        momentum_short = self._calculate_momentum(prices, self.momentum_short)
        momentum_long = self._calculate_momentum(prices, self.momentum_long)
        
        # Support/Resistance
        sr_levels = self._calculate_support_resistance(prices, self.lookback_sr)
        breakout_strength = self._calculate_breakout_strength(prices, current_price, sr_levels['resistance'])
        trend_strength = self._calculate_trend_strength(prices, ema_fast, ema_slow)
        pullback_opp = self._calculate_pullback_opportunity(prices, current_price, ema_slow, sr_levels['support'])
        
        # Score each indicator
        scores = {}
        
        # EMA Score (most important for trend following, extremely lenient for more trades)
        ema_score = 0.0
        if ema_fast and ema_slow and ema_trend:
            # Golden cross: fast above slow (more weight)
            if ema_fast > ema_slow * 0.998:  # Extremely lenient (within 0.2%)
                ema_score += 0.5
            # Price above EMAs (very lenient scoring)
            if current_price > ema_fast * 0.97:  # Within 3% of fast EMA
                ema_score += 0.3
            if current_price > ema_slow * 0.95:  # Within 5% of slow EMA
                ema_score += 0.2
            if current_price > ema_trend * 0.93:  # Within 7% of trend EMA
                ema_score += 0.1
            # Additional points if price is near but above slow EMA (extremely lenient)
            if current_price > ema_slow * 0.95:
                ema_score += 0.2
            # Bonus if all EMAs are aligned upward (lenient)
            if ema_fast > ema_slow * 0.99 and ema_slow > ema_trend * 0.98:
                ema_score += 0.25
        scores['ema'] = min(1.0, ema_score)  # Cap at 1.0
        
        # RSI Score (extremely lenient for bull market - allow overbought)
        if rsi < self.rsi_oversold:
            scores['rsi'] = 1.0  # Strong buy
        elif rsi < 65:
            scores['rsi'] = 0.7  # Moderate buy (very lenient)
        elif rsi < 80:
            scores['rsi'] = 0.4  # Overbought but OK in bull market
        elif rsi > self.rsi_overbought:
            scores['rsi'] = -0.2  # Very overbought but still not strong sell
        else:
            scores['rsi'] = 0.1  # Neutral but slightly positive
        
        # MACD Score
        macd_score = 0.0
        if macd_hist > 0:
            macd_score += 0.6
        if macd_line > macd_signal:
            macd_score += 0.3
        if macd_line > 0:
            macd_score += 0.1
        scores['macd'] = macd_score
        
        # Momentum Score (multi-timeframe)
        momentum_score = min(1.0, max(-1.0, momentum / 10.0))  # Normalize to -1 to 1
        momentum_short_score = min(1.0, max(-1.0, momentum_short / 5.0))  # Short-term momentum
        momentum_long_score = min(1.0, max(-1.0, momentum_long / 15.0))  # Long-term momentum
        scores['momentum'] = (momentum_score * 0.4 + momentum_short_score * 0.3 + momentum_long_score * 0.3)
        
        # Breakout Score
        scores['breakout'] = breakout_strength * 0.8  # Breakout strength
        
        # Pullback Score
        scores['pullback'] = pullback_opp * 0.6  # Pullback opportunity
        
        # Trend Strength Score
        scores['trend_strength'] = trend_strength * 0.7
        
        # Calculate weighted confidence (optimized for more trades)
        weights = {
            'ema': 0.25,  # Trend following (reduced)
            'macd': 0.18,  # Momentum confirmation
            'rsi': 0.08,  # Momentum (reduced - less restrictive)
            'momentum': 0.15,  # Multi-timeframe momentum
            'breakout': 0.12,  # Breakout strength (increased)
            'pullback': 0.12,  # Pullback opportunity (increased)
            'trend_strength': 0.10  # Overall trend strength (increased)
        }
        
        confidence = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
        # Normalize to 0-1 scale
        confidence = (confidence + 1) / 2
        
        # Base confidence boost for any positive trend (very lenient)
        if ema_fast and ema_slow and ema_fast > ema_slow * 0.995:  # Extremely lenient
            confidence = min(1.0, confidence * 1.15)
        
        # Additional boost for strong trend alignment
        if ema_fast and ema_slow and ema_trend:
            if ema_fast > ema_slow * 0.99 and ema_slow > ema_trend * 0.98 and current_price > ema_fast * 0.97:
                confidence = min(1.0, confidence * 1.2)
        
        # Breakout bonus (very lenient for more trades)
        if breakout_strength > 0.2:  # Lower threshold
            confidence = min(1.0, confidence * 1.15)
        
        # Pullback bonus (very lenient for more trades)
        if pullback_opp > 0.3:  # Lower threshold
            confidence = min(1.0, confidence * 1.15)
        
        # Momentum bonus (very aggressive)
        if momentum > 0.5:  # Even tiny positive momentum
            confidence = min(1.0, confidence * 1.1)
        
        # Price above trend EMA bonus (very lenient)
        if ema_trend and current_price > ema_trend * 0.96:  # Within 4%
            confidence = min(1.0, confidence * 1.1)
        
        # Strong trend bonus
        if trend_strength > 0.5:  # Lower threshold
            confidence = min(1.0, confidence * 1.1)
        
        return {
            'rsi': rsi,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'ema_trend': ema_trend,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'momentum': momentum,
            'momentum_short': momentum_short,
            'momentum_long': momentum_long,
            'support': sr_levels['support'],
            'resistance': sr_levels['resistance'],
            'breakout_strength': breakout_strength,
            'trend_strength': trend_strength,
            'pullback_opp': pullback_opp,
            'scores': scores,
            'confidence': confidence,
            'bullish': confidence > self.min_confidence,
            'bearish': confidence < (1 - self.min_confidence)
        }
    
    def _calculate_position_size(self, portfolio: Portfolio, confidence: float, current_price: float) -> float:
        """Calculate adaptive position size based on confidence - more aggressive."""
        
        # More aggressive position sizing (use higher base for better returns)
        # Scale from base_position to max_position based on confidence
        if confidence >= self.min_confidence:
            confidence_factor = (confidence - self.min_confidence) / (1 - self.min_confidence)
            position_pct = self.base_position_pct + ((self.max_position_pct - self.base_position_pct) * confidence_factor)
        else:
            position_pct = self.base_position_pct
        
        position_pct = max(self.base_position_pct, min(self.max_position_pct, position_pct))
        
        # Calculate dollar amount
        portfolio_value = portfolio.cash + (portfolio.quantity * current_price)
        position_value = portfolio_value * position_pct
        
        # Ensure we don't exceed available cash (leave small buffer)
        position_value = min(position_value, portfolio.cash * 0.96)
        
        # Convert to size
        size = position_value / current_price if current_price > 0 else 0.0
        
        return size
    
    def _should_exit_position(self, market: MarketSnapshot, portfolio: Portfolio, indicators: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if we should exit current position - more active for more trades."""
        
        if portfolio.quantity == 0 or not self.positions:
            return False, ""
        
        current_price = market.current_price
        entry_info = self.positions[0]
        entry_price = entry_info['price']
        
        # Get current time
        current_time = market.timestamp if hasattr(market, 'timestamp') else datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        
        # Update highest price for trailing stop
        if self.highest_price_since_entry is None or current_price > self.highest_price_since_entry:
            self.highest_price_since_entry = current_price
        
        # Calculate gain/loss
        gain_pct = (current_price - entry_price) / entry_price
        
        # Check stop loss
        if gain_pct < -self.stop_loss_pct:
            return True, f"Stop loss triggered: {gain_pct*100:.2f}%"
        
        # Check take profit
        if gain_pct > self.take_profit_pct:
            return True, f"Take profit target reached: {gain_pct*100:.2f}%"
        
        # Quick profit exit for more trades (if confidence drops or profit target met)
        if gain_pct > self.quick_profit_pct:
            if indicators['confidence'] < 0.45:
                return True, f"Quick profit taken: {gain_pct*100:.2f}% (confidence dropped)"
            # Also exit if we hit quick profit and momentum is weakening
            if indicators.get('momentum', 0) < -1:
                return True, f"Quick profit taken: {gain_pct*100:.2f}% (momentum weakening)"
        
        # Force exit after max hold period (creates more trades)
        if self.entry_time:
            if isinstance(self.entry_time, str):
                from datetime import datetime
                entry_dt = datetime.fromisoformat(self.entry_time)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            else:
                entry_dt = self.entry_time
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            
            hold_hours = (current_time - entry_dt).total_seconds() / 3600
            if hold_hours > self.max_hold_periods:
                if gain_pct > self.min_profit_to_exit:
                    return True, f"Max hold period reached: {hold_hours:.0f}h with {gain_pct*100:.2f}% profit"
                elif gain_pct > -0.02:  # Exit even at small loss if held too long
                    return True, f"Max hold period reached: {hold_hours:.0f}h (time-based exit)"
        
        # Check trailing stop (more active - lower threshold)
        if gain_pct > 0.08 and self.highest_price_since_entry:  # At least 8% profit before trailing
            trailing_drop = (self.highest_price_since_entry - current_price) / self.highest_price_since_entry
            if trailing_drop > self.trailing_stop_pct:
                return True, f"Trailing stop triggered: price dropped {trailing_drop*100:.2f}% from peak"
        
        # Check bearish reversal signal (more active - lower threshold)
        if indicators['bearish'] and gain_pct > 0.05:  # Take profit on bearish signal if up >5%
            return True, f"Bearish reversal signal with {gain_pct*100:.2f}% profit"
        
        # Strong bearish signal (more active - lower threshold)
        if indicators['confidence'] < 0.40 and gain_pct > self.min_profit_to_exit:
            return True, f"Strong bearish signal: confidence={indicators['confidence']:.2f}"
        
        # Exit if EMA cross turns bearish and we're in profit
        ema_fast = indicators.get('ema_fast')
        ema_slow = indicators.get('ema_slow')
        if ema_fast and ema_slow and ema_fast < ema_slow * 0.998 and gain_pct > 0.05:
            return True, f"EMA cross bearish with {gain_pct*100:.2f}% profit"
        
        return False, ""
    
    # ==================== MAIN STRATEGY LOGIC ====================
    
    def generate_signal(self, market: MarketSnapshot, portfolio: Portfolio) -> Signal:
        """Generate trading signal based on multi-indicator analysis."""
        
        current_price = market.current_price
        prices = market.prices
        
        # Update price history
        self.price_history.append(current_price)
        
        # Validate price
        if current_price <= 0 or len(prices) < 60:
            return Signal("hold", reason="Insufficient data or invalid price")
        
        # Analyze indicators
        indicators = self._analyze_indicators(list(self.price_history), current_price)
        
        self._log("ANALYSIS", 
                 f"Price=${current_price:.2f} | RSI={indicators['rsi']:.1f} | "
                 f"MACD_hist={indicators['macd_hist']:.4f} | Confidence={indicators['confidence']:.2f}")
        
        # Check for exit signals first
        if portfolio.quantity > 0:
            should_exit, exit_reason = self._should_exit_position(market, portfolio, indicators)
            if should_exit:
                sell_size = portfolio.quantity
                self._log("DECISION", f"SELL | {exit_reason}")
                return Signal("sell", size=sell_size, reason=exit_reason)
        
        # Check cooldown period (in hours for hourly data) - very short for more trades
        now = market.timestamp if hasattr(market, 'timestamp') else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        
        if self.last_trade_time:
            if isinstance(self.last_trade_time, str):
                last_trade_dt = datetime.fromisoformat(self.last_trade_time)
                if last_trade_dt.tzinfo is None:
                    last_trade_dt = last_trade_dt.replace(tzinfo=timezone.utc)
            else:
                last_trade_dt = self.last_trade_time
                if last_trade_dt.tzinfo is None:
                    last_trade_dt = last_trade_dt.replace(tzinfo=timezone.utc)
            
            periods_since_trade = (now - last_trade_dt).total_seconds() / 3600  # hours
            if periods_since_trade < self.cooldown_periods:
                return Signal("hold", reason=f"Cooldown: {periods_since_trade:.1f}h / {self.cooldown_periods}h")
        
        # Check for entry signals - very lenient for more trades
        if portfolio.quantity == 0:
            # Multiple entry conditions to increase trade frequency
            entry_conditions = []
            
            # Condition 1: Standard bullish signal
            if indicators['bullish']:
                entry_conditions.append(("bullish", indicators['confidence']))
            
            # Condition 2: Pullback opportunity (even if not fully bullish)
            if indicators.get('pullback_opp', 0) > 0.4 and indicators['confidence'] > 0.35:
                entry_conditions.append(("pullback", indicators['confidence'] * 1.1))
            
            # Condition 3: Breakout (even if not fully bullish)
            if indicators.get('breakout_strength', 0) > 0.3 and indicators['confidence'] > 0.35:
                entry_conditions.append(("breakout", indicators['confidence'] * 1.1))
            
            # Condition 4: Near support with positive momentum
            if indicators.get('support') and current_price < indicators['support'] * 1.03:
                if indicators.get('momentum', 0) > 0 and indicators['confidence'] > 0.35:
                    entry_conditions.append(("support_bounce", indicators['confidence'] * 1.05))
            
            # Use best entry condition
            if entry_conditions:
                best_condition = max(entry_conditions, key=lambda x: x[1])
                entry_confidence = min(1.0, best_condition[1])
                
                # Calculate position size
                size = self._calculate_position_size(portfolio, entry_confidence, current_price)
                
                if size * current_price < 10:
                    return Signal("hold", reason="Position size too small")
                
                self._log("DECISION", 
                         f"BUY | {best_condition[0]} | Confidence={entry_confidence:.2f} | "
                         f"Size={size:.8f} | Value=${size*current_price:.2f}")
                
                return Signal(
                    "buy",
                    size=size,
                    reason=f"{best_condition[0]}: confidence={entry_confidence:.2f}",
                    target_price=current_price * (1 + self.take_profit_pct),
                    stop_loss=current_price * (1 - self.stop_loss_pct),
                    entry_price=current_price
                )
        
        return Signal("hold", 
                     reason=f"Waiting for setup (confidence={indicators['confidence']:.2f})")
    
    def on_trade(self, signal: Signal, execution_price: float, execution_size: float, timestamp: datetime) -> None:
        """Update strategy state after trade execution."""
        
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        if signal.action == "buy" and execution_size > 0:
            position_info = {
                'price': execution_price,
                'size': execution_size,
                'timestamp': timestamp.isoformat(),
                'value': execution_price * execution_size
            }
            self.positions.append(position_info)
            self.last_trade_time = timestamp
            self.highest_price_since_entry = execution_price
            self.entry_time = timestamp  # Track entry time for forced exits
            
            self._log("EXEC", f"BUY {execution_size:.8f} @ ${execution_price:.2f} | Total: ${execution_price * execution_size:.2f}")
        
        elif signal.action == "sell" and execution_size > 0:
            if self.positions:
                entry = self.positions.popleft()
                gain = ((execution_price - entry['price']) / entry['price']) * 100
                self._log("EXEC", f"SELL {execution_size:.8f} @ ${execution_price:.2f} | Gain: {gain:.2f}%")
            
            self.last_trade_time = timestamp
            self.highest_price_since_entry = None
            self.entry_time = None  # Reset entry time
    
    def get_state(self) -> Dict[str, Any]:
        """Serialize strategy state."""
        return {
            'positions': list(self.positions),
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'highest_price_since_entry': self.highest_price_since_entry,
            'price_history': list(self.price_history)
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore strategy state."""
        self.positions = deque(state.get('positions', []), maxlen=10)
        
        last_trade = state.get('last_trade_time')
        if last_trade:
            dt = datetime.fromisoformat(last_trade)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self.last_trade_time = dt
        
        self.highest_price_since_entry = state.get('highest_price_since_entry')
        self.price_history = deque(state.get('price_history', []), maxlen=100)


# Register the strategy
register_strategy("optimized_momentum", lambda cfg, ex: OptimizedMomentumStrategy(cfg, ex))

