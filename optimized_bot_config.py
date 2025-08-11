#!/usr/bin/env python3
"""
Optimized configuration for SIRENUSDT trading bot
Based on backtesting analysis results
"""

# Optimized parameters for SIRENUSDT
OPTIMIZED_PARAMS = {
    "rsi_period": 14,
    "rsi_oversold": 41,
    "rsi_overbought": 73,
    "ema_fast": 7,
    "ema_slow": 18,
    "bb_period": 20,
    "bb_std": 2.0,
    "confidence_threshold": 75,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9
}

# Risk management (keep existing safe settings)
RISK_PARAMS = {
    "max_trade_size_usd": 2.0,
    "stop_loss_percent": 0.1,
    "risk_per_trade": 0.01,
    "max_leverage": 25,  # SIRENUSDT max leverage
    "take_profit_ratio": 2.0  # 2:1 risk-reward
}

# Trading rules optimized for SIRENUSDT volatility
TRADING_RULES = {
    "min_volume_ratio": 0.8,  # Lower threshold for SIREN
    "max_atr_percent": 3.0,   # Higher volatility tolerance
    "cooldown_minutes": 3,    # Shorter cooldown for scalping
    "max_daily_trades": 50,   # More trades allowed
    "trend_strength_min": 0.05  # Lower trend requirement
}