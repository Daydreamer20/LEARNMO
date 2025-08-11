#!/usr/bin/env python3
"""
Improved High-Leverage Multi-Symbol Trading Bot
Enhanced signal filtering and risk management for 75x leverage
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
import os
from typing import Dict, List, Optional
import wandb
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ImprovedHighLeverageParams:
    """Improved high-leverage parameters with better risk management"""
    leverage: float = 75.0
    margin_per_trade: float = 5.0
    stop_loss_pct: float = 0.0133       # ~1.33% (right before liquidation)
    take_profit_pct: float = 0.03       # 3% take profit
    max_positions_per_symbol: int = 1
    max_total_positions: int = 2
    min_balance: float = 20.0
    
    # Enhanced signal filtering
    momentum_threshold: float = 0.002   # Higher threshold for quality
    volume_multiplier: float = 1.5      # Stronger volume requirement
    rsi_lower: float = 20               # More extreme RSI levels
    rsi_upper: float = 80
    trend_strength_min: float = 0.001   # Minimum trend strength
    signal_quality_min: float = 70      # Minimum signal quality score
    
    # Risk management
    max_consecutive_losses: int = 3     # Stop after 3 consecutive los