#!/usr/bin/env python3
"""
Railway.com Deployment - Self-Learning Trading Bot
Continuously optimizes parameters using WANDB and live market data
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
import sqlite3  # Built into Python
import threading
from typing import Dict, List, Optional
import requests
import wandb
from dataclasses import dataclass, asdict
import pickle
import schedule

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output for Railway logs
        logging.FileHandler('trading_bot.log') if not os.getenv('RAILWAY_ENVIRONMENT') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingParameters:
    """Optimal high-risk trading parameters based on quantitative research"""
    # Risk Management (Kelly Criterion + Research-Based)
    leverage: float = 12.0                    # Reduced from 15x for better risk management
    position_size_pct: float = 0.08          # 8% per trade (Kelly-optimized)
    stop_loss_pct: float = 0.015             # 1.5% stop loss (ATR-based)
    take_profit_pct: float = 0.030           # 3% take profit (2:1 R:R ratio)
    
    # Signal Quality (Research-Backed Thresholds)
    momentum_threshold: float = 0.003        # Higher threshold for quality signals
    volume_multiplier: float = 1.8           # Strong volume confirmation required
    rsi_lower: float = 25.0                  # More extreme RSI levels
    rsi_upper: float = 75.0                  # More extreme RSI levels
    signal_quality_min: float = 70.0        # Minimum signal quality score
    
    # Risk Controls (Circuit Breakers)
    max_positions: int = 2                   # Max concurrent positions
    max_consecutive_losses: int = 2          # Stop after 2 consecutive losses
    daily_loss_limit: float = 0.10          # 10% max daily loss
    min_balance: float = 100.0               # Emergency stop threshold
    
    # Adaptive Learning
    optimization_interval: int = 1800       # Every 30 minutes (more frequent)
    lookback_period: int = 50               # Last 50 trades for analysis
    performance_threshold: float = 0.6      # 60% win rate threshold
    
    # Market Condition Filters
    volatility_filter: float = 0.025        # Skip trading if volatility > 2.5%
    trend_strength_min: float = 0.002       # Minimum trend strength required
    correlation_check: bool = True          # Avoid correlated positions
    
    def to_dict(self):
        return asdict(self)

class ContinuousLearningBot:
    """Self-learning trading bot for Railway deployment"""
    
    def __init__(self):
        # Railway environment setup
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
        self.port = int(os.getenv('PORT', 8080))
        
        # Trading state
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.positions = []
        self.trades = []
        
        # Learning parameters
        self.params = TradingParameters()
        self.performance_history = []
        self.optimization_interval = 3600  # Optimize every hour
        self.last_optimization = 0
        
        # Market data
        self.symbol = "SOONUSDT"
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        self.rest_url = "https://api.bybit.com"
        
        # Data storage
        self.price_history = []
        self.volume_history = []
        self.rsi_history = []
        self.market_data_buffer = []
        
        # Database for persistence
        self.db_path = "trading_data.db"
        self._init_database()
        
        # WANDB setup
        self._init_wandb()
        
        # Connection state
        self.ws_connection = None
        self.running = False
        self.last_kline_time = 0
        
        # Load previous state if exists
        self._load_state()
        
        logger.info("🚀 Continuous Learning Bot Initialized for Railway")
        logger.info(f"📊 Current Parameters: {self.params.to_dict()}")
        logger.info(f"💰 Starting Balance: ${self.balance:.2f}")
        logger.info(f"🌐 Railway Environment: {self.is_railway}")
    
    def _init_database(self):
        """Initialize SQLite database for data persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl_amount REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                duration_minutes REAL,
                parameters TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Parameters optimization history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameter_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parameters TEXT,
                performance_score REAL,
                total_trades INTEGER,
                win_rate REAL,
                total_return REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized")
    
    def _init_wandb(self):
        """Initialize WANDB for experiment tracking"""
        try:
            # Use environment variables for WANDB API key
            wandb_key = os.getenv('WANDB_API_KEY')
            if wandb_key:
                wandb.login(key=wandb_key)
            
            # Initialize WANDB project
            wandb.init(
                project="railway-trading-bot",
                name=f"continuous-learning-{datetime.now().strftime('%Y%m%d-%H%M')}",
                config=self.params.to_dict(),
                mode="online" if wandb_key else "disabled"
            )
            logger.info("✅ WANDB initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ WANDB initialization failed: {e}")
            logger.info("📊 Continuing without WANDB tracking")
    
    def _save_state(self):
        """Save current bot state"""
        state = {
            'balance': self.balance,
            'params': self.params.to_dict(),
            'trades': self.trades[-100:],  # Keep last 100 trades
            'performance_history': self.performance_history[-50:],  # Keep last 50 performance records
            'last_optimization': self.last_optimization
        }
        
        try:
            with open('bot_state.pkl', 'wb') as f:
                pickle.dump(state, f)
            logger.info("💾 Bot state saved")
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")
    
    def _load_state(self):
        """Load previous bot state"""
        try:
            if os.path.exists('bot_state.pkl'):
                with open('bot_state.pkl', 'rb') as f:
                    state = pickle.load(f)
                
                self.balance = state.get('balance', self.initial_balance)
                
                # Load parameters
                if 'params' in state:
                    for key, value in state['params'].items():
                        if hasattr(self.params, key):
                            setattr(self.params, key, value)
                
                self.trades = state.get('trades', [])
                self.performance_history = state.get('performance_history', [])
                self.last_optimization = state.get('last_optimization', 0)
                
                logger.info("✅ Previous state loaded")
                logger.info(f"💰 Restored balance: ${self.balance:.2f}")
                logger.info(f"📊 Restored {len(self.trades)} trades")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load previous state: {e}")
    
    async def connect_websocket(self):
        """Connect to Bybit WebSocket with retry logic"""
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔌 Connecting to WebSocket (attempt {attempt + 1}/{max_retries})")
                self.ws_connection = await asyncio.wait_for(
                    websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10),
                    timeout=15.0
                )
                
                # Subscribe to 3-minute klines
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"kline.3.{self.symbol}"]
                }
                
                await self.ws_connection.send(json.dumps(subscribe_msg))
                logger.info(f"✅ Connected to {self.symbol} 3-minute klines")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        logger.error("❌ All connection attempts failed")
        return False
    
    async def listen_websocket(self):
        """Listen for WebSocket messages with reconnection"""
        while self.running:
            try:
                async for message in self.ws_connection:
                    data = json.loads(message)
                    if 'topic' in data and data['topic'].startswith('kline'):
                        await self._process_kline_data(data)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("⚠️ WebSocket connection closed, attempting reconnection...")
                if self.running:
                    await asyncio.sleep(5)
                    if await self.connect_websocket():
                        continue
                    else:
                        break
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                if self.running:
                    await asyncio.sleep(10)
                    continue
                else:
                    break
    
    async def _process_kline_data(self, data: Dict):
        """Process incoming kline data"""
        try:
            kline_data = data['data'][0]
            
            kline = {
                'timestamp': int(kline_data['timestamp']),
                'open': float(kline_data['open']),
                'high': float(kline_data['high']),
                'low': float(kline_data['low']),
                'close': float(kline_data['close']),
                'volume': float(kline_data['volume']),
                'confirm': kline_data['confirm']
            }
            
            # Store market data
            self._store_market_data(kline)
            
            # Only process confirmed klines
            if kline['confirm'] and kline['timestamp'] != self.last_kline_time:
                self.last_kline_time = kline['timestamp']
                await self._process_new_candle(kline)
                
                # Check if it's time to optimize
                if time.time() - self.last_optimization > self.optimization_interval:
                    await self._trigger_optimization()
                
        except Exception as e:
            logger.error(f"❌ Error processing kline: {e}")
    
    def _store_market_data(self, kline: Dict):
        """Store market data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_data (timestamp, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                kline['timestamp'], kline['open'], kline['high'],
                kline['low'], kline['close'], kline['volume']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing market data: {e}")
    
    async def _process_new_candle(self, kline: Dict):
        """Process new confirmed candle"""
        price = kline['close']
        volume = kline['volume']
        timestamp = datetime.fromtimestamp(kline['timestamp'] / 1000)
        
        logger.info(f"📊 {self.symbol} @ ${price:.4f} | Vol: {volume:.0f} | {timestamp.strftime('%H:%M:%S')}")
        
        # Update history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]
            self.volume_history = self.volume_history[-200:]
        
        # Calculate RSI
        if len(self.price_history) >= 15:
            rsi = self._calculate_rsi(self.price_history)
            self.rsi_history.append(rsi)
            if len(self.rsi_history) > 200:
                self.rsi_history = self.rsi_history[-200:]
        
        # Trading logic
        if len(self.price_history) >= 25:
            await self._check_trading_signals(kline)
        
        await self._manage_positions(price)
        
        # Log to WANDB
        if wandb.run:
            wandb.log({
                'price': price,
                'volume': volume,
                'balance': self.balance,
                'open_positions': len(self.positions),
                'total_trades': len(self.trades),
                'rsi': self.rsi_history[-1] if self.rsi_history else 50
            })
        
        # Save state periodically
        if len(self.price_history) % 20 == 0:
            self._save_state()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    async def _check_trading_signals(self, kline: Dict):
        """Check for trading signals using optimal research-based parameters"""
        # Risk control: Check consecutive losses
        if self.consecutive_losses >= self.params.max_consecutive_losses:
            return
        
        # Risk control: Check daily loss limit
        if hasattr(self, 'daily_pnl') and self.daily_pnl < -self.initial_balance * self.params.daily_loss_limit:
            return
        
        # Position limits
        if len(self.positions) >= self.params.max_positions:
            return
        
        price = kline['close']
        volume = kline['volume']
        
        # Calculate volatility filter
        if len(self.price_history) >= 20:
            volatility = np.std(self.price_history[-20:]) / np.mean(self.price_history[-20:])
            if volatility > self.params.volatility_filter:
                return  # Skip trading in high volatility
        
        # Enhanced technical analysis
        sma_10 = np.mean(self.price_history[-10:])
        sma_20 = np.mean(self.price_history[-20:])
        ema_12 = self._calculate_ema(self.price_history, 12)
        
        # Momentum with higher threshold
        momentum = 0
        if len(self.price_history) >= 5:
            momentum = (price - self.price_history[-5]) / self.price_history[-5]
        
        # Trend strength calculation
        trend_strength = abs(sma_10 - sma_20) / sma_20 if sma_20 > 0 else 0
        if trend_strength < self.params.trend_strength_min:
            return  # Skip weak trends
        
        # Volume confirmation (stronger requirement)
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_spike = volume > avg_volume * self.params.volume_multiplier
        
        # RSI with more extreme levels
        rsi = self.rsi_history[-1] if self.rsi_history else 50
        rsi_ok = self.params.rsi_lower < rsi < self.params.rsi_upper
        
        # Calculate signal quality score (0-100)
        signal_quality = self._calculate_signal_quality(price, volume, momentum, rsi, trend_strength)
        
        # Only take high-quality signals
        if signal_quality < self.params.signal_quality_min:
            return
        
        # Enhanced signal logic with multiple confirmations
        bullish_signal = (
            price > ema_12 and  # Price above EMA
            sma_10 > sma_20 and  # Uptrend
            momentum > self.params.momentum_threshold and  # Strong momentum
            volume_spike and  # Volume confirmation required
            rsi_ok and  # RSI in acceptable range
            trend_strength > self.params.trend_strength_min  # Strong trend
        )
        
        bearish_signal = (
            price < ema_12 and  # Price below EMA
            sma_10 < sma_20 and  # Downtrend
            momentum < -self.params.momentum_threshold and  # Strong negative momentum
            volume_spike and  # Volume confirmation required
            rsi_ok and  # RSI in acceptable range
            trend_strength > self.params.trend_strength_min  # Strong trend
        )
        
        if bullish_signal:
            await self._open_position('long', price, signal_quality)
        elif bearish_signal:
            await self._open_position('short', price, signal_quality)
    
    def _calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def _calculate_signal_quality(self, price, volume, momentum, rsi, trend_strength):
        """Calculate signal quality score (0-100) based on research"""
        score = 0
        
        # Momentum quality (25 points)
        if abs(momentum) > self.params.momentum_threshold * 2:
            score += 25
        elif abs(momentum) > self.params.momentum_threshold:
            score += 15
        elif abs(momentum) > self.params.momentum_threshold * 0.5:
            score += 8
        
        # Volume quality (20 points)
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_ratio = volume / avg_volume
        if volume_ratio > 2.5:
            score += 20
        elif volume_ratio > self.params.volume_multiplier:
            score += 15
        elif volume_ratio > 1.2:
            score += 8
        
        # RSI quality (20 points)
        if self.params.rsi_lower < rsi < self.params.rsi_upper:
            if 40 < rsi < 60:  # Neutral zone
                score += 20
            elif 30 < rsi < 70:  # Good zone
                score += 15
            else:  # Acceptable zone
                score += 10
        
        # Trend alignment (20 points)
        if trend_strength > 0.005:
            score += 20
        elif trend_strength > 0.003:
            score += 15
        elif trend_strength > self.params.trend_strength_min:
            score += 10
        
        # Price action quality (15 points)
        if len(self.price_history) >= 3:
            price_consistency = abs(price - np.mean(self.price_history[-3:])) / price
            if price_consistency < 0.001:  # Consistent price action
                score += 15
            elif price_consistency < 0.002:
                score += 10
            elif price_consistency < 0.005:
                score += 5
        
        return min(score, 100)
    
    async def _open_position(self, side, price, signal_quality):
        """Open trading position with optimal risk management"""
        # Dynamic position sizing based on signal quality (Kelly Criterion inspired)
        base_size = self.balance * self.params.position_size_pct
        quality_multiplier = signal_quality / 100  # Scale by signal quality
        position_size = base_size * quality_multiplier
        
        # Ensure minimum position size
        position_size = max(position_size, self.balance * 0.02)  # Min 2%
        
        if side == 'long':
            stop_loss = price * (1 - self.params.stop_loss_pct)
            take_profit = price * (1 + self.params.take_profit_pct)
        else:
            stop_loss = price * (1 + self.params.stop_loss_pct)
            take_profit = price * (1 - self.params.take_profit_pct)
        
        position = {
            'id': f"{side}_{len(self.trades)}_{int(time.time())}",
            'side': side,
            'entry_price': price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'signal_quality': signal_quality,
            'parameters': self.params.to_dict()
        }
        
        self.positions.append(position)
        
        logger.info(f"🔥 OPENED {side.upper()}: ${price:.4f} | Quality: {signal_quality:.1f}")
        logger.info(f"   🛑 Stop Loss: ${stop_loss:.4f} | 🎯 Take Profit: ${take_profit:.4f}")
        logger.info(f"   💰 Size: ${position_size:.2f} ({self.params.leverage}x leverage)")
    
    async def _manage_positions(self, current_price):
        """Manage existing positions"""
        positions_to_close = []
        
        for position in self.positions:
            should_close = False
            exit_reason = ""
            
            if position['side'] == 'long':
                if current_price <= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif current_price >= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            else:
                if current_price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif current_price <= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position, current_price, exit_reason))
        
        for position, exit_price, exit_reason in positions_to_close:
            await self._close_position(position, exit_price, exit_reason)
    
    async def _close_position(self, position, exit_price, exit_reason):
        """Close position with optimal risk management tracking"""
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        pnl_amount = position['size'] * pnl_pct * self.params.leverage
        self.balance += pnl_amount
        
        # Update daily PnL tracking
        if not hasattr(self, 'daily_pnl'):
            self.daily_pnl = 0
        if not hasattr(self, 'last_reset_date'):
            self.last_reset_date = datetime.now().date()
        
        # Reset daily PnL at midnight
        if datetime.now().date() > self.last_reset_date:
            self.daily_pnl = 0
            self.last_reset_date = datetime.now().date()
        
        self.daily_pnl += pnl_amount
        
        # Update consecutive loss tracking
        if not hasattr(self, 'consecutive_losses'):
            self.consecutive_losses = 0
        
        if pnl_amount < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on winning trade
        
        duration = (datetime.now() - position['entry_time']).total_seconds() / 60
        
        trade = {
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct * 100,
            'leveraged_return': pnl_pct * self.params.leverage * 100,
            'exit_reason': exit_reason,
            'duration_minutes': duration,
            'signal_quality': position.get('signal_quality', 0),
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': self.daily_pnl,
            'parameters': position['parameters']
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        # Store in database
        self._store_trade(trade)
        
        # Enhanced logging
        logger.info(f"💰 CLOSED {position['side'].upper()}: ${pnl_amount:+.2f} ({pnl_pct*100:+.2f}%)")
        logger.info(f"   📊 Leveraged Return: {pnl_pct*self.params.leverage*100:+.1f}% | Quality: {position.get('signal_quality', 0):.1f}")
        logger.info(f"   💰 Balance: ${self.balance:.2f} | Daily PnL: ${self.daily_pnl:+.2f}")
        logger.info(f"   🔄 Consecutive Losses: {self.consecutive_losses} | Reason: {exit_reason.upper()}")
        
        # Risk warnings
        if self.consecutive_losses >= self.params.max_consecutive_losses:
            logger.warning(f"⚠️ RISK WARNING: {self.consecutive_losses} consecutive losses - trading paused")
        
        if self.daily_pnl < -self.initial_balance * self.params.daily_loss_limit:
            logger.warning(f"⚠️ DAILY LOSS LIMIT: ${self.daily_pnl:.2f} - trading paused")
        
        # Log to WANDB with enhanced metrics
        if wandb.run:
            wandb.log({
                'trade_pnl': pnl_amount,
                'trade_pnl_pct': pnl_pct * 100,
                'leveraged_return': pnl_pct * self.params.leverage * 100,
                'balance_after_trade': self.balance,
                'trade_duration': duration,
                'exit_reason': exit_reason,
                'signal_quality': position.get('signal_quality', 0),
                'consecutive_losses': self.consecutive_losses,
                'daily_pnl': self.daily_pnl,
                'risk_score': self._calculate_risk_score()
            })
    
    def _calculate_risk_score(self):
        """Calculate current risk score (0-100, higher = more risky)"""
        risk_score = 0
        
        # Consecutive losses risk (0-30 points)
        risk_score += min(self.consecutive_losses * 15, 30)
        
        # Daily loss risk (0-25 points)
        if hasattr(self, 'daily_pnl'):
            daily_loss_pct = abs(self.daily_pnl) / self.initial_balance
            risk_score += min(daily_loss_pct * 100, 25)
        
        # Balance risk (0-25 points)
        balance_risk = (self.initial_balance - self.balance) / self.initial_balance
        risk_score += min(balance_risk * 100, 25)
        
        # Position concentration risk (0-20 points)
        if len(self.positions) >= self.params.max_positions:
            risk_score += 20
        elif len(self.positions) > 0:
            risk_score += 10
        
        return min(risk_score, 100)
    
    def _store_trade(self, trade):
        """Store trade in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (side, entry_price, exit_price, pnl_amount, pnl_pct, 
                                  exit_reason, duration_minutes, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['side'], trade['entry_price'], trade['exit_price'],
                trade['pnl_amount'], trade['pnl_pct'], trade['exit_reason'],
                trade['duration_minutes'], json.dumps(trade['parameters'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing trade: {e}")
    
    async def _trigger_optimization(self):
        """Trigger parameter optimization"""
        logger.info("🧠 Starting parameter optimization...")
        self.last_optimization = time.time()
        
        try:
            # Get recent performance data
            recent_trades = self.trades[-50:] if len(self.trades) >= 50 else self.trades
            
            if len(recent_trades) < 10:
                logger.info("📊 Not enough trades for optimization")
                return
            
            # Calculate current performance
            current_performance = self._calculate_performance_score(recent_trades)
            
            # Run optimization in background thread to avoid blocking
            optimization_thread = threading.Thread(
                target=self._run_optimization_sync,
                args=(recent_trades, current_performance)
            )
            optimization_thread.start()
            
        except Exception as e:
            logger.error(f"❌ Optimization error: {e}")
    
    def _calculate_performance_score(self, trades):
        """Calculate performance score for optimization"""
        if not trades:
            return 0
        
        total_pnl = sum(t['pnl_amount'] for t in trades)
        win_rate = len([t for t in trades if t['pnl_amount'] > 0]) / len(trades)
        avg_duration = np.mean([t['duration_minutes'] for t in trades])
        
        # Composite score: PnL + win rate bonus - duration penalty
        score = total_pnl + (win_rate * 100) - (avg_duration / 60)
        return score
    
    def _run_optimization_sync(self, recent_trades, current_performance):
        """Run research-based optimization in separate thread"""
        try:
            # Calculate detailed performance metrics
            win_rate = len([t for t in recent_trades if t['pnl_amount'] > 0]) / len(recent_trades)
            avg_return = np.mean([t['leveraged_return'] for t in recent_trades])
            sharpe_ratio = self._calculate_sharpe_ratio(recent_trades)
            max_drawdown = self._calculate_max_drawdown(recent_trades)
            
            logger.info(f"📊 Performance Analysis: Win Rate: {win_rate:.1%}, Avg Return: {avg_return:.1f}%, Sharpe: {sharpe_ratio:.2f}")
            
            # Research-based parameter optimization
            if win_rate < 0.4:  # Low win rate
                # Increase signal quality requirements
                self.params.signal_quality_min = min(80, self.params.signal_quality_min + 5)
                self.params.momentum_threshold = min(0.005, self.params.momentum_threshold * 1.1)
                self.params.volume_multiplier = min(2.5, self.params.volume_multiplier * 1.1)
                logger.info("📈 Low win rate - increasing signal quality requirements")
                
            elif win_rate > 0.7:  # High win rate but maybe missing opportunities
                # Slightly relax signal requirements
                self.params.signal_quality_min = max(60, self.params.signal_quality_min - 2)
                self.params.momentum_threshold = max(0.002, self.params.momentum_threshold * 0.95)
                logger.info("📊 High win rate - slightly relaxing signal requirements")
            
            # Risk management adjustments based on drawdown
            if max_drawdown > 0.15:  # High drawdown
                # Reduce risk
                self.params.leverage = max(8, self.params.leverage * 0.9)
                self.params.position_size_pct = max(0.05, self.params.position_size_pct * 0.9)
                self.params.max_consecutive_losses = max(1, self.params.max_consecutive_losses - 1)
                logger.info("🛡️ High drawdown - reducing risk parameters")
                
            elif max_drawdown < 0.05 and sharpe_ratio > 1.5:  # Low drawdown, good Sharpe
                # Slightly increase risk for better returns
                self.params.leverage = min(15, self.params.leverage * 1.05)
                self.params.position_size_pct = min(0.12, self.params.position_size_pct * 1.05)
                logger.info("🚀 Low drawdown, good Sharpe - slightly increasing risk")
            
            # Adaptive stop loss based on volatility
            recent_volatility = self._calculate_recent_volatility()
            if recent_volatility > 0.03:  # High volatility
                self.params.stop_loss_pct = min(0.025, self.params.stop_loss_pct * 1.1)
                logger.info("📈 High volatility - widening stop loss")
            elif recent_volatility < 0.015:  # Low volatility
                self.params.stop_loss_pct = max(0.01, self.params.stop_loss_pct * 0.95)
                logger.info("📉 Low volatility - tightening stop loss")
            
            # Store optimization result
            self._store_optimization_result(current_performance)
            
            # Log comprehensive metrics to WANDB
            if wandb.run:
                wandb.log({
                    'optimization_score': current_performance,
                    'win_rate': win_rate,
                    'avg_leveraged_return': avg_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'recent_volatility': recent_volatility,
                    'leverage': self.params.leverage,
                    'position_size_pct': self.params.position_size_pct,
                    'stop_loss_pct': self.params.stop_loss_pct,
                    'signal_quality_min': self.params.signal_quality_min,
                    'momentum_threshold': self.params.momentum_threshold,
                    'volume_multiplier': self.params.volume_multiplier
                })
            
            logger.info(f"✅ Research-based optimization complete")
            logger.info(f"📊 Updated: Leverage={self.params.leverage:.1f}x, Size={self.params.position_size_pct:.1%}")
            logger.info(f"🎯 Signal Quality Min: {self.params.signal_quality_min:.0f}, Momentum: {self.params.momentum_threshold:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Optimization thread error: {e}")
    
    def _calculate_sharpe_ratio(self, trades):
        """Calculate Sharpe ratio for recent trades"""
        if len(trades) < 5:
            return 0
        
        returns = [t['leveraged_return'] / 100 for t in trades]  # Convert to decimal
        return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    def _calculate_max_drawdown(self, trades):
        """Calculate maximum drawdown"""
        if not trades:
            return 0
        
        cumulative_returns = np.cumsum([t['pnl_amount'] for t in trades])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / self.initial_balance
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0
    
    def _calculate_recent_volatility(self):
        """Calculate recent price volatility"""
        if len(self.price_history) < 20:
            return 0.02  # Default volatility
        
        returns = np.diff(self.price_history[-20:]) / self.price_history[-21:-1]
        return np.std(returns) if len(returns) > 0 else 0.02
    
    def _store_optimization_result(self, performance_score):
        """Store optimization result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            win_rate = 0
            total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
            
            if self.trades:
                winning_trades = len([t for t in self.trades if t['pnl_amount'] > 0])
                win_rate = (winning_trades / len(self.trades)) * 100
            
            cursor.execute('''
                INSERT INTO parameter_history (parameters, performance_score, total_trades, win_rate, total_return)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                json.dumps(self.params.to_dict()),
                performance_score,
                len(self.trades),
                win_rate,
                total_return
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing optimization result: {e}")
    
    async def start_continuous_learning(self):
        """Start the continuous learning bot"""
        logger.info("🚀 Starting Continuous Learning Trading Bot")
        logger.info("=" * 60)
        
        self.running = True
        
        # Connect to WebSocket
        connected = await self.connect_websocket()
        if not connected:
            logger.error("❌ Failed to connect - exiting")
            return False
        
        try:
            # Start main trading loop
            await self.listen_websocket()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Stopping bot (Ctrl+C)")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            await self._shutdown()
        
        return True
    
    async def _shutdown(self):
        """Graceful shutdown"""
        logger.info("🔄 Shutting down...")
        self.running = False
        
        if self.ws_connection:
            await self.ws_connection.close()
        
        # Close positions
        if self.positions:
            current_price = self.price_history[-1] if self.price_history else 0
            for position in self.positions[:]:
                await self._close_position(position, current_price, "shutdown")
        
        # Save final state
        self._save_state()
        
        # Final report
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        logger.info(f"📊 Final Balance: ${self.balance:.2f} ({total_return:+.2f}%)")
        logger.info(f"🎯 Total Trades: {len(self.trades)}")
        
        if wandb.run:
            wandb.finish()

# Railway health check endpoint
async def health_check_server():
    """Simple health check server for Railway"""
    from aiohttp import web
    
    async def health(request):
        return web.json_response({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    app = web.Application()
    app.router.add_get('/health', health)
    app.router.add_get('/', health)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    port = int(os.getenv('PORT', 8080))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"🌐 Health check server running on port {port}")

async def main():
    """Main function for Railway deployment"""
    logger.info("🚀 RAILWAY CONTINUOUS LEARNING TRADING BOT")
    logger.info("=" * 60)
    
    # Start health check server for Railway
    await health_check_server()
    
    # Initialize and start bot
    bot = ContinuousLearningBot()
    
    # Run bot
    await bot.start_continuous_learning()

if __name__ == "__main__":
    asyncio.run(main())