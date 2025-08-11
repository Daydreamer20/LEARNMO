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
import sqlite3
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
    """Trading parameters that can be optimized"""
    leverage: float = 15.0
    position_size_pct: float = 0.15
    stop_loss_pct: float = 0.020
    take_profit_pct: float = 0.035
    momentum_threshold: float = 0.002
    volume_multiplier: float = 1.4
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0
    max_positions: int = 1
    min_balance: float = 100.0
    
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
        """Check for trading signals"""
        if len(self.positions) >= self.params.max_positions:
            return
        
        price = kline['close']
        volume = kline['volume']
        
        # Technical analysis
        sma_10 = np.mean(self.price_history[-10:])
        sma_20 = np.mean(self.price_history[-20:])
        
        momentum = 0
        if len(self.price_history) >= 5:
            momentum = (price - self.price_history[-5]) / self.price_history[-5]
        
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_spike = volume > avg_volume * self.params.volume_multiplier
        
        rsi = self.rsi_history[-1] if self.rsi_history else 50
        rsi_ok = self.params.rsi_lower < rsi < self.params.rsi_upper
        
        # Signal logic
        bullish = (
            price > sma_10 and sma_10 > sma_20 and 
            momentum > self.params.momentum_threshold and
            (volume_spike or abs(momentum) > self.params.momentum_threshold * 2) and
            rsi_ok
        )
        
        bearish = (
            price < sma_10 and sma_10 < sma_20 and 
            momentum < -self.params.momentum_threshold and
            (volume_spike or abs(momentum) > self.params.momentum_threshold * 2) and
            rsi_ok
        )
        
        if bullish:
            await self._open_position('long', price)
        elif bearish:
            await self._open_position('short', price)
    
    async def _open_position(self, side, price):
        """Open trading position"""
        position_size = self.balance * self.params.position_size_pct
        
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
            'parameters': self.params.to_dict()
        }
        
        self.positions.append(position)
        
        logger.info(f"🔥 OPENED {side.upper()}: ${price:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
    
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
        """Close position and record trade"""
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        pnl_amount = position['size'] * pnl_pct * self.params.leverage
        self.balance += pnl_amount
        
        duration = (datetime.now() - position['entry_time']).total_seconds() / 60
        
        trade = {
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct * 100,
            'exit_reason': exit_reason,
            'duration_minutes': duration,
            'parameters': position['parameters']
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        # Store in database
        self._store_trade(trade)
        
        logger.info(f"💰 CLOSED {position['side'].upper()}: ${pnl_amount:.2f} ({pnl_pct*100:.2f}%) | Balance: ${self.balance:.2f}")
        
        # Log to WANDB
        if wandb.run:
            wandb.log({
                'trade_pnl': pnl_amount,
                'trade_pnl_pct': pnl_pct * 100,
                'balance_after_trade': self.balance,
                'trade_duration': duration,
                'exit_reason': exit_reason
            })
    
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
        """Run optimization in separate thread"""
        try:
            # Simple parameter adjustment based on recent performance
            if current_performance < 0:
                # Poor performance - make conservative adjustments
                self.params.leverage = max(10, self.params.leverage * 0.9)
                self.params.position_size_pct = max(0.1, self.params.position_size_pct * 0.9)
                self.params.stop_loss_pct = min(0.03, self.params.stop_loss_pct * 1.1)
                logger.info("📉 Poor performance - making conservative adjustments")
            
            elif current_performance > 50:
                # Good performance - slightly more aggressive
                self.params.leverage = min(20, self.params.leverage * 1.05)
                self.params.position_size_pct = min(0.2, self.params.position_size_pct * 1.05)
                logger.info("📈 Good performance - slightly more aggressive")
            
            # Store optimization result
            self._store_optimization_result(current_performance)
            
            # Log to WANDB
            if wandb.run:
                wandb.log({
                    'optimization_score': current_performance,
                    'leverage': self.params.leverage,
                    'position_size_pct': self.params.position_size_pct,
                    'stop_loss_pct': self.params.stop_loss_pct
                })
            
            logger.info(f"✅ Optimization complete - Score: {current_performance:.2f}")
            logger.info(f"📊 New parameters: Leverage={self.params.leverage:.1f}, Size={self.params.position_size_pct:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Optimization thread error: {e}")
    
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