#!/usr/bin/env python3
"""
Simplified Railway trading bot - minimal dependencies
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import os
import sqlite3
import threading
from typing import Dict, List, Optional
import requests
from dataclasses import dataclass, asdict

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

class SimpleTradingBot:
    """Simplified trading bot for Railway deployment"""
    
    def __init__(self):
        # Railway environment setup
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
        self.port = int(os.getenv('PORT', 8080))
        
        # Trading state
        self.initial_balance = float(os.getenv('INITIAL_BALANCE', 1000))
        self.balance = self.initial_balance
        self.positions = []
        self.trades = []
        
        # Parameters
        self.params = TradingParameters()
        self.optimization_interval = int(os.getenv('OPTIMIZATION_INTERVAL', 3600))
        self.last_optimization = 0
        
        # Market data
        self.symbol = "SOONUSDT"
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        
        # Data storage
        self.price_history = []
        self.volume_history = []
        self.rsi_history = []
        
        # Connection state
        self.ws_connection = None
        self.running = False
        self.last_kline_time = 0
        
        logger.info("🚀 Simple Trading Bot Initialized for Railway")
        logger.info(f"💰 Starting Balance: ${self.balance:.2f}")
        logger.info(f"🌐 Railway Environment: {self.is_railway}")
    
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
                    retry_delay *= 2
        
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
            
            # Only process confirmed klines
            if kline['confirm'] and kline['timestamp'] != self.last_kline_time:
                self.last_kline_time = kline['timestamp']
                await self._process_new_candle(kline)
                
                # Check if it's time to optimize
                if time.time() - self.last_optimization > self.optimization_interval:
                    await self._trigger_optimization()
                
        except Exception as e:
            logger.error(f"❌ Error processing kline: {e}")
    
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
        
        # Save state periodically
        if len(self.price_history) % 20 == 0:
            logger.info(f"💰 Balance: ${self.balance:.2f} | Trades: {len(self.trades)} | Positions: {len(self.positions)}")
    
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
            'entry_time': datetime.now()
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
            'duration_minutes': duration
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        logger.info(f"💰 CLOSED {position['side'].upper()}: ${pnl_amount:.2f} ({pnl_pct*100:.2f}%) | Balance: ${self.balance:.2f}")
    
    async def _trigger_optimization(self):
        """Simple parameter optimization"""
        logger.info("🧠 Starting parameter optimization...")
        self.last_optimization = time.time()
        
        try:
            recent_trades = self.trades[-20:] if len(self.trades) >= 20 else self.trades
            
            if len(recent_trades) < 5:
                logger.info("📊 Not enough trades for optimization")
                return
            
            # Calculate performance
            total_pnl = sum(t['pnl_amount'] for t in recent_trades)
            win_rate = len([t for t in recent_trades if t['pnl_amount'] > 0]) / len(recent_trades)
            
            # Simple optimization logic
            if total_pnl < 0:
                # Poor performance - be more conservative
                self.params.leverage = max(10, self.params.leverage * 0.95)
                self.params.position_size_pct = max(0.1, self.params.position_size_pct * 0.95)
                logger.info("📉 Poor performance - reducing risk")
            elif total_pnl > 50:
                # Good performance - slightly more aggressive
                self.params.leverage = min(20, self.params.leverage * 1.02)
                self.params.position_size_pct = min(0.2, self.params.position_size_pct * 1.02)
                logger.info("📈 Good performance - increasing position size")
            
            logger.info(f"✅ Optimization complete - PnL: ${total_pnl:.2f}, Win Rate: {win_rate:.1%}")
            
        except Exception as e:
            logger.error(f"❌ Optimization error: {e}")
    
    async def start_trading(self):
        """Start the trading bot"""
        logger.info("🚀 Starting Simple Trading Bot")
        logger.info("=" * 60)
        
        self.running = True
        
        # Start health check server
        asyncio.create_task(self._health_server())
        
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
    
    async def _health_server(self):
        """Simple health check server for Railway"""
        from aiohttp import web
        
        async def health(request):
            return web.json_response({
                "status": "healthy",
                "balance": self.balance,
                "trades": len(self.trades),
                "positions": len(self.positions),
                "timestamp": datetime.now().isoformat()
            })
        
        app = web.Application()
        app.router.add_get('/health', health)
        app.router.add_get('/', health)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"🌐 Health server running on port {self.port}")
    
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
        
        # Final report
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        logger.info(f"📊 Final Balance: ${self.balance:.2f} ({total_return:+.2f}%)")
        logger.info(f"🎯 Total Trades: {len(self.trades)}")

async def main():
    """Main function for Railway deployment"""
    logger.info("🚀 RAILWAY SIMPLE TRADING BOT")
    logger.info("=" * 60)
    
    # Initialize and start bot
    bot = SimpleTradingBot()
    
    # Run bot
    await bot.start_trading()

if __name__ == "__main__":
    asyncio.run(main())