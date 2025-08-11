#!/usr/bin/env python3
"""
Live Balanced Optimized Bot - Running with real-time SOONUSDT data
Based on our best-performing configuration (+9.86% return in backtesting)
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveBalancedBot:
    """Live trading bot with proven balanced parameters"""
    
    def __init__(self, initial_balance=1000.0, paper_trading=True):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.paper_trading = paper_trading
        self.positions = []
        self.trades = []
        
        # PROVEN OPTIMAL PARAMETERS (from balanced bot testing: +9.86% return)
        self.params = {
            'leverage': 15,                    # Optimal from testing
            'position_size_pct': 0.15,        # 15% - balanced risk/reward
            'stop_loss_pct': 0.020,           # 2.0% - not too tight
            'take_profit_pct': 0.035,         # 3.5% - better R:R
            'momentum_threshold': 0.002,      # Balanced momentum
            'volume_multiplier': 1.4,         # Reasonable volume confirmation
            'rsi_lower': 30,
            'rsi_upper': 70,
            'max_positions': 1,               # Single position for safety
            'min_balance': 100                # Emergency stop
        }
        
        # Market data
        self.symbol = "SOONUSDT"
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        self.rest_url = "https://api.bybit.com"
        
        # Data storage
        self.price_history = []
        self.volume_history = []
        self.rsi_history = []
        self.kline_buffer = []
        
        # Connection state
        self.ws_connection = None
        self.running = False
        self.last_kline_time = 0
        
        logger.info("🚀 Live Balanced Bot Initialized")
        logger.info(f"📊 Proven Parameters: {self.params}")
        logger.info(f"💰 Starting Balance: ${self.balance:.2f}")
        logger.info(f"📈 Paper Trading: {'ON' if self.paper_trading else 'OFF'}")
    
    async def connect_websocket(self):
        """Connect to Bybit WebSocket"""
        try:
            logger.info(f"🔌 Connecting to {self.ws_url}")
            self.ws_connection = await asyncio.wait_for(
                websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10),
                timeout=10.0
            )
            
            # Subscribe to 3-minute klines (matches our backtesting)
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"kline.3.{self.symbol}"]
            }
            
            await self.ws_connection.send(json.dumps(subscribe_msg))
            logger.info(f"✅ Connected and subscribed to {self.symbol} 3-minute klines")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def listen_websocket(self):
        """Listen for WebSocket messages"""
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                if 'topic' in data and data['topic'].startswith('kline'):
                    await self._process_kline_data(data)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ Error in WebSocket listener: {e}")
    
    async def _process_kline_data(self, data: Dict):
        """Process incoming kline data"""
        try:
            kline_data = data['data'][0]
            
            # Extract kline information
            kline = {
                'timestamp': int(kline_data['timestamp']),
                'open': float(kline_data['open']),
                'high': float(kline_data['high']),
                'low': float(kline_data['low']),
                'close': float(kline_data['close']),
                'volume': float(kline_data['volume']),
                'confirm': kline_data['confirm']  # True when kline is closed
            }
            
            # Only process confirmed (closed) klines to avoid reprocessing
            if kline['confirm'] and kline['timestamp'] != self.last_kline_time:
                self.last_kline_time = kline['timestamp']
                await self._process_new_candle(kline)
                
        except Exception as e:
            logger.error(f"❌ Error processing kline data: {e}")
    
    async def _process_new_candle(self, kline: Dict):
        """Process new confirmed candle"""
        price = kline['close']
        volume = kline['volume']
        timestamp = datetime.fromtimestamp(kline['timestamp'] / 1000)
        
        logger.info(f"📊 NEW CANDLE: {self.symbol} @ ${price:.4f} | Vol: {volume:.0f} | {timestamp}")
        
        # Update price and volume history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Keep history manageable (last 100 candles)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
            self.volume_history = self.volume_history[-100:]
        
        # Calculate RSI
        if len(self.price_history) >= 15:
            rsi = self._calculate_rsi(self.price_history)
            self.rsi_history.append(rsi)
            if len(self.rsi_history) > 100:
                self.rsi_history = self.rsi_history[-100:]
        
        # Check for trading signals (only if we have enough data)
        if len(self.price_history) >= 25:
            await self._check_trading_signals(kline)
        
        # Manage existing positions
        await self._manage_positions(price)
        
        # Report status every 10 candles
        if len(self.price_history) % 10 == 0:
            await self._report_status()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
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
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def _check_trading_signals(self, kline: Dict):
        """Check for trading signals using proven balanced logic"""
        if len(self.positions) >= self.params['max_positions']:
            return
        
        price = kline['close']
        volume = kline['volume']
        
        # Technical indicators
        sma_10 = np.mean(self.price_history[-10:])
        sma_20 = np.mean(self.price_history[-20:])
        
        # Momentum calculation
        momentum = 0
        if len(self.price_history) >= 5:
            momentum = (price - self.price_history[-5]) / self.price_history[-5]
        
        # Volume confirmation
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_spike = volume > avg_volume * self.params['volume_multiplier']
        
        # RSI filter
        rsi = self.rsi_history[-1] if self.rsi_history else 50
        rsi_ok = self.params['rsi_lower'] < rsi < self.params['rsi_upper']
        
        # Balanced signal logic (proven from backtesting)
        bullish_signal = (
            (price > sma_10 and sma_10 > sma_20 and momentum > self.params['momentum_threshold']) and
            (volume_spike or abs(momentum) > self.params['momentum_threshold'] * 2) and
            rsi_ok
        )
        
        bearish_signal = (
            (price < sma_10 and sma_10 < sma_20 and momentum < -self.params['momentum_threshold']) and
            (volume_spike or abs(momentum) > self.params['momentum_threshold'] * 2) and
            rsi_ok
        )
        
        if bullish_signal:
            await self._open_position('long', price, momentum, rsi, volume_spike)
        elif bearish_signal:
            await self._open_position('short', price, momentum, rsi, volume_spike)
    
    async def _open_position(self, side, price, momentum, rsi, volume_spike):
        """Open position with balanced parameters"""
        # Calculate position size
        position_size = self.balance * self.params['position_size_pct']
        
        # Calculate stop loss and take profit
        if side == 'long':
            stop_loss = price * (1 - self.params['stop_loss_pct'])
            take_profit = price * (1 + self.params['take_profit_pct'])
        else:
            stop_loss = price * (1 + self.params['stop_loss_pct'])
            take_profit = price * (1 - self.params['take_profit_pct'])
        
        position = {
            'id': f"{side}_{len(self.trades)}_{int(time.time())}",
            'side': side,
            'entry_price': price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'leverage': self.params['leverage'],
            'momentum': momentum,
            'rsi': rsi,
            'volume_spike': volume_spike
        }
        
        self.positions.append(position)
        
        logger.info(f"🔥 OPENED {side.upper()} POSITION")
        logger.info(f"   💰 Entry: ${price:.4f}")
        logger.info(f"   🛑 Stop Loss: ${stop_loss:.4f}")
        logger.info(f"   🎯 Take Profit: ${take_profit:.4f}")
        logger.info(f"   📊 Size: ${position_size:.2f} ({self.params['leverage']}x leverage)")
        logger.info(f"   📈 Momentum: {momentum:.4f} | RSI: {rsi:.1f} | Volume Spike: {volume_spike}")
    
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
            else:  # short
                if current_price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif current_price <= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position, current_price, exit_reason))
        
        # Close positions
        for position, exit_price, exit_reason in positions_to_close:
            await self._close_position(position, exit_price, exit_reason)
    
    async def _close_position(self, position, exit_price, exit_reason):
        """Close position and record trade"""
        # Calculate PnL
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        # Apply leverage
        pnl_amount = position['size'] * pnl_pct * position['leverage']
        self.balance += pnl_amount
        
        # Calculate trade duration
        duration = (datetime.now() - position['entry_time']).total_seconds() / 60  # minutes
        
        # Record trade
        trade = {
            'id': position['id'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct * 100,
            'exit_reason': exit_reason,
            'leverage': position['leverage'],
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'duration_minutes': duration,
            'entry_momentum': position['momentum'],
            'entry_rsi': position['rsi'],
            'volume_spike': position['volume_spike']
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        # Log trade closure
        logger.info(f"💰 CLOSED {position['side'].upper()} POSITION")
        logger.info(f"   📈 Entry: ${position['entry_price']:.4f} → Exit: ${exit_price:.4f}")
        logger.info(f"   💵 PnL: ${pnl_amount:.2f} ({pnl_pct*100:.2f}%)")
        logger.info(f"   🔄 Reason: {exit_reason.upper()}")
        logger.info(f"   ⏱️ Duration: {duration:.1f} minutes")
        logger.info(f"   💰 New Balance: ${self.balance:.2f}")
        
        # Check for emergency stop
        if self.balance < self.params['min_balance']:
            logger.error(f"🛑 EMERGENCY STOP: Balance below ${self.params['min_balance']}")
            await self._emergency_stop()
    
    async def _emergency_stop(self):
        """Emergency stop all trading"""
        logger.error("🚨 EMERGENCY STOP ACTIVATED")
        self.running = False
        
        # Close all positions at market price
        if self.positions:
            current_price = self.price_history[-1] if self.price_history else 0
            for position in self.positions[:]:
                await self._close_position(position, current_price, "emergency_stop")
    
    async def _report_status(self):
        """Report current status"""
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate win rate
        if self.trades:
            winning_trades = len([t for t in self.trades if t['pnl_amount'] > 0])
            win_rate = (winning_trades / len(self.trades)) * 100
        else:
            win_rate = 0
        
        logger.info("📊 STATUS REPORT")
        logger.info(f"   💰 Balance: ${self.balance:.2f} (Return: {total_return:+.2f}%)")
        logger.info(f"   📈 Trades: {len(self.trades)} | Win Rate: {win_rate:.1f}%")
        logger.info(f"   🔄 Open Positions: {len(self.positions)}")
        logger.info(f"   📊 Price History: {len(self.price_history)} candles")
        
        if self.price_history:
            current_price = self.price_history[-1]
            rsi = self.rsi_history[-1] if self.rsi_history else 50
            logger.info(f"   💹 Current: ${current_price:.4f} | RSI: {rsi:.1f}")
    
    async def start_live_trading(self):
        """Start live trading"""
        logger.info("🚀 Starting Live Balanced Bot")
        logger.info("=" * 60)
        
        self.running = True
        
        # Connect to WebSocket
        connected = await self.connect_websocket()
        if not connected:
            logger.error("❌ Failed to connect to WebSocket")
            return False
        
        try:
            # Start listening for data
            await self.listen_websocket()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Stopping bot (Ctrl+C pressed)")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            await self._shutdown()
        
        return True
    
    async def _shutdown(self):
        """Graceful shutdown"""
        logger.info("🔄 Shutting down...")
        self.running = False
        
        # Close WebSocket connection
        if self.ws_connection:
            await self.ws_connection.close()
        
        # Close any open positions
        if self.positions:
            current_price = self.price_history[-1] if self.price_history else 0
            logger.info(f"🔄 Closing {len(self.positions)} open positions...")
            for position in self.positions[:]:
                await self._close_position(position, current_price, "shutdown")
        
        # Final report
        await self._final_report()
    
    async def _final_report(self):
        """Generate final trading report"""
        if not self.trades:
            logger.info("📊 No trades executed during session")
            return
        
        # Calculate metrics
        winning_trades = [t for t in self.trades if t['pnl_amount'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_amount'] < 0]
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        win_rate = len(winning_trades) / len(self.trades) * 100
        avg_win = np.mean([t['pnl_amount'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0
        avg_duration = np.mean([t['duration_minutes'] for t in self.trades])
        
        logger.info("\n" + "=" * 60)
        logger.info("🏆 LIVE TRADING SESSION RESULTS")
        logger.info("=" * 60)
        logger.info(f"💰 Final Balance: ${self.balance:.2f}")
        logger.info(f"📈 Total Return: {total_return:+.2f}%")
        logger.info(f"🎯 Total Trades: {len(self.trades)}")
        logger.info(f"✅ Win Rate: {win_rate:.1f}%")
        logger.info(f"🏅 Winning Trades: {len(winning_trades)}")
        logger.info(f"❌ Losing Trades: {len(losing_trades)}")
        logger.info(f"💵 Average Win: ${avg_win:.2f}")
        logger.info(f"💸 Average Loss: ${avg_loss:.2f}")
        logger.info(f"⏱️ Average Duration: {avg_duration:.1f} minutes")
        logger.info("=" * 60)
        
        # Performance assessment
        if total_return > 5:
            logger.info("🚀 EXCELLENT live performance!")
        elif total_return > 0:
            logger.info("✅ POSITIVE live performance!")
        else:
            logger.info("📉 Consider adjusting parameters")

async def main():
    """Main function to run live bot"""
    print("🤖 LIVE BALANCED OPTIMIZED BOT")
    print("=" * 60)
    print("Running with proven parameters from backtesting (+9.86% return)")
    print("Using real-time SOONUSDT data from Bybit")
    print()
    print("⚠️  PAPER TRADING MODE - No real money at risk")
    print("Press Ctrl+C to stop the bot")
    print("=" * 60)
    
    # Initialize bot
    bot = LiveBalancedBot(
        initial_balance=1000.0,
        paper_trading=True  # Safe mode
    )
    
    # Start live trading
    await bot.start_live_trading()

if __name__ == "__main__":
    asyncio.run(main())