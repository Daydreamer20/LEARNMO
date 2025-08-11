#!/usr/bin/env python3
"""
Demo of the real-time trading system with simulated data
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class SimulatedDataCollector:
    """Simulated data collector for demo purposes"""
    
    def __init__(self):
        self.callbacks = []
        self.running = False
        
        # Generate some realistic price data
        self.base_price = 0.5  # SOONUSDT around $0.50
        self.current_price = self.base_price
        self.timestamp = int(time.time() * 1000)
        
    def add_data_callback(self, callback):
        self.callbacks.append(callback)
    
    async def simulate_market_data(self):
        """Generate simulated market data"""
        while self.running:
            # Simulate price movement
            change = np.random.normal(0, 0.002)  # 0.2% volatility
            self.current_price *= (1 + change)
            
            # Keep price in reasonable range
            self.current_price = max(0.3, min(0.8, self.current_price))
            
            # Create kline data
            kline_data = {
                'symbol': 'SOONUSDT',
                'timestamp': self.timestamp,
                'open_time': self.timestamp - 180000,  # 3 minutes ago
                'close_time': self.timestamp,
                'open_price': self.current_price * (1 + np.random.normal(0, 0.001)),
                'high_price': self.current_price * (1 + abs(np.random.normal(0, 0.002))),
                'low_price': self.current_price * (1 - abs(np.random.normal(0, 0.002))),
                'close_price': self.current_price,
                'volume': np.random.uniform(10000, 50000),
                'turnover': self.current_price * np.random.uniform(10000, 50000),
                'interval': '3m'
            }
            
            # Trigger callbacks
            for callback in self.callbacks:
                try:
                    await callback('kline', kline_data)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
            
            self.timestamp += 180000  # Next 3-minute candle
            await asyncio.sleep(2)  # 2 seconds between updates for demo
    
    async def start_collection(self):
        self.running = True
        await self.simulate_market_data()
    
    def stop_collection(self):
        self.running = False

class DemoTradingBot:
    """Simplified trading bot for demo"""
    
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.balance = 1000.0
        self.positions = []
        self.trades = []
        
        # Current parameters (would be optimized in real system)
        self.params = {
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03,
            'position_size_pct': 0.1,
            'leverage': 10
        }
        
        # Add callback
        self.data_collector.add_data_callback(self._on_new_data)
        
        # Price history for signals
        self.price_history = []
    
    async def _on_new_data(self, data_type, data):
        """Process new market data"""
        if data_type == 'kline':
            price = data['close_price']
            self.price_history.append(price)
            
            # Keep only last 20 prices
            if len(self.price_history) > 20:
                self.price_history = self.price_history[-20:]
            
            # Check for trading signals
            await self._check_signals(data)
            
            # Manage existing positions
            await self._manage_positions(data)
    
    async def _check_signals(self, data):
        """Check for trading signals"""
        if len(self.price_history) < 10:
            return
        
        price = data['close_price']
        
        # Simple momentum signal
        recent_prices = self.price_history[-5:]
        older_prices = self.price_history[-10:-5]
        
        recent_avg = np.mean(recent_prices)
        older_avg = np.mean(older_prices)
        
        # Volume spike detection
        volume_spike = data['volume'] > 30000  # Arbitrary threshold
        
        # Bullish signal: recent prices higher than older + volume spike
        if recent_avg > older_avg * 1.002 and volume_spike and len(self.positions) == 0:
            await self._open_position('long', price)
        
        # Bearish signal: recent prices lower than older + volume spike  
        elif recent_avg < older_avg * 0.998 and volume_spike and len(self.positions) == 0:
            await self._open_position('short', price)
    
    async def _open_position(self, side, price):
        """Open a new position"""
        position_size = self.balance * self.params['position_size_pct']
        
        if side == 'long':
            stop_loss = price * (1 - self.params['stop_loss_pct'])
            take_profit = price * (1 + self.params['take_profit_pct'])
        else:
            stop_loss = price * (1 + self.params['stop_loss_pct'])
            take_profit = price * (1 - self.params['take_profit_pct'])
        
        position = {
            'id': f"{side}_{int(time.time())}",
            'side': side,
            'entry_price': price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now()
        }
        
        self.positions.append(position)
        
        logger.info(f"🔥 OPENED {side.upper()} position @ ${price:.4f}")
        logger.info(f"   Size: ${position_size:.2f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
    
    async def _manage_positions(self, data):
        """Manage existing positions"""
        price = data['close_price']
        positions_to_close = []
        
        for position in self.positions:
            should_close = False
            exit_reason = ""
            
            if position['side'] == 'long':
                if price <= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif price >= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            else:  # short
                if price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif price <= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position, price, exit_reason))
        
        # Close positions
        for position, exit_price, exit_reason in positions_to_close:
            await self._close_position(position, exit_price, exit_reason)
    
    async def _close_position(self, position, exit_price, exit_reason):
        """Close a position"""
        # Calculate PnL
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        pnl_amount = position['size'] * pnl_pct * self.params['leverage']
        self.balance += pnl_amount
        
        # Record trade
        trade = {
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'duration': datetime.now() - position['entry_time']
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        logger.info(f"💰 CLOSED {position['side'].upper()} position @ ${exit_price:.4f}")
        logger.info(f"   PnL: ${pnl_amount:.2f} ({pnl_pct*100:.2f}%) | Reason: {exit_reason}")
        logger.info(f"   Balance: ${self.balance:.2f}")
    
    def get_performance_summary(self):
        """Get performance summary"""
        if not self.trades:
            return {
                'balance': self.balance,
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0
            }
        
        winning_trades = [t for t in self.trades if t['pnl_amount'] > 0]
        
        return {
            'balance': self.balance,
            'total_return': (self.balance - 1000) / 1000 * 100,
            'total_trades': len(self.trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100,
            'avg_pnl': np.mean([t['pnl_amount'] for t in self.trades]),
            'best_trade': max([t['pnl_amount'] for t in self.trades]),
            'worst_trade': min([t['pnl_amount'] for t in self.trades])
        }

async def run_demo():
    """Run the demo"""
    print("🤖 REAL-TIME TRADING SYSTEM DEMO")
    print("=" * 50)
    print("Simulating live market data and trading...")
    print("Press Ctrl+C to stop")
    print()
    
    # Initialize components
    data_collector = SimulatedDataCollector()
    trading_bot = DemoTradingBot(data_collector)
    
    # Performance reporting task
    async def performance_reporter():
        while data_collector.running:
            await asyncio.sleep(10)  # Report every 10 seconds
            
            summary = trading_bot.get_performance_summary()
            
            print(f"\n📊 PERFORMANCE UPDATE:")
            print(f"   Balance: ${summary['balance']:.2f} ({summary['total_return']:+.2f}%)")
            print(f"   Trades: {summary['total_trades']} | Win Rate: {summary['win_rate']:.1f}%")
            if summary['total_trades'] > 0:
                print(f"   Avg PnL: ${summary['avg_pnl']:.2f} | Best: ${summary['best_trade']:.2f} | Worst: ${summary['worst_trade']:.2f}")
            print(f"   Active Positions: {len(trading_bot.positions)}")
            print(f"   Current Price: ${data_collector.current_price:.4f}")
    
    try:
        # Start performance reporting
        reporter_task = asyncio.create_task(performance_reporter())
        
        # Start data collection (this runs the main loop)
        await data_collector.start_collection()
        
    except KeyboardInterrupt:
        print("\n\nStopping demo...")
        data_collector.stop_collection()
        
        # Final summary
        final_summary = trading_bot.get_performance_summary()
        
        print("\n" + "=" * 50)
        print("📈 FINAL DEMO RESULTS:")
        print(f"   Final Balance: ${final_summary['balance']:.2f}")
        print(f"   Total Return: {final_summary['total_return']:+.2f}%")
        print(f"   Total Trades: {final_summary['total_trades']}")
        if final_summary['total_trades'] > 0:
            print(f"   Win Rate: {final_summary['win_rate']:.1f}%")
            print(f"   Average PnL: ${final_summary['avg_pnl']:.2f}")
        print()
        print("🎯 This demonstrates how the real system would:")
        print("   ✓ Collect real-time market data")
        print("   ✓ Generate trading signals")
        print("   ✓ Execute trades automatically")
        print("   ✓ Manage risk with stop losses")
        print("   ✓ Track performance in real-time")
        print("   ✓ Optimize parameters continuously")

if __name__ == "__main__":
    asyncio.run(run_demo())