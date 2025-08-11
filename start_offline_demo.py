#!/usr/bin/env python3
"""
Offline demo using historical data to simulate real-time trading
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class HistoricalDataSimulator:
    """Simulate real-time data using historical CSV data"""
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.callbacks = []
        self.running = False
        self.data_index = 0
        
        # Load historical data
        self.load_data()
    
    def load_data(self):
        """Load historical data from CSV"""
        try:
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(self.df)} historical candles from {self.csv_file}")
            
            # Ensure we have the required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in self.df.columns:
                    logger.error(f"Missing required column: {col}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def add_data_callback(self, callback):
        self.callbacks.append(callback)
    
    async def simulate_realtime_data(self):
        """Stream historical data as if it were real-time"""
        logger.info("Starting historical data simulation...")
        
        while self.running and self.data_index < len(self.df):
            try:
                # Get current candle
                row = self.df.iloc[self.data_index]
                
                # Convert to our format
                kline_data = {
                    'symbol': 'SOONUSDT',
                    'timestamp': int(time.time() * 1000),  # Current timestamp
                    'open_time': int(time.time() * 1000) - 180000,
                    'close_time': int(time.time() * 1000),
                    'open_price': float(row['open']),
                    'high_price': float(row['high']),
                    'low_price': float(row['low']),
                    'close_price': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in row else 10000,
                    'turnover': float(row['close']) * float(row['volume']) if 'volume' in row else 5000,
                    'interval': '3m'
                }
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        await callback('kline', kline_data)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")
                
                logger.info(f"Candle {self.data_index + 1}/{len(self.df)}: SOONUSDT @ ${kline_data['close_price']:.4f}")
                
                self.data_index += 1
                
                # Wait 2 seconds between candles (speed up simulation)
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in simulation: {e}")
                await asyncio.sleep(1)
    
    async def start_collection(self):
        self.running = True
        await self.simulate_realtime_data()
    
    def stop_collection(self):
        self.running = False

class OfflineTradingBot:
    """Simplified trading bot for offline demo"""
    
    def __init__(self, data_simulator):
        self.data_simulator = data_simulator
        self.balance = 1000.0
        self.positions = []
        self.trades = []
        
        # Trading parameters
        self.params = {
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03,
            'position_size_pct': 0.1,
            'leverage': 10
        }
        
        # Price history for signals
        self.price_history = []
        self.volume_history = []
        
        # Add callback
        self.data_simulator.add_data_callback(self._on_new_data)
    
    async def _on_new_data(self, data_type, data):
        """Process new market data"""
        if data_type == 'kline':
            price = data['close_price']
            volume = data['volume']
            
            self.price_history.append(price)
            self.volume_history.append(volume)
            
            # Keep only last 50 candles
            if len(self.price_history) > 50:
                self.price_history = self.price_history[-50:]
                self.volume_history = self.volume_history[-50:]
            
            # Check for trading signals
            await self._check_signals(data)
            
            # Manage existing positions
            await self._manage_positions(data)
    
    async def _check_signals(self, data):
        """Check for trading signals using technical analysis"""
        if len(self.price_history) < 20:
            return
        
        price = data['close_price']
        volume = data['volume']
        
        # Calculate moving averages
        sma_10 = np.mean(self.price_history[-10:])
        sma_20 = np.mean(self.price_history[-20:])
        
        # Calculate RSI
        rsi = self._calculate_rsi()
        
        # Volume analysis
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_spike = volume > avg_volume * 1.5
        
        # Bullish signal
        if (price > sma_10 and sma_10 > sma_20 and 
            rsi > 40 and rsi < 70 and volume_spike and 
            len(self.positions) == 0):
            await self._open_position('long', price)
        
        # Bearish signal
        elif (price < sma_10 and sma_10 < sma_20 and 
              rsi > 30 and rsi < 60 and volume_spike and 
              len(self.positions) == 0):
            await self._open_position('short', price)
    
    def _calculate_rsi(self, period=14):
        """Calculate RSI"""
        if len(self.price_history) < period + 1:
            return 50  # Neutral RSI
        
        prices = np.array(self.price_history[-period-1:])
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
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
        
        logger.info(f"🔥 OPENED {side.upper()} @ ${price:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
    
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
        
        logger.info(f"💰 CLOSED {position['side'].upper()} @ ${exit_price:.4f} | PnL: ${pnl_amount:.2f} ({pnl_pct*100:.2f}%) | Balance: ${self.balance:.2f}")
    
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

async def main():
    """Main offline demo function"""
    print("📊 OFFLINE REAL-TIME TRADING DEMO")
    print("=" * 50)
    print("Using historical SOONUSDT data to simulate real-time trading")
    print("Press Ctrl+C to stop")
    print()
    
    # Find available CSV files
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'SOONUSDT' in f]
    
    if not csv_files:
        logger.error("No SOONUSDT CSV files found!")
        logger.info("Available files:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                logger.info(f"  - {f}")
        return
    
    # Use the first available file
    csv_file = csv_files[0]
    logger.info(f"Using data file: {csv_file}")
    
    # Initialize components
    data_simulator = HistoricalDataSimulator(csv_file)
    trading_bot = OfflineTradingBot(data_simulator)
    
    # Performance reporting
    async def performance_reporter():
        while data_simulator.running:
            await asyncio.sleep(15)  # Report every 15 seconds
            
            summary = trading_bot.get_performance_summary()
            
            print(f"\n📊 PERFORMANCE UPDATE:")
            print(f"   Balance: ${summary['balance']:.2f} ({summary['total_return']:+.2f}%)")
            print(f"   Trades: {summary['total_trades']} | Win Rate: {summary['win_rate']:.1f}%")
            if summary['total_trades'] > 0:
                print(f"   Avg PnL: ${summary['avg_pnl']:.2f} | Best: ${summary['best_trade']:.2f} | Worst: ${summary['worst_trade']:.2f}")
            print(f"   Active Positions: {len(trading_bot.positions)}")
            print(f"   Progress: {data_simulator.data_index}/{len(data_simulator.df)} candles")
    
    try:
        # Start performance reporting
        reporter_task = asyncio.create_task(performance_reporter())
        
        # Start simulation
        await data_simulator.start_collection()
        
    except KeyboardInterrupt:
        print("\n\nStopping offline demo...")
        data_simulator.stop_collection()
        
        # Final summary
        final_summary = trading_bot.get_performance_summary()
        
        print("\n" + "=" * 50)
        print("📈 FINAL OFFLINE DEMO RESULTS:")
        print(f"   Final Balance: ${final_summary['balance']:.2f}")
        print(f"   Total Return: {final_summary['total_return']:+.2f}%")
        print(f"   Total Trades: {final_summary['total_trades']}")
        if final_summary['total_trades'] > 0:
            print(f"   Win Rate: {final_summary['win_rate']:.1f}%")
            print(f"   Average PnL: ${final_summary['avg_pnl']:.2f}")
        print(f"   Processed: {data_simulator.data_index}/{len(data_simulator.df)} candles")
        print()
        print("🎯 This offline demo shows how the real system works:")
        print("   ✓ Technical analysis signals (SMA, RSI, Volume)")
        print("   ✓ Automatic position management")
        print("   ✓ Risk management with stop losses")
        print("   ✓ Real-time performance tracking")
        print("   ✓ Historical data simulation")

if __name__ == "__main__":
    asyncio.run(main())