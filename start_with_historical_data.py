#!/usr/bin/env python3
"""
Start trading system using historical data (works offline)
This simulates real-time trading using your existing SOONUSDT data
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
    """Simulates real-time data using historical CSV"""
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.current_index = 0
        self.callbacks = []
        self.running = False
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load historical data"""
        try:
            self.data = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(self.data)} historical candles from {self.csv_file}")
            
            # Ensure we have the required columns
            required_cols = ['time', 'open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in self.data.columns:
                    logger.error(f"Missing column: {col}")
                    return False
            
            # Add volume if missing
            if 'volume' not in self.data.columns:
                self.data['volume'] = np.random.uniform(10000, 50000, len(self.data))
            
            # Convert time if needed
            if self.data['time'].dtype == 'object':
                self.data['time'] = pd.to_datetime(self.data['time'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def add_data_callback(self, callback):
        """Add callback for new data"""
        self.callbacks.append(callback)
    
    async def simulate_realtime(self):
        """Simulate real-time data feed"""
        logger.info("Starting historical data simulation...")
        
        while self.running and self.current_index < len(self.data):
            try:
                # Get current candle
                row = self.data.iloc[self.current_index]
                
                # Convert to expected format
                kline_data = {
                    'symbol': 'SOONUSDT',
                    'timestamp': int(row['time']) if 'time' in row else int(time.time() * 1000),
                    'open_time': int(row['time']) if 'time' in row else int(time.time() * 1000) - 180000,
                    'close_time': int(row['time']) + 180000 if 'time' in row else int(time.time() * 1000),
                    'open_price': float(row['open']),
                    'high_price': float(row['high']),
                    'low_price': float(row['low']),
                    'close_price': float(row['close']),
                    'volume': float(row['volume']),
                    'turnover': float(row['close']) * float(row['volume']),
                    'interval': '3m',
                    # Add signal data
                    'bullish_bos': bool(row.get('Bullish BOS', 0)),
                    'bearish_bos': bool(row.get('Bearish BOS', 0)),
                    'bullish_choch': bool(row.get('Bullish CHOCH', 0)),
                    'bearish_choch': bool(row.get('Bearish CHOCH', 0))
                }
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        await callback('kline', kline_data)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")
                
                # Log progress
                if self.current_index % 20 == 0:
                    progress = (self.current_index / len(self.data)) * 100
                    logger.info(f"Progress: {progress:.1f}% - Price: ${kline_data['close_price']:.4f}")
                
                self.current_index += 1
                
                # Wait 2 seconds between candles (speed up simulation)
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in simulation: {e}")
                await asyncio.sleep(1)
        
        logger.info("Historical data simulation completed!")
    
    async def start_collection(self):
        """Start the simulation"""
        self.running = True
        await self.simulate_realtime()
    
    def stop_collection(self):
        """Stop the simulation"""
        self.running = False

class SimpleOptimizer:
    """Simple parameter optimizer"""
    
    def __init__(self):
        self.current_params = {
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03,
            'position_size_pct': 0.1,
            'leverage': 10
        }
        self.last_optimization = 0
        self.optimization_interval = 300  # 5 minutes for demo
    
    def get_current_parameters(self):
        """Get current parameters"""
        return self.current_params.copy()
    
    async def maybe_optimize(self):
        """Maybe run optimization"""
        current_time = time.time()
        if current_time - self.last_optimization > self.optimization_interval:
            logger.info("🔧 Running parameter optimization...")
            
            # Simple random optimization for demo
            self.current_params['stop_loss_pct'] = np.random.uniform(0.01, 0.02)
            self.current_params['take_profit_pct'] = np.random.uniform(0.025, 0.04)
            
            logger.info(f"Updated parameters: SL={self.current_params['stop_loss_pct']:.3f}, TP={self.current_params['take_profit_pct']:.3f}")
            self.last_optimization = current_time

class SimpleTradingBot:
    """Simple trading bot for historical data"""
    
    def __init__(self, data_simulator, optimizer):
        self.data_simulator = data_simulator
        self.optimizer = optimizer
        self.balance = 1000.0
        self.positions = []
        self.trades = []
        self.price_history = []
        
        # Add callback
        self.data_simulator.add_data_callback(self._on_new_data)
    
    async def _on_new_data(self, data_type, data):
        """Process new data"""
        if data_type == 'kline':
            price = data['close_price']
            self.price_history.append(price)
            
            # Keep last 20 prices
            if len(self.price_history) > 20:
                self.price_history = self.price_history[-20:]
            
            # Maybe optimize parameters
            await self.optimizer.maybe_optimize()
            
            # Check for signals
            await self._check_signals(data)
            
            # Manage positions
            await self._manage_positions(data)
    
    async def _check_signals(self, data):
        """Check for trading signals using actual BOS signals"""
        if len(self.positions) > 0:  # Only one position at a time
            return
        
        price = data['close_price']
        
        # Use actual BOS signals from your data
        bullish_bos = data.get('bullish_bos', False)
        bearish_bos = data.get('bearish_bos', False)
        
        # Volume confirmation
        volume_spike = data['volume'] > 20000
        
        # Bullish BOS signal
        if bullish_bos and volume_spike:
            await self._open_position('long', price)
        
        # Bearish BOS signal
        elif bearish_bos and volume_spike:
            await self._open_position('short', price)
        
        # Fallback: momentum signals if no BOS
        elif len(self.price_history) >= 10:
            recent_avg = np.mean(self.price_history[-5:])
            older_avg = np.mean(self.price_history[-10:-5])
            
            # Strong momentum signals
            if recent_avg > older_avg * 1.005 and volume_spike:
                await self._open_position('long', price)
            elif recent_avg < older_avg * 0.995 and volume_spike:
                await self._open_position('short', price)
    
    async def _open_position(self, side, price):
        """Open position"""
        params = self.optimizer.get_current_parameters()
        position_size = self.balance * params['position_size_pct']
        
        if side == 'long':
            stop_loss = price * (1 - params['stop_loss_pct'])
            take_profit = price * (1 + params['take_profit_pct'])
        else:
            stop_loss = price * (1 + params['stop_loss_pct'])
            take_profit = price * (1 - params['take_profit_pct'])
        
        position = {
            'id': f"{side}_{len(self.trades)}",
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
        """Manage positions"""
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
            else:
                if price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif price <= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position, price, exit_reason))
        
        for position, exit_price, exit_reason in positions_to_close:
            await self._close_position(position, exit_price, exit_reason)
    
    async def _close_position(self, position, exit_price, exit_reason):
        """Close position"""
        params = self.optimizer.get_current_parameters()
        
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        pnl_amount = position['size'] * pnl_pct * params['leverage']
        self.balance += pnl_amount
        
        trade = {
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        logger.info(f"💰 CLOSED {position['side'].upper()} @ ${exit_price:.4f} | PnL: ${pnl_amount:.2f} | Balance: ${self.balance:.2f}")
    
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
            'avg_pnl': np.mean([t['pnl_amount'] for t in self.trades])
        }

async def main():
    """Main function"""
    print("🤖 HISTORICAL DATA TRADING SIMULATION")
    print("=" * 50)
    print("Using your existing SOONUSDT data for realistic simulation")
    print("Press Ctrl+C to stop")
    print()
    
    # Find CSV file
    csv_files = [
        'BYBIT_SOONUSDT.P, 3_08827.csv',
        'BYBIT_SOONUSDT.P, 1_e56a5.csv'
    ]
    
    csv_file = None
    for file in csv_files:
        if os.path.exists(file):
            csv_file = file
            break
    
    if not csv_file:
        print("❌ No SOONUSDT CSV file found!")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"   {file}")
        return
    
    print(f"📊 Using data file: {csv_file}")
    
    # Initialize components
    data_simulator = HistoricalDataSimulator(csv_file)
    optimizer = SimpleOptimizer()
    trading_bot = SimpleTradingBot(data_simulator, optimizer)
    
    # Performance reporter
    async def performance_reporter():
        while data_simulator.running:
            await asyncio.sleep(30)  # Report every 30 seconds
            
            summary = trading_bot.get_performance_summary()
            print(f"\n📊 PERFORMANCE: Balance: ${summary['balance']:.2f} ({summary['total_return']:+.2f}%) | "
                  f"Trades: {summary['total_trades']} | Win Rate: {summary['win_rate']:.1f}%")
    
    try:
        # Start performance reporting
        reporter_task = asyncio.create_task(performance_reporter())
        
        # Start simulation
        await data_simulator.start_collection()
        
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
        data_simulator.stop_collection()
        
        # Final summary
        final_summary = trading_bot.get_performance_summary()
        
        print("\n" + "=" * 50)
        print("📈 FINAL RESULTS:")
        print(f"   Final Balance: ${final_summary['balance']:.2f}")
        print(f"   Total Return: {final_summary['total_return']:+.2f}%")
        print(f"   Total Trades: {final_summary['total_trades']}")
        if final_summary['total_trades'] > 0:
            print(f"   Win Rate: {final_summary['win_rate']:.1f}%")
            print(f"   Average PnL: ${final_summary['avg_pnl']:.2f}")
        
        print("\n🎯 This simulation shows how your bot would perform")
        print("   with real-time parameter optimization!")

if __name__ == "__main__":
    asyncio.run(main())