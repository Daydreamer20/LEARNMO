#!/usr/bin/env python3
"""
Simple Windows-compatible real-time trading system
"""

import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import json
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class SimpleBybitDataCollector:
    """Simple REST API-based data collector for Windows"""
    
    def __init__(self, symbol="SOONUSDT", testnet=True):
        self.symbol = symbol
        self.testnet = testnet
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.callbacks = []
        self.running = False
        self.last_timestamp = 0
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect("simple_realtime.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp INTEGER,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_data_callback(self, callback):
        self.callbacks.append(callback)
    
    def fetch_latest_kline(self):
        """Fetch latest kline from Bybit REST API"""
        try:
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': self.symbol,
                'interval': '3',
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['retCode'] == 0 and data['result']['list']:
                    kline = data['result']['list'][0]
                    
                    kline_data = {
                        'symbol': self.symbol,
                        'timestamp': int(kline[0]),
                        'open_price': float(kline[1]),
                        'high_price': float(kline[2]),
                        'low_price': float(kline[3]),
                        'close_price': float(kline[4]),
                        'volume': float(kline[5]),
                        'turnover': float(kline[6])
                    }
                    
                    # Only process if it's a new candle
                    if kline_data['timestamp'] > self.last_timestamp:
                        self.last_timestamp = kline_data['timestamp']
                        
                        # Store in database
                        self._store_kline(kline_data)
                        
                        # Trigger callbacks
                        for callback in self.callbacks:
                            try:
                                callback('kline', kline_data)
                            except Exception as e:
                                logger.error(f"Error in callback: {e}")
                        
                        logger.info(f"New candle: {self.symbol} @ ${kline_data['close_price']:.4f}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return False
    
    def _store_kline(self, kline_data):
        """Store kline in database"""
        try:
            conn = sqlite3.connect("simple_realtime.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO klines (symbol, timestamp, open_price, high_price, 
                                  low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                kline_data['symbol'], kline_data['timestamp'],
                kline_data['open_price'], kline_data['high_price'],
                kline_data['low_price'], kline_data['close_price'],
                kline_data['volume']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing kline: {e}")
    
    async def start_collection(self):
        """Start data collection loop"""
        self.running = True
        logger.info(f"Starting data collection for {self.symbol}...")
        
        while self.running:
            try:
                # Fetch latest data
                self.fetch_latest_kline()
                
                # Wait 30 seconds (to avoid rate limits)
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def stop_collection(self):
        self.running = False

class SimpleTradingBot:
    """Simple trading bot for real-time data"""
    
    def __init__(self, data_collector):
        self.data_collector = data_collector
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
        
        # Price history
        self.price_history = []
        self.volume_history = []
        
        # Add callback
        self.data_collector.add_data_callback(self._on_new_data)
    
    def _on_new_data(self, data_type, data):
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
            self._check_signals(data)
            
            # Manage existing positions
            self._manage_positions(data)
    
    def _check_signals(self, data):
        """Check for trading signals"""
        if len(self.price_history) < 20:
            return
        
        price = data['close_price']
        volume = data['volume']
        
        # Simple moving averages
        sma_10 = np.mean(self.price_history[-10:])
        sma_20 = np.mean(self.price_history[-20:])
        
        # Volume analysis
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_spike = volume > avg_volume * 1.5
        
        # Price momentum
        momentum = (price - self.price_history[-5]) / self.price_history[-5] if len(self.price_history) >= 5 else 0
        
        # Bullish signal
        if (price > sma_10 and sma_10 > sma_20 and 
            momentum > 0.002 and volume_spike and 
            len(self.positions) == 0):
            self._open_position('long', price)
        
        # Bearish signal
        elif (price < sma_10 and sma_10 < sma_20 and 
              momentum < -0.002 and volume_spike and 
              len(self.positions) == 0):
            self._open_position('short', price)
    
    def _open_position(self, side, price):
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
    
    def _manage_positions(self, data):
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
            self._close_position(position, exit_price, exit_reason)
    
    def _close_position(self, position, exit_price, exit_reason):
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
    """Main function"""
    print("🤖 SIMPLE REAL-TIME TRADING SYSTEM")
    print("=" * 50)
    print("Using Bybit REST API for real-time SOONUSDT data")
    print("Press Ctrl+C to stop")
    print()
    
    # Test API connection first
    logger.info("Testing Bybit API connection...")
    try:
        response = requests.get("https://api-testnet.bybit.com/v5/market/time", timeout=10)
        if response.status_code == 200:
            logger.info("✓ Bybit API connection successful")
        else:
            logger.error("✗ Bybit API connection failed")
            return
    except Exception as e:
        logger.error(f"✗ API test failed: {e}")
        return
    
    # Initialize components
    data_collector = SimpleBybitDataCollector("SOONUSDT", testnet=True)
    trading_bot = SimpleTradingBot(data_collector)
    
    # Performance reporting
    async def performance_reporter():
        while data_collector.running:
            await asyncio.sleep(60)  # Report every minute
            
            summary = trading_bot.get_performance_summary()
            
            print(f"\n📊 PERFORMANCE UPDATE:")
            print(f"   Balance: ${summary['balance']:.2f} ({summary['total_return']:+.2f}%)")
            print(f"   Trades: {summary['total_trades']} | Win Rate: {summary['win_rate']:.1f}%")
            if summary['total_trades'] > 0:
                print(f"   Avg PnL: ${summary['avg_pnl']:.2f} | Best: ${summary['best_trade']:.2f} | Worst: ${summary['worst_trade']:.2f}")
            print(f"   Active Positions: {len(trading_bot.positions)}")
            print(f"   Price History: {len(trading_bot.price_history)} candles")
            if trading_bot.price_history:
                print(f"   Current Price: ${trading_bot.price_history[-1]:.4f}")
    
    try:
        # Start performance reporting
        reporter_task = asyncio.create_task(performance_reporter())
        
        # Start data collection
        await data_collector.start_collection()
        
    except KeyboardInterrupt:
        print("\n\nStopping simple real-time system...")
        data_collector.stop_collection()
        
        # Final summary
        final_summary = trading_bot.get_performance_summary()
        
        print("\n" + "=" * 50)
        print("📈 FINAL REAL-TIME RESULTS:")
        print(f"   Final Balance: ${final_summary['balance']:.2f}")
        print(f"   Total Return: {final_summary['total_return']:+.2f}%")
        print(f"   Total Trades: {final_summary['total_trades']}")
        if final_summary['total_trades'] > 0:
            print(f"   Win Rate: {final_summary['win_rate']:.1f}%")
            print(f"   Average PnL: ${final_summary['avg_pnl']:.2f}")
        print(f"   Candles Processed: {len(trading_bot.price_history)}")
        print()
        print("🎯 This simple system demonstrates:")
        print("   ✓ Real-time data from Bybit API")
        print("   ✓ Technical analysis signals")
        print("   ✓ Automatic trading execution")
        print("   ✓ Risk management")
        print("   ✓ Performance tracking")

if __name__ == "__main__":
    asyncio.run(main())