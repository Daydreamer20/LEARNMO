#!/usr/bin/env python3
"""
Windows-compatible real-time trading system using synchronous requests
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
import threading
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class WindowsCompatibleDataCollector:
    """Windows-compatible data collector using synchronous requests"""
    
    def __init__(self, symbol="SOONUSDT", testnet=False):
        self.symbol = symbol
        self.testnet = testnet
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.callbacks = []
        self.running = False
        self.last_timestamp = 0
        
        # Initialize database
        self._init_database()
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingBot/1.0'
        })
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect("windows_realtime.db")
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
        """Fetch latest kline from Bybit REST API using synchronous requests"""
        try:
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': self.symbol,
                'interval': '3',
                'limit': 1
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
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
                        
                        # Trigger callbacks (run in thread to avoid blocking)
                        for callback in self.callbacks:
                            try:
                                # Run callback in thread pool
                                threading.Thread(
                                    target=self._run_callback,
                                    args=(callback, 'kline', kline_data),
                                    daemon=True
                                ).start()
                            except Exception as e:
                                logger.error(f"Error starting callback thread: {e}")
                        
                        logger.info(f"New candle: {self.symbol} @ ${kline_data['close_price']:.4f}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return False
    
    def _run_callback(self, callback, data_type, data):
        """Run callback in separate thread"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async callback
            if asyncio.iscoroutinefunction(callback):
                loop.run_until_complete(callback(data_type, data))
            else:
                callback(data_type, data)
                
        except Exception as e:
            logger.error(f"Error in callback: {e}")
        finally:
            try:
                loop.close()
            except:
                pass
    
    def _store_kline(self, kline_data):
        """Store kline in database"""
        try:
            conn = sqlite3.connect("windows_realtime.db")
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
    
    def start_collection_sync(self):
        """Start data collection loop (synchronous)"""
        self.running = True
        logger.info(f"Starting Windows-compatible data collection for {self.symbol}...")
        
        while self.running:
            try:
                # Fetch latest data
                self.fetch_latest_kline()
                
                # Wait 30 seconds (to avoid rate limits)
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def stop_collection(self):
        self.running = False

class WindowsTradingBot:
    """Windows-compatible trading bot"""
    
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
        
        # Performance tracking
        self.last_performance_report = time.time()
    
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
            self._check_signals(data)
            
            # Manage existing positions
            self._manage_positions(data)
            
            # Report performance periodically
            current_time = time.time()
            if current_time - self.last_performance_report > 300:  # Every 5 minutes
                self._report_performance()
                self.last_performance_report = current_time
    
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
        
        # Only trade if no active positions
        if len(self.positions) == 0:
            # Bullish signal
            if (price > sma_10 and sma_10 > sma_20 and 
                momentum > 0.003 and volume_spike):
                self._open_position('long', price)
            
            # Bearish signal
            elif (price < sma_10 and sma_10 < sma_20 and 
                  momentum < -0.003 and volume_spike):
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
    
    def _report_performance(self):
        """Report current performance"""
        if not self.trades:
            win_rate = 0
            avg_pnl = 0
        else:
            winning_trades = [t for t in self.trades if t['pnl_amount'] > 0]
            win_rate = len(winning_trades) / len(self.trades) * 100
            avg_pnl = np.mean([t['pnl_amount'] for t in self.trades])
        
        total_return = (self.balance - 1000) / 1000 * 100
        
        logger.info(f"📊 PERFORMANCE: Balance: ${self.balance:.2f} ({total_return:+.2f}%) | "
                   f"Trades: {len(self.trades)} | Win Rate: {win_rate:.1f}% | "
                   f"Active: {len(self.positions)}")
    
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
            'active_positions': len(self.positions)
        }

def main():
    """Main function"""
    print("🤖 WINDOWS-COMPATIBLE REAL-TIME TRADING SYSTEM")
    print("=" * 60)
    print("Using synchronous requests to avoid Windows async issues")
    print("Press Ctrl+C to stop")
    print()
    
    # Test API connection first
    logger.info("Testing Bybit API connection...")
    try:
        response = requests.get("https://api.bybit.com/v5/market/time", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0:
                server_time = datetime.fromtimestamp(int(data['result']['timeSecond']))
                logger.info(f"✅ Bybit API connection successful - Server time: {server_time}")
            else:
                logger.error(f"❌ API error: {data}")
                return
        else:
            logger.error(f"❌ HTTP error: {response.status_code}")
            return
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return
    
    # Test SOONUSDT data
    logger.info("Testing SOONUSDT market data...")
    try:
        response = requests.get("https://api.bybit.com/v5/market/kline", params={
            'category': 'linear',
            'symbol': 'SOONUSDT',
            'interval': '3',
            'limit': 1
        }, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0 and data['result']['list']:
                kline = data['result']['list'][0]
                price = float(kline[4])
                logger.info(f"✅ SOONUSDT current price: ${price:.4f}")
            else:
                logger.error(f"❌ No SOONUSDT data: {data}")
                return
        else:
            logger.error(f"❌ Market data error: {response.status_code}")
            return
    except Exception as e:
        logger.error(f"❌ Market data test failed: {e}")
        return
    
    # Initialize components
    data_collector = WindowsCompatibleDataCollector("SOONUSDT", testnet=False)
    trading_bot = WindowsTradingBot(data_collector)
    
    logger.info("🚀 Starting Windows-compatible real-time trading...")
    logger.info(f"Initial balance: ${trading_bot.balance:.2f}")
    
    try:
        # Start data collection (this will run the main loop)
        data_collector.start_collection_sync()
        
    except KeyboardInterrupt:
        print("\n\nStopping Windows-compatible trading system...")
        data_collector.stop_collection()
        
        # Final summary
        final_summary = trading_bot.get_performance_summary()
        
        print("\n" + "=" * 60)
        print("📈 FINAL RESULTS:")
        print(f"   Final Balance: ${final_summary['balance']:.2f}")
        print(f"   Total Return: {final_summary['total_return']:+.2f}%")
        print(f"   Total Trades: {final_summary['total_trades']}")
        if final_summary['total_trades'] > 0:
            print(f"   Win Rate: {final_summary['win_rate']:.1f}%")
            print(f"   Average PnL: ${final_summary['avg_pnl']:.2f}")
        print(f"   Active Positions: {final_summary['active_positions']}")
        print()
        print("🎯 This Windows-compatible system demonstrates:")
        print("   ✅ Real-time data from Bybit mainnet API")
        print("   ✅ Technical analysis signals")
        print("   ✅ Automatic trading execution")
        print("   ✅ Risk management with stop losses")
        print("   ✅ Windows compatibility (no async HTTP issues)")

if __name__ == "__main__":
    main()