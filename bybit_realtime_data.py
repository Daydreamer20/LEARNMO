#!/usr/bin/env python3
"""
Real-time Bybit data integration for continuous bot training and optimization
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Callable
import requests
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BybitConfig:
    """Configuration for Bybit API connection"""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["SOONUSDT"]
        
        # Load API keys from environment if not provided
        import os
        if not self.api_key:
            self.api_key = os.getenv('BYBIT_API_KEY', '')
        if not self.api_secret:
            self.api_secret = os.getenv('BYBIT_API_SECRET', '')

class BybitRealtimeDataCollector:
    """
    Real-time data collector from Bybit WebSocket API
    """
    
    def __init__(self, config: BybitConfig, db_path: str = "realtime_data.db"):
        self.config = config
        self.db_path = db_path
        
        # Multiple WebSocket URLs to try
        if config.testnet:
            self.ws_urls = [
                "wss://stream-testnet.bybit.com/v5/public/linear",
                "wss://stream-testnet.bybit.com/realtime_public"
            ]
            self.rest_url = "https://api-testnet.bybit.com"
        else:
            self.ws_urls = [
                "wss://stream.bybit.com/v5/public/linear",
                "wss://stream.bybit.com/realtime_public"
            ]
            self.rest_url = "https://api.bybit.com"
        
        # Data storage
        self.kline_data = {}
        self.trade_data = {}
        self.orderbook_data = {}
        
        # Callbacks for real-time processing
        self.data_callbacks = []
        
        # Initialize database
        self._init_database()
        
        # Control flags
        self.running = False
        self.ws_connection = None
    
    def _init_database(self):
        """Initialize SQLite database for storing real-time data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create klines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp INTEGER,
                open_time INTEGER,
                close_time INTEGER,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                turnover REAL,
                interval TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                trade_id TEXT,
                price REAL,
                size REAL,
                side TEXT,
                timestamp INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create orderbook table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orderbook (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                price REAL,
                size REAL,
                timestamp INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_klines_symbol_time ON klines(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_time ON orderbook(symbol, timestamp)')
        
        conn.commit()
        conn.close()
    
    def add_data_callback(self, callback: Callable):
        """Add callback function to be called when new data arrives"""
        self.data_callbacks.append(callback)
    
    async def connect_websocket(self):
        """Connect to Bybit WebSocket and subscribe to data streams"""
        for ws_url in self.ws_urls:
            try:
                logger.info(f"Attempting to connect to: {ws_url}")
                self.ws_connection = await asyncio.wait_for(
                    websockets.connect(ws_url, ping_interval=20, ping_timeout=10),
                    timeout=10.0
                )
                logger.info(f"Connected to Bybit WebSocket: {ws_url}")
                
                # Subscribe to kline data (3-minute intervals)
                for symbol in self.config.symbols:
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [
                            f"kline.3.{symbol}",  # 3-minute klines
                            f"publicTrade.{symbol}",  # Real-time trades
                            f"orderbook.1.{symbol}"  # Order book depth
                        ]
                    }
                    await self.ws_connection.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to {symbol} data streams")
                
                return True
                
            except asyncio.TimeoutError:
                logger.warning(f"Connection timeout for {ws_url}")
                continue
            except Exception as e:
                logger.warning(f"Failed to connect to {ws_url}: {e}")
                continue
        
        logger.error("Failed to connect to any WebSocket URL")
        return False
    
    async def listen_websocket(self):
        """Listen for WebSocket messages and process data"""
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                await self._process_websocket_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
    
    async def _process_websocket_message(self, data: Dict):
        """Process incoming WebSocket message"""
        if 'topic' not in data:
            return
        
        topic = data['topic']
        
        if topic.startswith('kline'):
            await self._process_kline_data(data)
        elif topic.startswith('publicTrade'):
            await self._process_trade_data(data)
        elif topic.startswith('orderbook'):
            await self._process_orderbook_data(data)
    
    async def _process_kline_data(self, data: Dict):
        """Process kline (candlestick) data"""
        try:
            kline_data = data['data'][0]
            symbol = kline_data['symbol']
            
            # Store in memory
            if symbol not in self.kline_data:
                self.kline_data[symbol] = []
            
            kline_record = {
                'symbol': symbol,
                'timestamp': int(kline_data['timestamp']),
                'open_time': int(kline_data['start']),
                'close_time': int(kline_data['end']),
                'open_price': float(kline_data['open']),
                'high_price': float(kline_data['high']),
                'low_price': float(kline_data['low']),
                'close_price': float(kline_data['close']),
                'volume': float(kline_data['volume']),
                'turnover': float(kline_data['turnover']),
                'interval': '3m'
            }
            
            self.kline_data[symbol].append(kline_record)
            
            # Keep only last 1000 records in memory
            if len(self.kline_data[symbol]) > 1000:
                self.kline_data[symbol] = self.kline_data[symbol][-1000:]
            
            # Store in database
            self._store_kline_data(kline_record)
            
            # Trigger callbacks
            for callback in self.data_callbacks:
                try:
                    await callback('kline', kline_record)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
            
            logger.info(f"New kline: {symbol} @ {kline_record['close_price']}")
            
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
    
    async def _process_trade_data(self, data: Dict):
        """Process real-time trade data"""
        try:
            for trade in data['data']:
                trade_record = {
                    'symbol': trade['s'],
                    'trade_id': trade['i'],
                    'price': float(trade['p']),
                    'size': float(trade['v']),
                    'side': trade['S'],
                    'timestamp': int(trade['T'])
                }
                
                # Store in memory
                symbol = trade_record['symbol']
                if symbol not in self.trade_data:
                    self.trade_data[symbol] = []
                
                self.trade_data[symbol].append(trade_record)
                
                # Keep only last 500 trades in memory
                if len(self.trade_data[symbol]) > 500:
                    self.trade_data[symbol] = self.trade_data[symbol][-500:]
                
                # Store in database
                self._store_trade_data(trade_record)
                
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    async def _process_orderbook_data(self, data: Dict):
        """Process order book data"""
        try:
            orderbook = data['data']
            symbol = data['topic'].split('.')[-1]
            timestamp = int(data['ts'])
            
            # Process bids and asks
            for side, orders in [('bid', orderbook['b']), ('ask', orderbook['a'])]:
                for order in orders:
                    orderbook_record = {
                        'symbol': symbol,
                        'side': side,
                        'price': float(order[0]),
                        'size': float(order[1]),
                        'timestamp': timestamp
                    }
                    
                    # Store in database (only store top 10 levels)
                    if len(orders) <= 10:
                        self._store_orderbook_data(orderbook_record)
            
        except Exception as e:
            logger.error(f"Error processing orderbook data: {e}")
    
    def _store_kline_data(self, kline_record: Dict):
        """Store kline data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO klines (symbol, timestamp, open_time, close_time, 
                                  open_price, high_price, low_price, close_price, 
                                  volume, turnover, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kline_record['symbol'], kline_record['timestamp'],
                kline_record['open_time'], kline_record['close_time'],
                kline_record['open_price'], kline_record['high_price'],
                kline_record['low_price'], kline_record['close_price'],
                kline_record['volume'], kline_record['turnover'],
                kline_record['interval']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing kline data: {e}")
    
    def _store_trade_data(self, trade_record: Dict):
        """Store trade data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (symbol, trade_id, price, size, side, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                trade_record['symbol'], trade_record['trade_id'],
                trade_record['price'], trade_record['size'],
                trade_record['side'], trade_record['timestamp']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing trade data: {e}")
    
    def _store_orderbook_data(self, orderbook_record: Dict):
        """Store orderbook data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO orderbook (symbol, side, price, size, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                orderbook_record['symbol'], orderbook_record['side'],
                orderbook_record['price'], orderbook_record['size'],
                orderbook_record['timestamp']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing orderbook data: {e}")
    
    def get_recent_klines(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get recent kline data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM klines 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        conn.close()
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get recent trade data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM trades 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        conn.close()
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    async def start_collection(self):
        """Start real-time data collection with fallback to REST API"""
        self.running = True
        
        # Try WebSocket connection
        connected = await self.connect_websocket()
        if connected:
            # Start listening
            logger.info("Starting real-time data collection via WebSocket...")
            try:
                await self.listen_websocket()
            except Exception as e:
                logger.error(f"WebSocket listening failed: {e}")
                logger.info("Falling back to REST API polling...")
                await self.start_rest_polling()
        else:
            logger.warning("WebSocket connection failed, using REST API polling...")
            await self.start_rest_polling()
        
        return True
    
    async def start_rest_polling(self):
        """Fallback: Poll REST API for data"""
        logger.info("Starting REST API polling (fallback mode)...")
        
        while self.running:
            try:
                for symbol in self.config.symbols:
                    # Get latest kline data from REST API
                    await self._fetch_rest_kline(symbol)
                
                # Wait 3 minutes (to match 3-minute klines)
                await asyncio.sleep(180)
                
            except Exception as e:
                logger.error(f"Error in REST polling: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _fetch_rest_kline(self, symbol: str):
        """Fetch kline data from REST API"""
        try:
            import aiohttp
            
            url = f"{self.rest_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': '3',
                'limit': 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data['retCode'] == 0 and data['result']['list']:
                            kline = data['result']['list'][0]
                            
                            # Convert to our format
                            kline_record = {
                                'symbol': symbol,
                                'timestamp': int(kline[0]),
                                'open_time': int(kline[0]),
                                'close_time': int(kline[0]) + 180000,  # 3 minutes later
                                'open_price': float(kline[1]),
                                'high_price': float(kline[2]),
                                'low_price': float(kline[3]),
                                'close_price': float(kline[4]),
                                'volume': float(kline[5]),
                                'turnover': float(kline[6]),
                                'interval': '3m'
                            }
                            
                            # Store and trigger callbacks
                            self._store_kline_data(kline_record)
                            
                            for callback in self.data_callbacks:
                                try:
                                    await callback('kline', kline_record)
                                except Exception as e:
                                    logger.error(f"Error in data callback: {e}")
                            
                            logger.info(f"REST API: {symbol} @ ${kline_record['close_price']:.4f}")
                    
        except ImportError:
            logger.error("aiohttp not installed. Installing...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'aiohttp'])
            logger.info("aiohttp installed. Please restart the system.")
            
        except Exception as e:
            logger.error(f"Error fetching REST data for {symbol}: {e}")
    
    def stop_collection(self):
        """Stop real-time data collection"""
        self.running = False
        if self.ws_connection:
            asyncio.create_task(self.ws_connection.close())
        logger.info("Stopped real-time data collection")

class RealtimeTrainingPipeline:
    """
    Pipeline for continuous training using real-time data
    """
    
    def __init__(self, data_collector: BybitRealtimeDataCollector, 
                 wandb_tracker=None, training_interval: int = 300):
        self.data_collector = data_collector
        self.wandb_tracker = wandb_tracker
        self.training_interval = training_interval  # seconds
        
        # Training state
        self.last_training_time = 0
        self.training_data_buffer = []
        
        # Add callback to data collector
        self.data_collector.add_data_callback(self._on_new_data)
    
    async def _on_new_data(self, data_type: str, data: Dict):
        """Callback for new data arrival"""
        if data_type == 'kline':
            # Add to training buffer
            self.training_data_buffer.append(data)
            
            # Check if it's time to retrain
            current_time = time.time()
            if current_time - self.last_training_time > self.training_interval:
                await self._trigger_training()
                self.last_training_time = current_time
    
    async def _trigger_training(self):
        """Trigger model retraining with latest data"""
        try:
            logger.info("Triggering model retraining with latest data...")
            
            # Get recent data for training
            symbol = self.data_collector.config.symbols[0]
            recent_data = self.data_collector.get_recent_klines(symbol, limit=500)
            
            if len(recent_data) < 100:
                logger.warning("Not enough data for training")
                return
            
            # Convert to format expected by backtester
            training_df = self._prepare_training_data(recent_data)
            
            # Run quick optimization
            await self._run_quick_optimization(training_df)
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
    
    def _prepare_training_data(self, kline_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for training"""
        # Convert to expected format
        df = pd.DataFrame()
        df['timestamp'] = pd.to_datetime(kline_data['timestamp'], unit='ms')
        df['open'] = kline_data['open_price']
        df['high'] = kline_data['high_price']
        df['low'] = kline_data['low_price']
        df['close'] = kline_data['close_price']
        df['volume'] = kline_data['volume']
        
        # Add technical indicators (simplified)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Add dummy signals for now (replace with your signal logic)
        df['Bullish BOS'] = (df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume'].rolling(10).mean())
        df['Bearish BOS'] = (df['close'] < df['close'].shift(1)) & (df['volume'] > df['volume'].rolling(10).mean())
        df['Bullish CHOCH'] = False
        df['Bearish CHOCH'] = False
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def _run_quick_optimization(self, data: pd.DataFrame):
        """Run quick parameter optimization with latest data"""
        try:
            from chunked_backtester import ChunkedBacktester
            
            # Save data temporarily
            temp_file = "temp_realtime_data.csv"
            data.to_csv(temp_file, index=False)
            
            # Test current parameters
            config = {
                'initial_balance': 1000,
                'leverage': 10,
                'stop_loss_pct': 0.015,
                'take_profit_pct': 0.03,
                'position_size_pct': 0.1
            }
            
            backtester = ChunkedBacktester(temp_file, config)
            results = backtester.run_chunked_backtest('time', time_window='6H')
            analysis = backtester.analyze_results()
            
            # Log to wandb if available
            if self.wandb_tracker:
                self.wandb_tracker.log_backtest_results(
                    results, analysis, "Realtime_Training"
                )
            
            logger.info(f"Quick optimization complete - PnL: ${analysis['total_pnl']:.2f}")
            
            # Clean up
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            logger.error(f"Error in quick optimization: {e}")

async def main():
    """Main function to demonstrate real-time data collection and training"""
    # Configuration
    config = BybitConfig(
        testnet=True,
        symbols=["SOONUSDT"]
    )
    
    # Initialize data collector
    collector = BybitRealtimeDataCollector(config)
    
    # Initialize training pipeline
    training_pipeline = RealtimeTrainingPipeline(
        collector, 
        training_interval=300  # Retrain every 5 minutes
    )
    
    try:
        # Start data collection
        await collector.start_collection()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        collector.stop_collection()

if __name__ == "__main__":
    asyncio.run(main())