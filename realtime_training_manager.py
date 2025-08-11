#!/usr/bin/env python3
"""
Real-time training manager that continuously optimizes bot parameters
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import time
import logging
from typing import Dict, List, Optional
import wandb
from dataclasses import dataclass, asdict

from bybit_realtime_data import BybitRealtimeDataCollector, BybitConfig
from chunked_backtester import ChunkedBacktester
from wandb_integration import WandbTradingTracker

logger = logging.getLogger(__name__)

@dataclass
class OptimizedParameters:
    """Container for optimized trading parameters"""
    stop_loss_pct: float
    take_profit_pct: float
    position_size_pct: float
    leverage: int
    chunking_method: str
    chunk_size: Optional[int] = None
    time_window: Optional[str] = None
    performance_score: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class RealtimeTrainingManager:
    """
    Manages continuous training and optimization of trading bot parameters
    """
    
    def __init__(self, 
                 data_collector: BybitRealtimeDataCollector,
                 wandb_tracker: WandbTradingTracker = None,
                 optimization_interval: int = 1800,  # 30 minutes
                 min_data_points: int = 200):
        
        self.data_collector = data_collector
        self.wandb_tracker = wandb_tracker
        self.optimization_interval = optimization_interval
        self.min_data_points = min_data_points
        
        # Current best parameters
        self.current_params = OptimizedParameters(
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            position_size_pct=0.1,
            leverage=10,
            chunking_method='time',
            time_window='6H'
        )
        
        # Performance tracking
        self.performance_history = []
        self.last_optimization_time = 0
        
        # Parameter ranges for optimization
        self.param_ranges = {
            'stop_loss_pct': (0.005, 0.03),
            'take_profit_pct': (0.01, 0.08),
            'position_size_pct': (0.05, 0.25),
            'leverage': [5, 10, 15, 20],
            'chunking_method': ['time', 'rows'],
            'chunk_size': [50, 75, 100, 150, 200],
            'time_window': ['3H', '6H', '12H', '1D']
        }
        
        # Initialize parameter database
        self._init_parameter_db()
        
        # Add callback for new data
        self.data_collector.add_data_callback(self._on_new_data)
    
    def _init_parameter_db(self):
        """Initialize database for storing parameter optimization history"""
        conn = sqlite3.connect(self.data_collector.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameter_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                stop_loss_pct REAL,
                take_profit_pct REAL,
                position_size_pct REAL,
                leverage INTEGER,
                chunking_method TEXT,
                chunk_size INTEGER,
                time_window TEXT,
                performance_score REAL,
                total_pnl REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                data_points INTEGER,
                optimization_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def _on_new_data(self, data_type: str, data: Dict):
        """Callback for new data - check if optimization is needed"""
        if data_type == 'kline':
            current_time = time.time()
            
            # Check if it's time for optimization
            if current_time - self.last_optimization_time > self.optimization_interval:
                await self._trigger_optimization()
                self.last_optimization_time = current_time
    
    async def _trigger_optimization(self):
        """Trigger parameter optimization with latest data"""
        try:
            logger.info("🔄 Starting parameter optimization...")
            
            # Get recent data
            symbol = self.data_collector.config.symbols[0]
            recent_data = self.data_collector.get_recent_klines(symbol, limit=1000)
            
            if len(recent_data) < self.min_data_points:
                logger.warning(f"Not enough data for optimization: {len(recent_data)} < {self.min_data_points}")
                return
            
            # Prepare data for backtesting
            training_data = self._prepare_training_data(recent_data)
            
            # Run optimization
            optimized_params = await self._optimize_parameters(training_data)
            
            # Evaluate if new parameters are better
            if self._should_update_parameters(optimized_params):
                logger.info(f"📈 Updating parameters - Performance improved by {optimized_params.performance_score - self.current_params.performance_score:.2f}")
                self.current_params = optimized_params
                
                # Log parameter update
                if self.wandb_tracker:
                    self._log_parameter_update(optimized_params)
            
            # Store optimization result
            self._store_optimization_result(optimized_params, len(training_data))
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
    
    def _prepare_training_data(self, kline_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare real-time data for backtesting"""
        # Convert to expected format
        df = pd.DataFrame()
        df['timestamp'] = pd.to_datetime(kline_data['timestamp'], unit='ms')
        df['open'] = kline_data['open_price']
        df['high'] = kline_data['high_price']
        df['low'] = kline_data['low_price']
        df['close'] = kline_data['close_price']
        df['volume'] = kline_data['volume']
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Add trading signals
        df = self._add_trading_signals(df)
        
        return df.dropna()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _add_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading signals based on technical analysis"""
        # Trend signals
        df['trend_up'] = (df['close'] > df['sma_20']) & (df['sma_10'] > df['sma_20'])
        df['trend_down'] = (df['close'] < df['sma_20']) & (df['sma_10'] < df['sma_20'])
        
        # Momentum signals
        df['momentum_up'] = (df['rsi'] > 30) & (df['rsi'] < 70) & (df['macd'] > df['macd_signal'])
        df['momentum_down'] = (df['rsi'] > 30) & (df['rsi'] < 70) & (df['macd'] < df['macd_signal'])
        
        # Volume confirmation
        df['volume_confirm'] = df['volume_ratio'] > 1.2
        
        # Combined signals (replace with your actual signal logic)
        df['Bullish BOS'] = df['trend_up'] & df['momentum_up'] & df['volume_confirm']
        df['Bearish BOS'] = df['trend_down'] & df['momentum_down'] & df['volume_confirm']
        df['Bullish CHOCH'] = False  # Placeholder
        df['Bearish CHOCH'] = False  # Placeholder
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    async def _optimize_parameters(self, data: pd.DataFrame) -> OptimizedParameters:
        """Optimize parameters using grid search"""
        best_params = None
        best_score = float('-inf')
        
        # Save data temporarily
        temp_file = f"temp_optimization_{int(time.time())}.csv"
        data.to_csv(temp_file, index=False)
        
        try:
            # Grid search over parameter space
            param_combinations = self._generate_parameter_combinations()
            
            logger.info(f"Testing {len(param_combinations)} parameter combinations...")
            
            for i, params in enumerate(param_combinations):
                try:
                    # Create config
                    config = {
                        'initial_balance': 1000,
                        'leverage': params['leverage'],
                        'stop_loss_pct': params['stop_loss_pct'],
                        'take_profit_pct': params['take_profit_pct'],
                        'position_size_pct': params['position_size_pct']
                    }
                    
                    # Run backtest
                    backtester = ChunkedBacktester(temp_file, config)
                    
                    if params['chunking_method'] == 'time':
                        results = backtester.run_chunked_backtest('time', time_window=params['time_window'])
                    else:
                        results = backtester.run_chunked_backtest('rows', chunk_size=params['chunk_size'])
                    
                    analysis = backtester.analyze_results()
                    
                    # Calculate performance score (weighted combination of metrics)
                    score = self._calculate_performance_score(analysis)
                    
                    if score > best_score:
                        best_score = score
                        best_params = OptimizedParameters(
                            stop_loss_pct=params['stop_loss_pct'],
                            take_profit_pct=params['take_profit_pct'],
                            position_size_pct=params['position_size_pct'],
                            leverage=params['leverage'],
                            chunking_method=params['chunking_method'],
                            chunk_size=params.get('chunk_size'),
                            time_window=params.get('time_window'),
                            performance_score=score
                        )
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        logger.info(f"Tested {i + 1}/{len(param_combinations)} combinations, best score: {best_score:.2f}")
                
                except Exception as e:
                    logger.warning(f"Error testing parameter combination {i}: {e}")
                    continue
            
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        if best_params is None:
            logger.warning("No valid parameter combination found, keeping current parameters")
            return self.current_params
        
        logger.info(f"✅ Optimization complete - Best score: {best_score:.2f}")
        return best_params
    
    def _generate_parameter_combinations(self, max_combinations: int = 50) -> List[Dict]:
        """Generate parameter combinations for testing"""
        import itertools
        import random
        
        # Create parameter grids
        stop_loss_values = np.linspace(self.param_ranges['stop_loss_pct'][0], 
                                     self.param_ranges['stop_loss_pct'][1], 5)
        take_profit_values = np.linspace(self.param_ranges['take_profit_pct'][0], 
                                       self.param_ranges['take_profit_pct'][1], 5)
        position_size_values = np.linspace(self.param_ranges['position_size_pct'][0], 
                                         self.param_ranges['position_size_pct'][1], 4)
        
        combinations = []
        
        # Generate combinations
        for stop_loss in stop_loss_values:
            for take_profit in take_profit_values:
                for position_size in position_size_values:
                    for leverage in self.param_ranges['leverage']:
                        # Time-based chunking
                        for time_window in self.param_ranges['time_window']:
                            combinations.append({
                                'stop_loss_pct': round(stop_loss, 4),
                                'take_profit_pct': round(take_profit, 4),
                                'position_size_pct': round(position_size, 3),
                                'leverage': leverage,
                                'chunking_method': 'time',
                                'time_window': time_window
                            })
                        
                        # Row-based chunking
                        for chunk_size in self.param_ranges['chunk_size']:
                            combinations.append({
                                'stop_loss_pct': round(stop_loss, 4),
                                'take_profit_pct': round(take_profit, 4),
                                'position_size_pct': round(position_size, 3),
                                'leverage': leverage,
                                'chunking_method': 'rows',
                                'chunk_size': chunk_size
                            })
        
        # Randomly sample if too many combinations
        if len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)
        
        return combinations
    
    def _calculate_performance_score(self, analysis: Dict) -> float:
        """Calculate weighted performance score"""
        # Weights for different metrics
        weights = {
            'total_pnl': 0.4,
            'win_rate': 0.2,
            'sharpe_ratio': 0.2,
            'consistency': 0.2
        }
        
        # Normalize metrics
        pnl_score = min(analysis['total_pnl'] / 100, 5)  # Cap at 5x
        win_rate_score = analysis['overall_win_rate'] * 5  # 0-5 scale
        
        # Calculate Sharpe ratio
        sharpe_ratio = analysis['avg_chunk_return'] / max(analysis['return_std'], 0.01)
        sharpe_score = min(max(sharpe_ratio, -2), 3)  # -2 to 3 scale
        
        # Consistency score (profitable chunks ratio)
        consistency_score = (analysis['profitable_chunks'] / analysis['total_chunks']) * 5
        
        # Weighted score
        score = (weights['total_pnl'] * pnl_score +
                weights['win_rate'] * win_rate_score +
                weights['sharpe_ratio'] * sharpe_score +
                weights['consistency'] * consistency_score)
        
        return score
    
    def _should_update_parameters(self, new_params: OptimizedParameters) -> bool:
        """Determine if new parameters are significantly better"""
        improvement_threshold = 0.1  # 10% improvement required
        
        return new_params.performance_score > (self.current_params.performance_score + improvement_threshold)
    
    def _store_optimization_result(self, params: OptimizedParameters, data_points: int):
        """Store optimization result in database"""
        try:
            conn = sqlite3.connect(self.data_collector.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO parameter_optimization 
                (timestamp, stop_loss_pct, take_profit_pct, position_size_pct, 
                 leverage, chunking_method, chunk_size, time_window, 
                 performance_score, data_points, optimization_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                params.timestamp, params.stop_loss_pct, params.take_profit_pct,
                params.position_size_pct, params.leverage, params.chunking_method,
                params.chunk_size, params.time_window, params.performance_score,
                data_points, 'realtime'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing optimization result: {e}")
    
    def _log_parameter_update(self, params: OptimizedParameters):
        """Log parameter update to wandb"""
        if not self.wandb_tracker:
            return
        
        try:
            wandb.log({
                'parameter_update': True,
                'new_stop_loss_pct': params.stop_loss_pct,
                'new_take_profit_pct': params.take_profit_pct,
                'new_position_size_pct': params.position_size_pct,
                'new_leverage': params.leverage,
                'new_chunking_method': params.chunking_method,
                'new_performance_score': params.performance_score,
                'improvement': params.performance_score - self.current_params.performance_score
            })
            
        except Exception as e:
            logger.error(f"Error logging parameter update: {e}")
    
    def get_current_parameters(self) -> Dict:
        """Get current optimized parameters for trading"""
        return asdict(self.current_params)
    
    def get_optimization_history(self, limit: int = 100) -> pd.DataFrame:
        """Get parameter optimization history"""
        conn = sqlite3.connect(self.data_collector.db_path)
        
        query = '''
            SELECT * FROM parameter_optimization 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df

async def main():
    """Main function to demonstrate real-time training"""
    # Configuration
    config = BybitConfig(
        testnet=True,
        symbols=["SOONUSDT"]
    )
    
    # Initialize components
    data_collector = BybitRealtimeDataCollector(config)
    wandb_tracker = WandbTradingTracker("realtime-optimization")
    
    # Initialize training manager
    training_manager = RealtimeTrainingManager(
        data_collector=data_collector,
        wandb_tracker=wandb_tracker,
        optimization_interval=900,  # 15 minutes for demo
        min_data_points=100
    )
    
    logger.info("🚀 Starting real-time training and optimization...")
    
    try:
        # Start data collection and training
        await data_collector.start_collection()
        
    except KeyboardInterrupt:
        logger.info("Shutting down real-time training...")
        data_collector.stop_collection()
        
        # Print final parameters
        final_params = training_manager.get_current_parameters()
        logger.info(f"Final optimized parameters: {final_params}")

if __name__ == "__main__":
    asyncio.run(main())