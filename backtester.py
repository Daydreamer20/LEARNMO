#!/usr/bin/env python3
"""
Comprehensive backtesting engine for trading bot optimization
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Trade:
    """Trade execution record"""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    stop_loss: float
    take_profit: float
    exit_reason: str  # 'TP', 'SL', 'SIGNAL', 'TIME'
    pnl: float
    pnl_pct: float
    duration: timedelta
    confidence: float
    reasoning: str

@dataclass
class BacktestResults:
    """Backtest results summary"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: timedelta
    best_trade: float
    worst_trade: float
    trades: List[Trade]

class TradingBacktester:
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.logger = logging.getLogger(__name__)
        
        # Trading parameters (matching your bot)
        self.max_trade_size = 2.0  # $2 max per trade
        self.stop_loss_pct = 0.1   # 0.1% before liquidation
        self.risk_per_trade = 0.01 # 1% of balance
        
    def load_data(self, file_path: str, symbol: str = None) -> pd.DataFrame:
        """Load market data from various formats"""
        try:
            # Detect file format
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Add symbol if provided
            if symbol:
                df['symbol'] = symbol
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                # Handle Unix timestamp (convert from seconds to datetime)
                if df['timestamp'].dtype in ['int64', 'float64']:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            elif 'time' in df.columns:
                # Handle 'time' column
                if df['time'].dtype in ['int64', 'float64']:
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                else:
                    df['timestamp'] = pd.to_datetime(df['time'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(df)} data points from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match expected format"""
        column_mapping = {
            # Common variations
            'time': 'timestamp',
            'datetime': 'timestamp',
            'date': 'timestamp',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'vol': 'volume',
            'volume_24h': 'volume',
            # Bybit format
            'start_at': 'timestamp',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")
            # Try to infer missing columns
            if 'close' in df.columns and 'open' not in df.columns:
                df['open'] = df['close'].shift(1)
            if 'close' in df.columns and 'high' not in df.columns:
                df['high'] = df['close']
            if 'close' in df.columns and 'low' not in df.columns:
                df['low'] = df['close']
            if 'volume' not in df.columns:
                df['volume'] = 1000000  # Default volume
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # EMA
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df
    
    def generate_signals(self, df: pd.DataFrame, strategy: str = 'technical') -> pd.DataFrame:
        """Generate trading signals based on strategy"""
        signals = []
        
        for i in range(len(df)):
            if i < 50:  # Need enough data for indicators
                signals.append({'signal': 'HOLD', 'confidence': 0, 'reasoning': 'Insufficient data'})
                continue
            
            row = df.iloc[i]
            
            if strategy == 'technical':
                signal = self._technical_signal(row, df.iloc[max(0, i-20):i+1])
            elif strategy == 'rsi_mean_reversion':
                signal = self._rsi_mean_reversion_signal(row)
            elif strategy == 'trend_following':
                signal = self._trend_following_signal(row)
            else:
                signal = {'signal': 'HOLD', 'confidence': 0, 'reasoning': 'Unknown strategy'}
            
            signals.append(signal)
        
        # Convert to DataFrame columns
        df['signal'] = [s['signal'] for s in signals]
        df['confidence'] = [s['confidence'] for s in signals]
        df['reasoning'] = [s['reasoning'] for s in signals]
        
        return df
    
    def _technical_signal(self, row, history_df) -> Dict:
        """Generate signal using technical analysis (matching your bot)"""
        try:
            rsi = row['rsi']
            ema_9 = row['ema_9']
            ema_21 = row['ema_21']
            bb_upper = row['bb_upper']
            bb_lower = row['bb_lower']
            macd = row['macd']
            macd_signal = row['macd_signal']
            volume_ratio = row['volume_ratio']
            current_price = row['close']
            
            # Scoring system (matching your simple_trading_bot.py)
            buy_score = 0
            sell_score = 0
            
            # RSI signals
            if rsi < 30:
                buy_score += 2
            elif rsi > 70:
                sell_score += 2
            elif 40 < rsi < 60:
                buy_score += 1
                sell_score += 1
            
            # EMA crossover
            if ema_9 > ema_21:
                buy_score += 2
            elif ema_9 < ema_21:
                sell_score += 2
            
            # Bollinger Bands
            if current_price < bb_lower:
                buy_score += 2
            elif current_price > bb_upper:
                sell_score += 2
            
            # MACD
            if macd > macd_signal:
                buy_score += 1
            elif macd < macd_signal:
                sell_score += 1
            
            # Volume
            if volume_ratio > 1.5:
                buy_score += 1
                sell_score += 1
            
            # Determine signal
            if buy_score >= 4 and buy_score > sell_score:
                signal_type = "BUY"
                confidence = min(85 + (buy_score - 4) * 3, 95)
            elif sell_score >= 4 and sell_score > buy_score:
                signal_type = "SELL"
                confidence = min(85 + (sell_score - 4) * 3, 95)
            else:
                signal_type = "HOLD"
                confidence = 0
            
            reasoning = f"RSI:{rsi:.1f}, EMA:{'Bull' if ema_9>ema_21 else 'Bear'}, Vol:{volume_ratio:.1f}x"
            
            return {
                'signal': signal_type,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            return {'signal': 'HOLD', 'confidence': 0, 'reasoning': f'Error: {e}'}
    
    def _rsi_mean_reversion_signal(self, row) -> Dict:
        """RSI mean reversion strategy"""
        rsi = row['rsi']
        
        if rsi < 25:
            return {'signal': 'BUY', 'confidence': 90, 'reasoning': f'RSI oversold: {rsi:.1f}'}
        elif rsi > 75:
            return {'signal': 'SELL', 'confidence': 90, 'reasoning': f'RSI overbought: {rsi:.1f}'}
        else:
            return {'signal': 'HOLD', 'confidence': 0, 'reasoning': f'RSI neutral: {rsi:.1f}'}
    
    def _trend_following_signal(self, row) -> Dict:
        """EMA trend following strategy"""
        ema_9 = row['ema_9']
        ema_21 = row['ema_21']
        current_price = row['close']
        
        if ema_9 > ema_21 and current_price > ema_9:
            return {'signal': 'BUY', 'confidence': 85, 'reasoning': 'Uptrend confirmed'}
        elif ema_9 < ema_21 and current_price < ema_9:
            return {'signal': 'SELL', 'confidence': 85, 'reasoning': 'Downtrend confirmed'}
        else:
            return {'signal': 'HOLD', 'confidence': 0, 'reasoning': 'No clear trend'}
    
    def run_backtest(self, df: pd.DataFrame, strategy: str = 'technical') -> BacktestResults:
        """Run complete backtest"""
        self.logger.info(f"Starting backtest with {len(df)} data points")
        
        # Reset state
        self.current_balance = self.initial_balance
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        
        # Calculate indicators and signals
        df = self.calculate_indicators(df)
        df = self.generate_signals(df, strategy)
        
        # Process each data point
        for i in range(len(df)):
            row = df.iloc[i]
            self._process_tick(row, i)
            
            # Record equity
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'balance': self.current_balance,
                'equity': self._calculate_equity(row)
            })
        
        # Close any remaining positions
        self._close_all_positions(df.iloc[-1])
        
        # Calculate results
        results = self._calculate_results()
        
        self.logger.info(f"Backtest completed: {results.total_trades} trades, {results.win_rate:.1f}% win rate")
        
        return results
    
    def _process_tick(self, row, index):
        """Process single data point"""
        timestamp = row['timestamp']
        symbol = row.get('symbol', 'UNKNOWN')
        current_price = row['close']
        
        # Check existing positions for exits
        self._check_position_exits(row)
        
        # Check for new entries
        if row['signal'] in ['BUY', 'SELL'] and row['confidence'] >= 80:
            if symbol not in self.positions:  # No existing position
                self._enter_position(row)
    
    def _enter_position(self, row):
        """Enter new position"""
        symbol = row.get('symbol', 'UNKNOWN')
        side = row['signal']
        current_price = row['close']
        confidence = row['confidence']
        reasoning = row['reasoning']
        
        # Calculate position size (matching your bot logic)
        trade_size = min(self.max_trade_size, self.current_balance * self.risk_per_trade)
        
        # Calculate leverage (simplified)
        leverage = 10  # Default leverage
        
        # Calculate stop loss and take profit
        if side == 'BUY':
            stop_loss = current_price * (1 - self.stop_loss_pct / 100)
            take_profit = current_price * (1 + (self.stop_loss_pct * 2) / 100)  # 2:1 R:R
        else:
            stop_loss = current_price * (1 + self.stop_loss_pct / 100)
            take_profit = current_price * (1 - (self.stop_loss_pct * 2) / 100)
        
        quantity = (trade_size * leverage) / current_price
        
        # Create position
        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': current_price,
            'entry_time': row['timestamp'],
            'quantity': quantity,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'reasoning': reasoning
        }
        
        self.positions[symbol] = position
        self.logger.debug(f"Entered {side} position: {symbol} @ {current_price:.5f}")
    
    def _check_position_exits(self, row):
        """Check if positions should be closed"""
        symbol = row.get('symbol', 'UNKNOWN')
        current_price = row['close']
        
        if symbol in self.positions:
            position = self.positions[symbol]
            exit_reason = None
            
            # Check stop loss
            if position['side'] == 'BUY' and current_price <= position['stop_loss']:
                exit_reason = 'SL'
            elif position['side'] == 'SELL' and current_price >= position['stop_loss']:
                exit_reason = 'SL'
            
            # Check take profit
            elif position['side'] == 'BUY' and current_price >= position['take_profit']:
                exit_reason = 'TP'
            elif position['side'] == 'SELL' and current_price <= position['take_profit']:
                exit_reason = 'TP'
            
            # Check signal reversal
            elif row['signal'] != 'HOLD' and row['signal'] != position['side'] and row['confidence'] >= 80:
                exit_reason = 'SIGNAL'
            
            if exit_reason:
                self._exit_position(symbol, current_price, row['timestamp'], exit_reason)
    
    def _exit_position(self, symbol, exit_price, exit_time, exit_reason):
        """Exit position and record trade"""
        position = self.positions[symbol]
        
        # Calculate PnL
        if position['side'] == 'BUY':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        pnl_usd = pnl_pct * position['quantity'] * position['entry_price'] / position['leverage']
        
        # Create trade record
        trade = Trade(
            timestamp=position['entry_time'],
            symbol=symbol,
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            leverage=position['leverage'],
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            exit_reason=exit_reason,
            pnl=pnl_usd,
            pnl_pct=pnl_pct * 100,
            duration=exit_time - position['entry_time'],
            confidence=position['confidence'],
            reasoning=position['reasoning']
        )
        
        self.trades.append(trade)
        self.current_balance += pnl_usd
        
        # Remove position
        del self.positions[symbol]
        
        self.logger.debug(f"Exited {position['side']} position: {symbol} @ {exit_price:.5f} ({exit_reason}) PnL: ${pnl_usd:.2f}")
    
    def _close_all_positions(self, final_row):
        """Close all remaining positions at end of backtest"""
        for symbol in list(self.positions.keys()):
            self._exit_position(symbol, final_row['close'], final_row['timestamp'], 'TIME')
    
    def _calculate_equity(self, row):
        """Calculate current equity including unrealized PnL"""
        equity = self.current_balance
        
        for symbol, position in self.positions.items():
            if symbol == row.get('symbol', 'UNKNOWN'):
                current_price = row['close']
                if position['side'] == 'BUY':
                    unrealized_pnl = (current_price - position['entry_price']) / position['entry_price']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) / position['entry_price']
                
                unrealized_usd = unrealized_pnl * position['quantity'] * position['entry_price'] / position['leverage']
                equity += unrealized_usd
        
        return equity
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate backtest results"""
        if not self.trades:
            return BacktestResults(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
                total_pnl=0, total_pnl_pct=0, max_drawdown=0, sharpe_ratio=0,
                profit_factor=0, avg_trade_duration=timedelta(0),
                best_trade=0, worst_trade=0, trades=[]
            )
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate max drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        peak = equity_values[0]
        max_drawdown = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in self.trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return BacktestResults(
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(self.trades) * 100,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_drawdown * 100,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_trade_duration=sum([t.duration for t in self.trades], timedelta(0)) / len(self.trades),
            best_trade=max(t.pnl for t in self.trades),
            worst_trade=min(t.pnl for t in self.trades),
            trades=self.trades
        )
    
    def generate_report(self, results: BacktestResults, save_path: str = None):
        """Generate comprehensive backtest report"""
        report = f"""
        ═══════════════════════════════════════════════════════════════
                            BACKTEST RESULTS REPORT
        ═══════════════════════════════════════════════════════════════
        
        📊 PERFORMANCE SUMMARY
        ───────────────────────────────────────────────────────────────
        Total Trades:           {results.total_trades}
        Winning Trades:         {results.winning_trades} ({results.win_rate:.1f}%)
        Losing Trades:          {results.losing_trades}
        
        💰 PROFIT & LOSS
        ───────────────────────────────────────────────────────────────
        Total PnL:              ${results.total_pnl:.2f}
        Total Return:           {results.total_pnl_pct:.2f}%
        Best Trade:             ${results.best_trade:.2f}
        Worst Trade:            ${results.worst_trade:.2f}
        
        📈 RISK METRICS
        ───────────────────────────────────────────────────────────────
        Max Drawdown:           {results.max_drawdown:.2f}%
        Sharpe Ratio:           {results.sharpe_ratio:.2f}
        Profit Factor:          {results.profit_factor:.2f}
        
        ⏱️  TIMING
        ───────────────────────────────────────────────────────────────
        Avg Trade Duration:     {results.avg_trade_duration}
        
        ═══════════════════════════════════════════════════════════════
        """
        
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
                # Add detailed trade list
                f.write("\n\nDETAILED TRADE LIST\n")
                f.write("=" * 80 + "\n")
                
                for i, trade in enumerate(results.trades, 1):
                    f.write(f"\nTrade {i}:\n")
                    f.write(f"  Time: {trade.timestamp}\n")
                    f.write(f"  Symbol: {trade.symbol}\n")
                    f.write(f"  Side: {trade.side}\n")
                    f.write(f"  Entry: ${trade.entry_price:.5f}\n")
                    f.write(f"  Exit: ${trade.exit_price:.5f}\n")
                    f.write(f"  PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)\n")
                    f.write(f"  Duration: {trade.duration}\n")
                    f.write(f"  Exit Reason: {trade.exit_reason}\n")
                    f.write(f"  Confidence: {trade.confidence}%\n")
                    f.write(f"  Reasoning: {trade.reasoning}\n")
        
        return report

if __name__ == "__main__":
    # Example usage
    backtester = TradingBacktester(initial_balance=1000.0)
    
    print("Trading Bot Backtester Ready!")
    print("Usage:")
    print("1. Place your data file in the current directory")
    print("2. Run: python backtester.py")
    print("3. Follow the prompts to load data and run backtest")