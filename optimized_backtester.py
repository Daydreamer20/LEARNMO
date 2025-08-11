#!/usr/bin/env python3
"""
Run backtest with optimized parameters for SIRENUSDT
"""

from backtester import TradingBacktester
from optimized_bot_config import OPTIMIZED_PARAMS, RISK_PARAMS, TRADING_RULES
import pandas as pd
import numpy as np

class OptimizedBacktester(TradingBacktester):
    def __init__(self):
        super().__init__(initial_balance=1000.0)
        self.params = OPTIMIZED_PARAMS
        self.risk_params = RISK_PARAMS
        self.trading_rules = TRADING_RULES
    
    def calculate_optimized_indicators(self, df):
        """Calculate indicators with optimized parameters"""
        try:
            indicators = {}
            
            # RSI with optimized period
            rsi_period = self.params['rsi_period']
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # EMA with optimized periods
            ema_fast = self.params['ema_fast']
            ema_slow = self.params['ema_slow']
            df['ema_fast'] = df['close'].ewm(span=ema_fast).mean()
            df['ema_slow'] = df['close'].ewm(span=ema_slow).mean()
            
            # Bollinger Bands with optimized parameters
            bb_period = self.params['bb_period']
            bb_std = self.params['bb_std']
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std_val = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            
            # MACD with standard parameters
            ema_12 = df['close'].ewm(span=self.params['macd_fast']).mean()
            ema_26 = df['close'].ewm(span=self.params['macd_slow']).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=self.params['macd_signal']).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # ATR for volatility
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators (using default volume since missing)
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating optimized indicators: {e}")
            return df
    
    def generate_optimized_signals(self, df):
        """Generate signals with optimized parameters"""
        signals = []
        
        for i in range(len(df)):
            if i < 50:  # Need enough data for indicators
                signals.append({'signal': 'HOLD', 'confidence': 0, 'reasoning': 'Insufficient data'})
                continue
            
            row = df.iloc[i]
            
            try:
                rsi = row['rsi']
                ema_fast = row['ema_fast']
                ema_slow = row['ema_slow']
                bb_upper = row['bb_upper']
                bb_lower = row['bb_lower']
                macd = row['macd']
                macd_signal = row['macd_signal']
                volume_ratio = row['volume_ratio']
                current_price = row['close']
                
                # Optimized scoring system
                buy_score = 0
                sell_score = 0
                
                # RSI with optimized thresholds
                rsi_oversold = self.params['rsi_oversold']
                rsi_overbought = self.params['rsi_overbought']
                
                if rsi < rsi_oversold:
                    buy_score += 3  # Strong buy signal
                elif rsi > rsi_overbought:
                    sell_score += 3  # Strong sell signal
                elif 45 < rsi < 65:  # Neutral zone
                    buy_score += 1
                    sell_score += 1
                
                # EMA crossover (faster EMAs)
                if ema_fast > ema_slow:
                    buy_score += 2
                elif ema_fast < ema_slow:
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
                
                # Volume confirmation (more lenient)
                if volume_ratio > self.trading_rules['min_volume_ratio']:
                    buy_score += 1
                    sell_score += 1
                
                # Trend strength bonus
                trend_strength = abs(ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0
                if trend_strength > self.trading_rules['trend_strength_min']:
                    if ema_fast > ema_slow:
                        buy_score += 1
                    else:
                        sell_score += 1
                
                # Determine signal with optimized confidence threshold
                confidence_threshold = self.params['confidence_threshold']
                
                if buy_score >= 4 and buy_score > sell_score:
                    signal_type = "BUY"
                    confidence = min(70 + (buy_score - 4) * 5, 95)
                elif sell_score >= 4 and sell_score > buy_score:
                    signal_type = "SELL"
                    confidence = min(70 + (sell_score - 4) * 5, 95)
                else:
                    signal_type = "HOLD"
                    confidence = 0
                
                # Apply confidence threshold
                if confidence < confidence_threshold:
                    signal_type = "HOLD"
                    confidence = 0
                
                reasoning = f"RSI:{rsi:.1f}, EMA:{ema_fast:.5f}/{ema_slow:.5f}, Vol:{volume_ratio:.1f}x, Score:{buy_score}/{sell_score}"
                
                signals.append({
                    'signal': signal_type,
                    'confidence': confidence,
                    'reasoning': reasoning
                })
                
            except Exception as e:
                signals.append({'signal': 'HOLD', 'confidence': 0, 'reasoning': f'Error: {e}'})
        
        # Add signals to dataframe
        df['signal'] = [s['signal'] for s in signals]
        df['confidence'] = [s['confidence'] for s in signals]
        df['reasoning'] = [s['reasoning'] for s in signals]
        
        return df
    
    def run_optimized_backtest(self, data_file):
        """Run backtest with optimized parameters"""
        print("🚀 Running Optimized SIRENUSDT Backtest...")
        
        # Load data
        data = self.load_data(data_file, "SIRENUSDT")
        if data.empty:
            print("❌ Failed to load data")
            return None
        
        print(f"📊 Loaded {len(data)} data points")
        print(f"📅 Period: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        # Calculate optimized indicators
        print("🔧 Calculating optimized indicators...")
        data = self.calculate_optimized_indicators(data)
        
        # Generate optimized signals
        print("🎯 Generating optimized signals...")
        data = self.generate_optimized_signals(data)
        
        # Count signals
        buy_signals = (data['signal'] == 'BUY').sum()
        sell_signals = (data['signal'] == 'SELL').sum()
        total_signals = buy_signals + sell_signals
        
        print(f"📈 Generated {total_signals} signals ({buy_signals} BUY, {sell_signals} SELL)")
        
        # Run backtest simulation
        print("⚡ Running backtest simulation...")
        
        # Reset state
        self.current_balance = self.initial_balance
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        
        # Process each data point
        for i in range(len(data)):
            row = data.iloc[i]
            self._process_optimized_tick(row, i)
            
            # Record equity
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'balance': self.current_balance,
                'equity': self._calculate_equity(row)
            })
        
        # Close any remaining positions
        self._close_all_positions(data.iloc[-1])
        
        # Calculate results
        results = self._calculate_results()
        
        print(f"✅ Backtest completed!")
        return results, data
    
    def _process_optimized_tick(self, row, index):
        """Process single data point with optimized logic"""
        timestamp = row['timestamp']
        symbol = row.get('symbol', 'SIRENUSDT')
        current_price = row['close']
        
        # Check existing positions for exits
        self._check_optimized_exits(row)
        
        # Check for new entries with optimized rules
        if row['signal'] in ['BUY', 'SELL'] and row['confidence'] >= self.params['confidence_threshold']:
            if symbol not in self.positions:  # No existing position
                # Additional market condition checks
                if self._is_market_suitable_optimized(row):
                    self._enter_position(row)
    
    def _check_optimized_exits(self, row):
        """Check position exits with optimized logic"""
        symbol = row.get('symbol', 'SIRENUSDT')
        current_price = row['close']
        
        if symbol in self.positions:
            position = self.positions[symbol]
            exit_reason = None
            
            # Standard stop loss and take profit
            if position['side'] == 'BUY' and current_price <= position['stop_loss']:
                exit_reason = 'SL'
            elif position['side'] == 'SELL' and current_price >= position['stop_loss']:
                exit_reason = 'SL'
            elif position['side'] == 'BUY' and current_price >= position['take_profit']:
                exit_reason = 'TP'
            elif position['side'] == 'SELL' and current_price <= position['take_profit']:
                exit_reason = 'TP'
            
            # Optimized signal reversal (faster exits)
            elif row['signal'] != 'HOLD' and row['signal'] != position['side'] and row['confidence'] >= 70:
                exit_reason = 'SIGNAL'
            
            # Quick profit taking for scalping (optimized)
            elif position['side'] == 'BUY':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                if pnl_pct > 0.3:  # Take profit at 0.3% for scalping
                    exit_reason = 'QUICK_PROFIT'
            elif position['side'] == 'SELL':
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price'] * 100
                if pnl_pct > 0.3:  # Take profit at 0.3% for scalping
                    exit_reason = 'QUICK_PROFIT'
            
            if exit_reason:
                self._exit_position(symbol, current_price, row['timestamp'], exit_reason)
    
    def _is_market_suitable_optimized(self, row):
        """Check if market conditions are suitable with optimized criteria"""
        try:
            # More lenient volatility check
            atr_pct = (row['atr'] / row['close']) * 100 if row['close'] > 0 else 0
            if atr_pct > self.trading_rules['max_atr_percent']:
                return False
            
            # Volume check (more lenient)
            if row['volume_ratio'] < self.trading_rules['min_volume_ratio']:
                return False
            
            return True
            
        except Exception as e:
            return False

def main():
    """Run optimized backtest"""
    print("=" * 80)
    print("OPTIMIZED SIRENUSDT BACKTEST")
    print("=" * 80)
    
    # Initialize optimized backtester
    backtester = OptimizedBacktester()
    
    # Run optimized backtest
    results, data = backtester.run_optimized_backtest("BYBIT_SIRENUSDT.P, 5_e106e.csv")
    
    if results:
        # Generate detailed report
        print("\n" + "=" * 80)
        print("OPTIMIZED BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"📊 PERFORMANCE SUMMARY")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Winning Trades: {results.winning_trades} ({results.win_rate:.1f}%)")
        print(f"   Losing Trades: {results.losing_trades}")
        
        print(f"\n💰 PROFIT & LOSS")
        print(f"   Total PnL: ${results.total_pnl:.2f}")
        print(f"   Total Return: {results.total_pnl_pct:.2f}%")
        print(f"   Best Trade: ${results.best_trade:.2f}")
        print(f"   Worst Trade: ${results.worst_trade:.2f}")
        
        print(f"\n📈 RISK METRICS")
        print(f"   Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   Profit Factor: {results.profit_factor:.2f}")
        
        print(f"\n⏱️  TIMING")
        print(f"   Avg Trade Duration: {results.avg_trade_duration}")
        
        # Compare with original results
        print(f"\n📊 IMPROVEMENT vs ORIGINAL:")
        print(f"   Original: 63 trades, 49.2% win rate, -0.01% return")
        print(f"   Optimized: {results.total_trades} trades, {results.win_rate:.1f}% win rate, {results.total_pnl_pct:.2f}% return")
        
        if results.total_trades > 0:
            improvement = results.total_pnl_pct - (-0.01)
            print(f"   📈 Return Improvement: +{improvement:.2f}%")
        
        # Save detailed report
        backtester.generate_report(results, 'optimized_backtest_report.txt')
        print(f"\n📄 Detailed report saved to 'optimized_backtest_report.txt'")
        
        # Show parameter summary
        print(f"\n🎯 OPTIMIZED PARAMETERS USED:")
        for key, value in backtester.params.items():
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()