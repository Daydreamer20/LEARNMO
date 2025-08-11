#!/usr/bin/env python3
"""
Trading bot parameter optimization engine
"""

import pandas as pd
import numpy as np
import json
import logging
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from backtester import TradingBacktester, BacktestResults
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ParameterOptimizer:
    def __init__(self, data_file: str, symbol: str = "SIRENUSDT"):
        self.data_file = data_file
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)
        
        # Load and prepare data
        self.backtester = TradingBacktester()
        self.data = self.backtester.load_data(data_file, symbol)
        
        if self.data.empty:
            raise ValueError("Failed to load data")
        
        self.logger.info(f"Loaded {len(self.data)} data points for optimization")
    
    def optimize_technical_strategy(self, 
                                  rsi_periods: List[int] = [10, 14, 18, 21],
                                  rsi_oversold: List[int] = [20, 25, 30, 35],
                                  rsi_overbought: List[int] = [65, 70, 75, 80],
                                  ema_fast: List[int] = [5, 9, 12, 15],
                                  ema_slow: List[int] = [18, 21, 26, 30],
                                  bb_periods: List[int] = [15, 20, 25],
                                  bb_std: List[float] = [1.5, 2.0, 2.5],
                                  confidence_threshold: List[int] = [75, 80, 85, 90],
                                  max_workers: int = 4) -> pd.DataFrame:
        """Optimize technical analysis strategy parameters"""
        
        # Generate parameter combinations
        param_combinations = list(product(
            rsi_periods, rsi_oversold, rsi_overbought,
            ema_fast, ema_slow, bb_periods, bb_std,
            confidence_threshold
        ))
        
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        results = []
        
        # Test each combination
        for i, params in enumerate(param_combinations):
            if i % 100 == 0:
                self.logger.info(f"Progress: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.1f}%)")
            
            try:
                result = self._test_parameters(params)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error testing parameters {params}: {e}")
        
        # Convert to DataFrame and sort by performance
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        self.logger.info("Optimization completed!")
        return results_df
    
    def _test_parameters(self, params) -> Dict:
        """Test single parameter combination"""
        (rsi_period, rsi_oversold, rsi_overbought, 
         ema_fast, ema_slow, bb_period, bb_std, conf_threshold) = params
        
        # Skip invalid combinations
        if ema_fast >= ema_slow:
            return self._empty_result(params)
        
        if rsi_oversold >= rsi_overbought:
            return self._empty_result(params)
        
        # Create custom backtester with these parameters
        backtester = TradingBacktester()
        
        # Modify the data with custom indicators
        data = self.data.copy()
        data = self._calculate_custom_indicators(data, rsi_period, ema_fast, ema_slow, bb_period, bb_std)
        data = self._generate_custom_signals(data, rsi_oversold, rsi_overbought, conf_threshold)
        
        # Run backtest
        try:
            results = backtester.run_backtest(data, strategy='custom')
            
            return {
                'rsi_period': rsi_period,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'bb_period': bb_period,
                'bb_std': bb_std,
                'confidence_threshold': conf_threshold,
                'total_trades': results.total_trades,
                'win_rate': results.win_rate,
                'total_return': results.total_pnl_pct,
                'max_drawdown': results.max_drawdown,
                'sharpe_ratio': results.sharpe_ratio,
                'profit_factor': results.profit_factor,
                'avg_trade_duration_hours': results.avg_trade_duration.total_seconds() / 3600,
                'best_trade': results.best_trade,
                'worst_trade': results.worst_trade
            }
        except Exception as e:
            return self._empty_result(params)
    
    def _empty_result(self, params) -> Dict:
        """Return empty result for invalid parameters"""
        return {
            'rsi_period': params[0],
            'rsi_oversold': params[1],
            'rsi_overbought': params[2],
            'ema_fast': params[3],
            'ema_slow': params[4],
            'bb_period': params[5],
            'bb_std': params[6],
            'confidence_threshold': params[7],
            'total_trades': 0,
            'win_rate': 0,
            'total_return': -100,
            'max_drawdown': 100,
            'sharpe_ratio': -10,
            'profit_factor': 0,
            'avg_trade_duration_hours': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
    
    def _calculate_custom_indicators(self, df, rsi_period, ema_fast, ema_slow, bb_period, bb_std):
        """Calculate indicators with custom parameters"""
        # RSI with custom period
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA with custom periods
        df['ema_fast'] = df['close'].ewm(span=ema_fast).mean()
        df['ema_slow'] = df['close'].ewm(span=ema_slow).mean()
        
        # Bollinger Bands with custom parameters
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        
        # MACD (standard)
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volume ratio
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _generate_custom_signals(self, df, rsi_oversold, rsi_overbought, conf_threshold):
        """Generate signals with custom parameters"""
        signals = []
        
        for i in range(len(df)):
            if i < 50:  # Need enough data
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
                
                # Custom scoring with optimized parameters
                buy_score = 0
                sell_score = 0
                
                # RSI with custom thresholds
                if rsi < rsi_oversold:
                    buy_score += 3
                elif rsi > rsi_overbought:
                    sell_score += 3
                elif 40 < rsi < 60:
                    buy_score += 1
                    sell_score += 1
                
                # EMA crossover
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
                
                # Volume
                if volume_ratio > 1.5:
                    buy_score += 1
                    sell_score += 1
                
                # Determine signal
                if buy_score >= 4 and buy_score > sell_score:
                    signal_type = "BUY"
                    confidence = min(80 + (buy_score - 4) * 3, 95)
                elif sell_score >= 4 and sell_score > buy_score:
                    signal_type = "SELL"
                    confidence = min(80 + (sell_score - 4) * 3, 95)
                else:
                    signal_type = "HOLD"
                    confidence = 0
                
                # Apply confidence threshold
                if confidence < conf_threshold:
                    signal_type = "HOLD"
                    confidence = 0
                
                reasoning = f"RSI:{rsi:.1f}, EMA:{'Bull' if ema_fast>ema_slow else 'Bear'}, Vol:{volume_ratio:.1f}x"
                
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
    
    def analyze_results(self, results_df: pd.DataFrame, top_n: int = 10):
        """Analyze optimization results"""
        print("=" * 80)
        print("PARAMETER OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Top performing parameters
        print(f"\n🏆 TOP {top_n} PARAMETER COMBINATIONS:")
        print("-" * 80)
        
        top_results = results_df.head(top_n)
        
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"\n#{i} - Return: {row['total_return']:.2f}% | Win Rate: {row['win_rate']:.1f}% | Trades: {row['total_trades']}")
            print(f"    RSI: {row['rsi_period']} period, {row['rsi_oversold']}/{row['rsi_overbought']} levels")
            print(f"    EMA: {row['ema_fast']}/{row['ema_slow']} periods")
            print(f"    BB: {row['bb_period']} period, {row['bb_std']} std")
            print(f"    Confidence: {row['confidence_threshold']}%")
            print(f"    Max DD: {row['max_drawdown']:.2f}% | Sharpe: {row['sharpe_ratio']:.2f}")
        
        # Parameter analysis
        print(f"\n📊 PARAMETER ANALYSIS:")
        print("-" * 80)
        
        # Best parameters by category
        profitable_results = results_df[results_df['total_return'] > 0]
        
        if len(profitable_results) > 0:
            print(f"Profitable strategies: {len(profitable_results)}/{len(results_df)} ({len(profitable_results)/len(results_df)*100:.1f}%)")
            
            print(f"\nBest RSI periods: {profitable_results['rsi_period'].mode().values}")
            print(f"Best RSI oversold levels: {profitable_results['rsi_oversold'].mode().values}")
            print(f"Best RSI overbought levels: {profitable_results['rsi_overbought'].mode().values}")
            print(f"Best EMA fast periods: {profitable_results['ema_fast'].mode().values}")
            print(f"Best EMA slow periods: {profitable_results['ema_slow'].mode().values}")
            print(f"Best BB periods: {profitable_results['bb_period'].mode().values}")
            print(f"Best BB std: {profitable_results['bb_std'].mode().values}")
            print(f"Best confidence thresholds: {profitable_results['confidence_threshold'].mode().values}")
        else:
            print("No profitable strategies found. Consider:")
            print("- Adjusting parameter ranges")
            print("- Using different data period")
            print("- Modifying strategy logic")
        
        return top_results
    
    def save_results(self, results_df: pd.DataFrame, filename: str = "optimization_results.csv"):
        """Save optimization results"""
        results_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to {filename}")
        
        # Save best parameters as JSON
        if len(results_df) > 0:
            best_params = results_df.iloc[0].to_dict()
            
            # Clean up for JSON
            for key, value in best_params.items():
                if pd.isna(value):
                    best_params[key] = None
                elif isinstance(value, np.integer):
                    best_params[key] = int(value)
                elif isinstance(value, np.floating):
                    best_params[key] = float(value)
            
            with open("best_parameters.json", "w") as f:
                json.dump(best_params, f, indent=2)
            
            print("💾 Best parameters saved to best_parameters.json")
    
    def create_visualizations(self, results_df: pd.DataFrame):
        """Create optimization result visualizations"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Parameter Optimization Results', fontsize=16)
        
        # 1. Return vs Win Rate scatter
        axes[0, 0].scatter(results_df['win_rate'], results_df['total_return'], alpha=0.6)
        axes[0, 0].set_xlabel('Win Rate (%)')
        axes[0, 0].set_ylabel('Total Return (%)')
        axes[0, 0].set_title('Return vs Win Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Return vs Max Drawdown
        axes[0, 1].scatter(results_df['max_drawdown'], results_df['total_return'], alpha=0.6, color='red')
        axes[0, 1].set_xlabel('Max Drawdown (%)')
        axes[0, 1].set_ylabel('Total Return (%)')
        axes[0, 1].set_title('Return vs Max Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sharpe Ratio distribution
        axes[0, 2].hist(results_df['sharpe_ratio'], bins=30, alpha=0.7, color='green')
        axes[0, 2].set_xlabel('Sharpe Ratio')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Sharpe Ratio Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. RSI Period vs Return
        rsi_returns = results_df.groupby('rsi_period')['total_return'].mean()
        axes[1, 0].bar(rsi_returns.index, rsi_returns.values, alpha=0.7)
        axes[1, 0].set_xlabel('RSI Period')
        axes[1, 0].set_ylabel('Avg Return (%)')
        axes[1, 0].set_title('RSI Period vs Average Return')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. EMA Fast vs Return
        ema_returns = results_df.groupby('ema_fast')['total_return'].mean()
        axes[1, 1].bar(ema_returns.index, ema_returns.values, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('EMA Fast Period')
        axes[1, 1].set_ylabel('Avg Return (%)')
        axes[1, 1].set_title('EMA Fast Period vs Average Return')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Confidence Threshold vs Return
        conf_returns = results_df.groupby('confidence_threshold')['total_return'].mean()
        axes[1, 2].bar(conf_returns.index, conf_returns.values, alpha=0.7, color='purple')
        axes[1, 2].set_xlabel('Confidence Threshold (%)')
        axes[1, 2].set_ylabel('Avg Return (%)')
        axes[1, 2].set_title('Confidence Threshold vs Average Return')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualizations saved as optimization_results.png")

def main():
    """Main optimization function"""
    print("=" * 80)
    print("TRADING BOT PARAMETER OPTIMIZER")
    print("=" * 80)
    
    # Get data file from user
    data_file = input("Enter path to your data file: ").strip()
    
    if not data_file:
        print("No data file specified. Exiting.")
        return
    
    try:
        # Initialize optimizer
        optimizer = ParameterOptimizer(data_file)
        
        # Run optimization
        print("\n🚀 Starting parameter optimization...")
        results = optimizer.optimize_technical_strategy()
        
        # Analyze results
        top_results = optimizer.analyze_results(results)
        
        # Save results
        optimizer.save_results(results)
        
        # Create visualizations
        optimizer.create_visualizations(results)
        
        print("\n✅ Optimization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        logging.error(f"Optimization error: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()