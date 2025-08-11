#!/usr/bin/env python3
"""
Chunked backtesting engine that runs backtests on data segments
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

from data_chunker import DataChunker
from backtester import BacktestResults, Trade

class ChunkedBacktester:
    """
    Run backtests on data chunks to analyze performance across different market conditions
    """
    
    def __init__(self, data_file: str, config: Dict = None):
        """
        Initialize chunked backtester
        
        Args:
            data_file: Path to CSV data file
            config: Backtesting configuration
        """
        self.data_file = data_file
        self.config = config or self._default_config()
        self.data = pd.read_csv(data_file)
        self.chunker = DataChunker(self.data)
        self.results = []
        
    def _default_config(self) -> Dict:
        """Default backtesting configuration"""
        return {
            'initial_balance': 1000,
            'leverage': 10,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'position_size_pct': 0.1,
            'min_confidence': 0.6
        }
        
    def run_chunk_backtest(self, chunk: pd.DataFrame, chunk_id: int) -> Dict:
        """
        Run backtest on a single chunk
        
        Args:
            chunk: Data chunk to backtest
            chunk_id: Unique identifier for this chunk
            
        Returns:
            Dictionary with chunk backtest results
        """
        trades = []
        balance = self.config['initial_balance']
        position = None
        
        for i, row in chunk.iterrows():
            # Enhanced signal-based trading logic with multiple signal types
            if position is None:  # No open position
                # Look for entry signals - prioritize BOS over CHOCH
                if row.get('Bullish BOS', 0) == 1:
                    # Enter long position on Bullish BOS
                    entry_price = row['close']
                    position_size = balance * self.config['position_size_pct']
                    quantity = (position_size * self.config['leverage']) / entry_price
                    
                    position = {
                        'side': 'LONG',
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'entry_time': row['time'],
                        'stop_loss': entry_price * (1 - self.config['stop_loss_pct']),
                        'take_profit': entry_price * (1 + self.config['take_profit_pct']),
                        'entry_signal': 'Bullish BOS'
                    }
                    
                elif row.get('Bearish BOS', 0) == 1:
                    # Enter short position on Bearish BOS
                    entry_price = row['close']
                    position_size = balance * self.config['position_size_pct']
                    quantity = (position_size * self.config['leverage']) / entry_price
                    
                    position = {
                        'side': 'SHORT',
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'entry_time': row['time'],
                        'stop_loss': entry_price * (1 + self.config['stop_loss_pct']),
                        'take_profit': entry_price * (1 - self.config['take_profit_pct']),
                        'entry_signal': 'Bearish BOS'
                    }
                    
                elif row.get('Bullish CHOCH', 0) == 1:
                    # Enter long position on Bullish CHOCH (weaker signal)
                    entry_price = row['close']
                    position_size = balance * self.config['position_size_pct'] * 0.5  # Smaller position
                    quantity = (position_size * self.config['leverage']) / entry_price
                    
                    position = {
                        'side': 'LONG',
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'entry_time': row['time'],
                        'stop_loss': entry_price * (1 - self.config['stop_loss_pct']),
                        'take_profit': entry_price * (1 + self.config['take_profit_pct']),
                        'entry_signal': 'Bullish CHOCH'
                    }
                    
                elif row.get('Bearish CHOCH', 0) == 1:
                    # Enter short position on Bearish CHOCH (weaker signal)
                    entry_price = row['close']
                    position_size = balance * self.config['position_size_pct'] * 0.5  # Smaller position
                    quantity = (position_size * self.config['leverage']) / entry_price
                    
                    position = {
                        'side': 'SHORT',
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'entry_time': row['time'],
                        'stop_loss': entry_price * (1 + self.config['stop_loss_pct']),
                        'take_profit': entry_price * (1 - self.config['take_profit_pct']),
                        'entry_signal': 'Bearish CHOCH'
                    }
                    
            else:  # Position is open
                current_price = row['close']
                exit_reason = None
                
                # Check exit conditions
                if position['side'] == 'LONG':
                    if current_price <= position['stop_loss']:
                        exit_reason = 'SL'
                    elif current_price >= position['take_profit']:
                        exit_reason = 'TP'
                    elif row.get('Bearish BOS', 0) == 1:
                        exit_reason = 'Bearish BOS'
                    elif row.get('Bearish CHOCH', 0) == 1:
                        exit_reason = 'Bearish CHOCH'
                        
                elif position['side'] == 'SHORT':
                    if current_price >= position['stop_loss']:
                        exit_reason = 'SL'
                    elif current_price <= position['take_profit']:
                        exit_reason = 'TP'
                    elif row.get('Bullish BOS', 0) == 1:
                        exit_reason = 'Bullish BOS'
                    elif row.get('Bullish CHOCH', 0) == 1:
                        exit_reason = 'Bullish CHOCH'
                
                # Close position if exit condition met
                if exit_reason:
                    if position['side'] == 'LONG':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:  # SHORT
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    # Calculate position size used for this trade
                    if 'CHOCH' in position.get('entry_signal', ''):
                        position_size_used = balance * self.config['position_size_pct'] * 0.5
                    else:
                        position_size_used = balance * self.config['position_size_pct']
                    
                    pnl_pct = pnl / position_size_used
                    balance += pnl
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': row['time'],
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'entry_signal': position.get('entry_signal', 'Unknown')
                    }
                    
                    trades.append(trade)
                    position = None
        
        # Calculate chunk statistics
        if trades:
            total_pnl = sum(trade['pnl'] for trade in trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / len(trades)
            
            chunk_result = {
                'chunk_id': chunk_id,
                'start_time': chunk['time'].iloc[0],
                'end_time': chunk['time'].iloc[-1],
                'rows': len(chunk),
                'trades': len(trades),
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'final_balance': balance,
                'return_pct': (balance - self.config['initial_balance']) / self.config['initial_balance'] * 100,
                'trade_details': trades
            }
        else:
            chunk_result = {
                'chunk_id': chunk_id,
                'start_time': chunk['time'].iloc[0],
                'end_time': chunk['time'].iloc[-1],
                'rows': len(chunk),
                'trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'final_balance': self.config['initial_balance'],
                'return_pct': 0,
                'trade_details': []
            }
            
        return chunk_result
        
    def run_chunked_backtest(self, chunk_method: str = 'rows', **kwargs) -> List[Dict]:
        """
        Run backtest across all chunks
        
        Args:
            chunk_method: 'rows', 'time', 'signals', or 'volatility'
            **kwargs: Parameters for chunking method
            
        Returns:
            List of chunk results
        """
        # Get chunks based on method
        if chunk_method == 'rows':
            chunks = list(self.chunker.chunk_by_rows(kwargs.get('chunk_size', 100)))
        elif chunk_method == 'time':
            chunks = list(self.chunker.chunk_by_time(kwargs.get('time_window', '1D')))
        elif chunk_method == 'signals':
            chunks = list(self.chunker.chunk_by_signals(kwargs.get('signal_columns')))
        elif chunk_method == 'volatility':
            chunks = list(self.chunker.chunk_by_volatility())
        else:
            raise ValueError(f"Unknown chunk method: {chunk_method}")
            
        print(f"Running backtest on {len(chunks)} chunks using {chunk_method} method...")
        
        # Run backtest on each chunk
        results = []
        for i, chunk in enumerate(chunks):
            result = self.run_chunk_backtest(chunk, i)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks")
                
        self.results = results
        return results
        
    def analyze_results(self) -> Dict:
        """
        Analyze results across all chunks
        
        Returns:
            Summary statistics
        """
        if not self.results:
            return {}
            
        # Aggregate statistics
        total_trades = sum(r['trades'] for r in self.results)
        total_winning = sum(r['winning_trades'] for r in self.results)
        total_pnl = sum(r['total_pnl'] for r in self.results)
        
        chunk_returns = [r['return_pct'] for r in self.results if r['trades'] > 0]
        win_rates = [r['win_rate'] for r in self.results if r['trades'] > 0]
        
        analysis = {
            'total_chunks': len(self.results),
            'profitable_chunks': len([r for r in self.results if r['return_pct'] > 0]),
            'total_trades': total_trades,
            'overall_win_rate': total_winning / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_chunk_return': np.mean(chunk_returns) if chunk_returns else 0,
            'best_chunk_return': max(chunk_returns) if chunk_returns else 0,
            'worst_chunk_return': min(chunk_returns) if chunk_returns else 0,
            'return_std': np.std(chunk_returns) if chunk_returns else 0,
            'win_rate_consistency': np.std(win_rates) if win_rates else 0
        }
        
        return analysis
        
    def plot_chunk_performance(self, save_path: str = None):
        """
        Plot chunk performance visualization
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            print("No results to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Chunk returns
        returns = [r['return_pct'] for r in self.results]
        axes[0, 0].bar(range(len(returns)), returns)
        axes[0, 0].set_title('Return % by Chunk')
        axes[0, 0].set_xlabel('Chunk ID')
        axes[0, 0].set_ylabel('Return %')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Win rates
        win_rates = [r['win_rate'] * 100 for r in self.results if r['trades'] > 0]
        axes[0, 1].hist(win_rates, bins=20, alpha=0.7)
        axes[0, 1].set_title('Win Rate Distribution')
        axes[0, 1].set_xlabel('Win Rate %')
        axes[0, 1].set_ylabel('Frequency')
        
        # Cumulative PnL
        cumulative_pnl = np.cumsum([r['total_pnl'] for r in self.results])
        axes[1, 0].plot(cumulative_pnl)
        axes[1, 0].set_title('Cumulative PnL Across Chunks')
        axes[1, 0].set_xlabel('Chunk ID')
        axes[1, 0].set_ylabel('Cumulative PnL')
        
        # Trades per chunk
        trades_per_chunk = [r['trades'] for r in self.results]
        axes[1, 1].bar(range(len(trades_per_chunk)), trades_per_chunk)
        axes[1, 1].set_title('Trades per Chunk')
        axes[1, 1].set_xlabel('Chunk ID')
        axes[1, 1].set_ylabel('Number of Trades')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def save_results(self, filename: str):
        """
        Save results to JSON file
        
        Args:
            filename: Output filename
        """
        output = {
            'config': self.config,
            'data_file': self.data_file,
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'analysis': self.analyze_results()
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
            
        print(f"Results saved to {filename}")

def main():
    """
    Example usage of chunked backtester
    """
    # Initialize backtester
    backtester = ChunkedBacktester('BYBIT_SIRENUSDT.P, 5_e106e.csv')
    
    # Test different chunking methods
    methods = [
        ('rows', {'chunk_size': 100}),
        ('time', {'time_window': '1D'}),
        ('signals', {}),
    ]
    
    for method, params in methods:
        print(f"\n=== Testing {method} chunking ===")
        
        # Run backtest
        results = backtester.run_chunked_backtest(method, **params)
        
        # Analyze results
        analysis = backtester.analyze_results()
        
        print(f"Total chunks: {analysis['total_chunks']}")
        print(f"Profitable chunks: {analysis['profitable_chunks']}")
        print(f"Total trades: {analysis['total_trades']}")
        print(f"Overall win rate: {analysis['overall_win_rate']:.2%}")
        print(f"Average chunk return: {analysis['avg_chunk_return']:.2f}%")
        print(f"Best chunk return: {analysis['best_chunk_return']:.2f}%")
        print(f"Worst chunk return: {analysis['worst_chunk_return']:.2f}%")
        
        # Save results
        backtester.save_results(f'chunked_backtest_{method}.json')
        
        # Plot performance
        backtester.plot_chunk_performance(f'chunk_performance_{method}.png')

if __name__ == "__main__":
    main()