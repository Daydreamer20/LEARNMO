#!/usr/bin/env python3
"""
Weights & Biases integration for trading bot training and backtesting
"""

import wandb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from chunked_backtester import ChunkedBacktester
from data_chunker import DataChunker

class WandbTradingTracker:
    """
    Weights & Biases integration for trading bot experiments
    """
    
    def __init__(self, project_name: str = "trading-bot-optimization", entity: str = None):
        """
        Initialize wandb tracking
        
        Args:
            project_name: Name of the wandb project
            entity: wandb entity (username/team)
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None
        
    def start_experiment(self, 
                        experiment_name: str,
                        config: Dict,
                        tags: List[str] = None,
                        notes: str = None):
        """
        Start a new wandb experiment
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
            tags: List of tags for the experiment
            notes: Notes about the experiment
            
        Returns:
            wandb Run object
        """
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=experiment_name,
            config=config,
            tags=tags or [],
            notes=notes,
            reinit=True
        )
        
        return self.run
    
    def log_backtest_results(self, 
                           results: List[Dict], 
                           analysis: Dict,
                           strategy_name: str):
        """
        Log backtest results to wandb
        
        Args:
            results: List of chunk results
            analysis: Analysis summary
            strategy_name: Name of the strategy
        """
        if not self.run:
            raise ValueError("No active wandb run. Call start_experiment first.")
        
        # Log overall metrics
        wandb.log({
            "strategy": strategy_name,
            "total_chunks": analysis['total_chunks'],
            "profitable_chunks": analysis['profitable_chunks'],
            "profitability_rate": analysis['profitable_chunks'] / analysis['total_chunks'],
            "total_trades": analysis['total_trades'],
            "overall_win_rate": analysis['overall_win_rate'],
            "total_pnl": analysis['total_pnl'],
            "avg_chunk_return": analysis['avg_chunk_return'],
            "best_chunk_return": analysis['best_chunk_return'],
            "worst_chunk_return": analysis['worst_chunk_return'],
            "return_std": analysis['return_std'],
            "sharpe_ratio": analysis['avg_chunk_return'] / max(analysis['return_std'], 0.01),
            "profit_factor": self._calculate_profit_factor(results)
        })
        
        # Log chunk-by-chunk performance
        chunk_data = []
        for i, chunk_result in enumerate(results):
            chunk_data.append({
                "chunk_id": i,
                "return_pct": chunk_result['return_pct'],
                "trades": chunk_result['trades'],
                "win_rate": chunk_result['win_rate'],
                "pnl": chunk_result['total_pnl']
            })
        
        # Create wandb table for chunk results
        chunk_table = wandb.Table(
            columns=["chunk_id", "return_pct", "trades", "win_rate", "pnl"],
            data=[[row["chunk_id"], row["return_pct"], row["trades"], 
                   row["win_rate"], row["pnl"]] for row in chunk_data]
        )
        wandb.log({"chunk_performance": chunk_table})
        
        # Log performance charts
        self._log_performance_charts(results, strategy_name)
    
    def log_signal_analysis(self, signal_stats: Dict):
        """
        Log signal performance analysis
        
        Args:
            signal_stats: Dictionary with signal performance statistics
        """
        if not self.run:
            raise ValueError("No active wandb run. Call start_experiment first.")
        
        # Log signal metrics
        for signal_name, stats in signal_stats.items():
            wandb.log({
                f"signal_{signal_name.lower().replace(' ', '_')}_trades": stats['total_trades'],
                f"signal_{signal_name.lower().replace(' ', '_')}_win_rate": stats['win_rate'],
                f"signal_{signal_name.lower().replace(' ', '_')}_avg_pnl": stats['avg_pnl'],
                f"signal_{signal_name.lower().replace(' ', '_')}_total_pnl": stats['total_pnl']
            })
        
        # Create signal comparison table
        signal_data = []
        for signal_name, stats in signal_stats.items():
            signal_data.append([
                signal_name,
                stats['total_trades'],
                stats['win_rate'],
                stats['avg_pnl'],
                stats['total_pnl']
            ])
        
        signal_table = wandb.Table(
            columns=["signal", "trades", "win_rate", "avg_pnl", "total_pnl"],
            data=signal_data
        )
        wandb.log({"signal_performance": signal_table})
    
    def log_hyperparameter_sweep(self, 
                                param_name: str, 
                                param_value: Any, 
                                performance_metric: float):
        """
        Log hyperparameter sweep results
        
        Args:
            param_name: Name of the parameter
            param_value: Value of the parameter
            performance_metric: Performance metric (e.g., total PnL)
        """
        wandb.log({
            f"param_{param_name}": param_value,
            "performance": performance_metric
        })
    
    def log_trade_analysis(self, trades: List[Dict]):
        """
        Log detailed trade analysis
        
        Args:
            trades: List of trade dictionaries
        """
        if not trades:
            return
            
        # Aggregate trade statistics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        trade_stats = {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades) if trades else 0,
            "avg_win": np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            "avg_loss": np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            "largest_win": max([t['pnl'] for t in trades]) if trades else 0,
            "largest_loss": min([t['pnl'] for t in trades]) if trades else 0,
            "avg_trade_duration": np.mean([
                (pd.to_datetime(t['exit_time'], unit='s') - 
                 pd.to_datetime(t['entry_time'], unit='s')).total_seconds() / 60
                for t in trades
            ]) if trades else 0
        }
        
        wandb.log(trade_stats)
        
        # Create trades table
        trade_data = []
        for trade in trades[:100]:  # Limit to first 100 trades to avoid large tables
            trade_data.append([
                pd.to_datetime(trade['entry_time'], unit='s').strftime('%Y-%m-%d %H:%M'),
                trade['side'],
                trade['entry_price'],
                trade['exit_price'],
                trade['pnl'],
                trade['pnl_pct'],
                trade['exit_reason'],
                trade.get('entry_signal', 'Unknown')
            ])
        
        trades_table = wandb.Table(
            columns=["entry_time", "side", "entry_price", "exit_price", 
                    "pnl", "pnl_pct", "exit_reason", "entry_signal"],
            data=trade_data
        )
        wandb.log({"trades_detail": trades_table})
    
    def _calculate_profit_factor(self, results: List[Dict]) -> float:
        """Calculate profit factor from results"""
        all_trades = []
        for chunk_result in results:
            all_trades.extend(chunk_result.get('trade_details', []))
        
        if not all_trades:
            return 0
        
        gross_profit = sum(t['pnl'] for t in all_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in all_trades if t['pnl'] < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _log_performance_charts(self, results: List[Dict], strategy_name: str):
        """Create and log performance visualization charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Chunk returns
        returns = [r['return_pct'] for r in results]
        axes[0, 0].bar(range(len(returns)), returns, alpha=0.7)
        axes[0, 0].set_title(f'{strategy_name} - Chunk Returns')
        axes[0, 0].set_xlabel('Chunk ID')
        axes[0, 0].set_ylabel('Return %')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Cumulative PnL
        cumulative_pnl = np.cumsum([r['total_pnl'] for r in results])
        axes[0, 1].plot(cumulative_pnl, linewidth=2)
        axes[0, 1].set_title('Cumulative PnL')
        axes[0, 1].set_xlabel('Chunk ID')
        axes[0, 1].set_ylabel('Cumulative PnL ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Win rates
        win_rates = [r['win_rate'] * 100 for r in results if r['trades'] > 0]
        if win_rates:
            axes[1, 0].hist(win_rates, bins=10, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Win Rate Distribution')
            axes[1, 0].set_xlabel('Win Rate %')
            axes[1, 0].set_ylabel('Frequency')
        
        # Trades per chunk
        trades_per_chunk = [r['trades'] for r in results]
        axes[1, 1].bar(range(len(trades_per_chunk)), trades_per_chunk, alpha=0.7)
        axes[1, 1].set_title('Trades per Chunk')
        axes[1, 1].set_xlabel('Chunk ID')
        axes[1, 1].set_ylabel('Number of Trades')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({f"{strategy_name}_performance_chart": wandb.Image(fig)})
        plt.close(fig)
    
    def finish_experiment(self):
        """Finish the current wandb run"""
        if self.run:
            wandb.finish()
            self.run = None

class WandbBacktester(ChunkedBacktester):
    """
    Enhanced ChunkedBacktester with wandb integration
    """
    
    def __init__(self, data_file: str, config: Dict = None, wandb_tracker: WandbTradingTracker = None):
        super().__init__(data_file, config)
        self.wandb_tracker = wandb_tracker
    
    def run_wandb_experiment(self, 
                           experiment_name: str,
                           chunking_strategies: List[Dict],
                           tags: List[str] = None,
                           notes: str = None) -> Dict:
        """
        Run multiple chunking strategies and log to wandb
        
        Args:
            experiment_name: Name of the experiment
            chunking_strategies: List of chunking strategy configurations
            tags: Tags for the experiment
            notes: Notes about the experiment
            
        Returns:
            Dictionary with all results
        """
        if not self.wandb_tracker:
            raise ValueError("WandbTradingTracker not provided")
        
        # Start wandb experiment
        config = {
            "data_file": self.data_file,
            "trading_config": self.config,
            "strategies": [s['name'] for s in chunking_strategies],
            "total_strategies": len(chunking_strategies)
        }
        
        self.wandb_tracker.start_experiment(
            experiment_name=experiment_name,
            config=config,
            tags=tags,
            notes=notes
        )
        
        all_results = {}
        
        for strategy in chunking_strategies:
            print(f"Running strategy: {strategy['name']}")
            
            # Run backtest
            results = self.run_chunked_backtest(
                strategy['method'],
                **strategy.get('params', {})
            )
            
            # Analyze results
            analysis = self.analyze_results()
            
            # Log to wandb
            self.wandb_tracker.log_backtest_results(
                results, analysis, strategy['name']
            )
            
            # Collect all trades for signal analysis
            all_trades = []
            for chunk_result in results:
                all_trades.extend(chunk_result.get('trade_details', []))
            
            if all_trades:
                self.wandb_tracker.log_trade_analysis(all_trades)
            
            all_results[strategy['name']] = {
                'results': results,
                'analysis': analysis,
                'trades': all_trades
            }
        
        # Analyze signal performance across all strategies
        signal_stats = self._analyze_signal_performance(all_results)
        if signal_stats:
            self.wandb_tracker.log_signal_analysis(signal_stats)
        
        return all_results
    
    def _analyze_signal_performance(self, all_results: Dict) -> Dict:
        """Analyze signal performance across all strategies"""
        signal_stats = {}
        
        for strategy_name, data in all_results.items():
            for trade in data['trades']:
                signal = trade.get('entry_signal', 'Unknown')
                
                if signal not in signal_stats:
                    signal_stats[signal] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_pnl': 0
                    }
                
                signal_stats[signal]['total_trades'] += 1
                signal_stats[signal]['total_pnl'] += trade['pnl']
                
                if trade['pnl'] > 0:
                    signal_stats[signal]['winning_trades'] += 1
        
        # Calculate derived metrics
        for signal, stats in signal_stats.items():
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
            stats['avg_pnl'] = stats['total_pnl'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        
        return signal_stats

def run_hyperparameter_sweep():
    """
    Example hyperparameter sweep with wandb
    """
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'total_pnl',
            'goal': 'maximize'
        },
        'parameters': {
            'stop_loss_pct': {
                'min': 0.01,
                'max': 0.03
            },
            'take_profit_pct': {
                'min': 0.02,
                'max': 0.06
            },
            'position_size_pct': {
                'min': 0.05,
                'max': 0.2
            },
            'chunk_size': {
                'values': [50, 75, 100, 150, 200]
            }
        }
    }
    
    def train():
        # Initialize wandb
        wandb.init()
        config = wandb.config
        
        # Create trading config
        trading_config = {
            'initial_balance': 1000,
            'leverage': 10,
            'stop_loss_pct': config.stop_loss_pct,
            'take_profit_pct': config.take_profit_pct,
            'position_size_pct': config.position_size_pct
        }
        
        # Run backtest
        backtester = ChunkedBacktester('BYBIT_SOONUSDT.P, 3_08827.csv', trading_config)
        results = backtester.run_chunked_backtest('rows', chunk_size=config.chunk_size)
        analysis = backtester.analyze_results()
        
        # Log results
        wandb.log({
            'total_pnl': analysis['total_pnl'],
            'win_rate': analysis['overall_win_rate'],
            'avg_return': analysis['avg_chunk_return'],
            'return_std': analysis['return_std'],
            'sharpe_ratio': analysis['avg_chunk_return'] / max(analysis['return_std'], 0.01)
        })
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="trading-bot-hyperparameter-sweep")
    
    # Run sweep
    wandb.agent(sweep_id, train, count=50)  # Run 50 experiments

def main():
    """
    Example usage of wandb integration
    """
    # Initialize wandb tracker
    tracker = WandbTradingTracker(
        project_name="soonusdt-chunking-analysis",
        entity=None  # Use your wandb username here
    )
    
    # Initialize enhanced backtester
    backtester = WandbBacktester(
        'BYBIT_SOONUSDT.P, 3_08827.csv',
        wandb_tracker=tracker
    )
    
    # Define strategies to test
    strategies = [
        {
            'name': 'Fixed_75_Candles',
            'method': 'rows',
            'params': {'chunk_size': 75}
        },
        {
            'name': 'Fixed_150_Candles',
            'method': 'rows',
            'params': {'chunk_size': 150}
        },
        {
            'name': 'Time_6H',
            'method': 'time',
            'params': {'time_window': '6H'}
        },
        {
            'name': 'BOS_Signals',
            'method': 'signals',
            'params': {'signal_columns': ['Bullish BOS', 'Bearish BOS']}
        }
    ]
    
    # Run experiment
    results = backtester.run_wandb_experiment(
        experiment_name="SOONUSDT_Chunking_Comparison",
        chunking_strategies=strategies,
        tags=["chunking", "soonusdt", "backtesting"],
        notes="Comparing different chunking strategies for SOONUSDT trading"
    )
    
    # Finish experiment
    tracker.finish_experiment()
    
    print("Experiment completed! Check your wandb dashboard for results.")

if __name__ == "__main__":
    main()