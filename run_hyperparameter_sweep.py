#!/usr/bin/env python3
"""
Hyperparameter optimization using Weights & Biases sweeps
"""

import wandb
from chunked_backtester import ChunkedBacktester
import numpy as np

def objective_function():
    """
    Objective function for hyperparameter optimization
    This function will be called by wandb sweep
    """
    # Initialize wandb run
    wandb.init()
    config = wandb.config
    
    # Create trading configuration from sweep parameters
    trading_config = {
        'initial_balance': 1000,
        'leverage': config.leverage,
        'stop_loss_pct': config.stop_loss_pct,
        'take_profit_pct': config.take_profit_pct,
        'position_size_pct': config.position_size_pct
    }
    
    try:
        # Initialize backtester
        backtester = ChunkedBacktester('BYBIT_SOONUSDT.P, 3_08827.csv', trading_config)
        
        # Run backtest with specified chunking method
        if config.chunking_method == 'rows':
            results = backtester.run_chunked_backtest('rows', chunk_size=config.chunk_size)
        elif config.chunking_method == 'time':
            time_windows = ['1H', '2H', '4H', '6H', '12H', '1D']
            time_window = time_windows[min(config.time_window_idx, len(time_windows)-1)]
            results = backtester.run_chunked_backtest('time', time_window=time_window)
        elif config.chunking_method == 'signals':
            results = backtester.run_chunked_backtest('signals')
        else:
            results = backtester.run_chunked_backtest('volatility')
        
        # Analyze results
        analysis = backtester.analyze_results()
        
        # Calculate additional metrics
        profit_factor = calculate_profit_factor(results)
        max_drawdown = calculate_max_drawdown(results)
        sharpe_ratio = analysis['avg_chunk_return'] / max(analysis['return_std'], 0.01)
        
        # Log metrics to wandb
        metrics = {
            'total_pnl': analysis['total_pnl'],
            'win_rate': analysis['overall_win_rate'],
            'avg_chunk_return': analysis['avg_chunk_return'],
            'return_std': analysis['return_std'],
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_trades': analysis['total_trades'],
            'profitable_chunks': analysis['profitable_chunks'],
            'total_chunks': analysis['total_chunks'],
            'profitability_rate': analysis['profitable_chunks'] / analysis['total_chunks'],
            
            # Risk-adjusted metrics
            'calmar_ratio': analysis['avg_chunk_return'] / max(abs(max_drawdown), 0.01),
            'risk_adjusted_return': analysis['total_pnl'] / max(analysis['return_std'], 0.01),
            
            # Consistency metrics
            'best_chunk': analysis['best_chunk_return'],
            'worst_chunk': analysis['worst_chunk_return'],
            'return_range': analysis['best_chunk_return'] - analysis['worst_chunk_return']
        }
        
        wandb.log(metrics)
        
        # Log trade-level analysis
        all_trades = []
        for chunk_result in results:
            all_trades.extend(chunk_result.get('trade_details', []))
        
        if all_trades:
            trade_metrics = analyze_trades(all_trades)
            wandb.log(trade_metrics)
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        # Log error metrics
        wandb.log({
            'total_pnl': -1000,  # Penalty for failed runs
            'win_rate': 0,
            'error': str(e)
        })

def calculate_profit_factor(results):
    """Calculate profit factor from results"""
    all_trades = []
    for chunk_result in results:
        all_trades.extend(chunk_result.get('trade_details', []))
    
    if not all_trades:
        return 0
    
    gross_profit = sum(t['pnl'] for t in all_trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in all_trades if t['pnl'] < 0))
    
    return gross_profit / gross_loss if gross_loss > 0 else float('inf')

def calculate_max_drawdown(results):
    """Calculate maximum drawdown"""
    cumulative_pnl = np.cumsum([r['total_pnl'] for r in results])
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - running_max
    return float(np.min(drawdown))

def analyze_trades(trades):
    """Analyze individual trades"""
    if not trades:
        return {}
    
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    
    return {
        'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
        'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
        'largest_win': max([t['pnl'] for t in trades]),
        'largest_loss': min([t['pnl'] for t in trades]),
        'win_loss_ratio': (np.mean([t['pnl'] for t in winning_trades]) / 
                          abs(np.mean([t['pnl'] for t in losing_trades]))) 
                         if winning_trades and losing_trades else 0,
        'consecutive_wins': calculate_max_consecutive(trades, True),
        'consecutive_losses': calculate_max_consecutive(trades, False)
    }

def calculate_max_consecutive(trades, wins=True):
    """Calculate maximum consecutive wins or losses"""
    if not trades:
        return 0
    
    max_consecutive = 0
    current_consecutive = 0
    
    for trade in trades:
        if (wins and trade['pnl'] > 0) or (not wins and trade['pnl'] < 0):
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

def create_sweep_config():
    """Create sweep configuration for different optimization strategies"""
    
    # Basic parameter optimization
    basic_sweep = {
        'method': 'bayes',
        'metric': {
            'name': 'total_pnl',
            'goal': 'maximize'
        },
        'parameters': {
            'stop_loss_pct': {
                'min': 0.005,
                'max': 0.03
            },
            'take_profit_pct': {
                'min': 0.01,
                'max': 0.08
            },
            'position_size_pct': {
                'min': 0.05,
                'max': 0.25
            },
            'leverage': {
                'values': [5, 10, 15, 20]
            },
            'chunking_method': {
                'values': ['rows', 'time']
            },
            'chunk_size': {
                'values': [50, 75, 100, 150, 200]
            },
            'time_window_idx': {
                'values': [0, 1, 2, 3, 4, 5]  # Maps to time windows
            }
        }
    }
    
    # Risk-focused optimization
    risk_focused_sweep = {
        'method': 'bayes',
        'metric': {
            'name': 'sharpe_ratio',
            'goal': 'maximize'
        },
        'parameters': {
            'stop_loss_pct': {
                'min': 0.01,
                'max': 0.025
            },
            'take_profit_pct': {
                'min': 0.02,
                'max': 0.05
            },
            'position_size_pct': {
                'min': 0.05,
                'max': 0.15
            },
            'leverage': {
                'values': [5, 10, 15]
            },
            'chunking_method': {
                'values': ['rows', 'time']
            },
            'chunk_size': {
                'values': [75, 100, 150]
            },
            'time_window_idx': {
                'values': [2, 3, 4]  # 4H, 6H, 12H
            }
        }
    }
    
    # Consistency-focused optimization
    consistency_sweep = {
        'method': 'bayes',
        'metric': {
            'name': 'profitability_rate',
            'goal': 'maximize'
        },
        'parameters': {
            'stop_loss_pct': {
                'min': 0.01,
                'max': 0.02
            },
            'take_profit_pct': {
                'min': 0.02,
                'max': 0.04
            },
            'position_size_pct': {
                'min': 0.08,
                'max': 0.12
            },
            'leverage': {
                'values': [8, 10, 12]
            },
            'chunking_method': {
                'value': 'time'
            },
            'time_window_idx': {
                'values': [3, 4, 5]  # 6H, 12H, 1D
            }
        }
    }
    
    return {
        'basic': basic_sweep,
        'risk_focused': risk_focused_sweep,
        'consistency': consistency_sweep
    }

def run_sweep(sweep_type='basic', count=50):
    """
    Run hyperparameter sweep
    
    Args:
        sweep_type: Type of sweep ('basic', 'risk_focused', 'consistency')
        count: Number of runs to execute
    """
    sweep_configs = create_sweep_config()
    
    if sweep_type not in sweep_configs:
        print(f"Unknown sweep type: {sweep_type}")
        print(f"Available types: {list(sweep_configs.keys())}")
        return
    
    sweep_config = sweep_configs[sweep_type]
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=f"trading-bot-{sweep_type}-optimization"
    )
    
    print(f"🚀 Starting {sweep_type} hyperparameter sweep...")
    print(f"📊 Sweep ID: {sweep_id}")
    print(f"🔄 Running {count} experiments...")
    
    # Run sweep
    wandb.agent(sweep_id, objective_function, count=count)
    
    print(f"✅ Sweep completed! Check your wandb dashboard for results.")
    print(f"🔗 https://wandb.ai/your-username/trading-bot-{sweep_type}-optimization")

def main():
    """Main function to run hyperparameter optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('--sweep-type', choices=['basic', 'risk_focused', 'consistency'], 
                       default='basic', help='Type of optimization sweep')
    parser.add_argument('--count', type=int, default=50, 
                       help='Number of experiments to run')
    
    args = parser.parse_args()
    
    print("🎯 Trading Bot Hyperparameter Optimization")
    print("=" * 50)
    
    # Check if wandb is configured
    try:
        import wandb
        # Test wandb connection
        wandb.init(project="test-connection", mode="disabled")
        wandb.finish()
        print("✅ wandb is configured and ready")
    except Exception as e:
        print(f"❌ wandb configuration error: {e}")
        print("Please run 'python setup_wandb.py' first")
        return
    
    # Run sweep
    run_sweep(args.sweep_type, args.count)

if __name__ == "__main__":
    main()