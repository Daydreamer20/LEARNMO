#!/usr/bin/env python3
"""
Complete example of using wandb for trading bot optimization
"""

from wandb_integration import WandbTradingTracker, WandbBacktester
import pandas as pd
import numpy as np

def run_basic_experiment():
    """Run a basic experiment with wandb tracking"""
    print("🧪 Running Basic Experiment with wandb")
    
    # Initialize wandb tracker
    tracker = WandbTradingTracker(
        project_name="soonusdt-basic-experiment",
        entity=None  # Use your wandb username
    )
    
    # Define trading configuration
    config = {
        'initial_balance': 1000,
        'leverage': 10,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.03,
        'position_size_pct': 0.1
    }
    
    # Initialize enhanced backtester
    backtester = WandbBacktester(
        'BYBIT_SOONUSDT.P, 3_08827.csv',
        config=config,
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
        experiment_name="SOONUSDT_Strategy_Comparison",
        chunking_strategies=strategies,
        tags=["basic", "comparison", "soonusdt"],
        notes="Basic comparison of chunking strategies with wandb tracking"
    )
    
    # Print summary
    print("\n📊 Experiment Results:")
    for strategy_name, data in results.items():
        analysis = data['analysis']
        print(f"  {strategy_name}:")
        print(f"    Total PnL: ${analysis['total_pnl']:.2f}")
        print(f"    Win Rate: {analysis['overall_win_rate']:.1%}")
        print(f"    Chunks: {analysis['total_chunks']}")
    
    # Finish experiment
    tracker.finish_experiment()
    print("✅ Basic experiment completed!")

def run_parameter_comparison():
    """Compare different parameter configurations"""
    print("🔬 Running Parameter Comparison Experiment")
    
    # Test different configurations
    configs_to_test = [
        {
            'name': 'Conservative',
            'config': {
                'initial_balance': 1000,
                'leverage': 5,
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.02,
                'position_size_pct': 0.08
            }
        },
        {
            'name': 'Moderate',
            'config': {
                'initial_balance': 1000,
                'leverage': 10,
                'stop_loss_pct': 0.015,
                'take_profit_pct': 0.03,
                'position_size_pct': 0.1
            }
        },
        {
            'name': 'Aggressive',
            'config': {
                'initial_balance': 1000,
                'leverage': 20,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'position_size_pct': 0.15
            }
        }
    ]
    
    results_summary = []
    
    for config_test in configs_to_test:
        print(f"\n🧪 Testing {config_test['name']} configuration...")
        
        # Initialize tracker for this configuration
        tracker = WandbTradingTracker(
            project_name="soonusdt-parameter-comparison",
            entity=None
        )
        
        # Initialize backtester
        backtester = WandbBacktester(
            'BYBIT_SOONUSDT.P, 3_08827.csv',
            config=config_test['config'],
            wandb_tracker=tracker
        )
        
        # Test with best performing strategy (6H time windows)
        strategy = {
            'name': f"Time_6H_{config_test['name']}",
            'method': 'time',
            'params': {'time_window': '6H'}
        }
        
        # Run experiment
        results = backtester.run_wandb_experiment(
            experiment_name=f"SOONUSDT_{config_test['name']}_Config",
            chunking_strategies=[strategy],
            tags=["parameter-comparison", config_test['name'].lower(), "6h-time"],
            notes=f"Testing {config_test['name']} risk parameters with 6H time windows"
        )
        
        # Store results
        analysis = results[strategy['name']]['analysis']
        results_summary.append({
            'config': config_test['name'],
            'total_pnl': analysis['total_pnl'],
            'win_rate': analysis['overall_win_rate'],
            'return_std': analysis['return_std'],
            'sharpe_ratio': analysis['avg_chunk_return'] / max(analysis['return_std'], 0.01)
        })
        
        tracker.finish_experiment()
    
    # Print comparison
    print("\n📊 Parameter Comparison Results:")
    print(f"{'Config':<12} {'PnL':<10} {'Win Rate':<10} {'Volatility':<12} {'Sharpe':<8}")
    print("-" * 60)
    for result in results_summary:
        print(f"{result['config']:<12} ${result['total_pnl']:<9.2f} "
              f"{result['win_rate']:<9.1%} {result['return_std']:<11.2f}% "
              f"{result['sharpe_ratio']:<8.2f}")
    
    print("✅ Parameter comparison completed!")

def run_signal_analysis():
    """Analyze signal performance in detail"""
    print("🎯 Running Detailed Signal Analysis")
    
    tracker = WandbTradingTracker(
        project_name="soonusdt-signal-analysis",
        entity=None
    )
    
    # Use moderate configuration
    config = {
        'initial_balance': 1000,
        'leverage': 10,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.03,
        'position_size_pct': 0.1
    }
    
    backtester = WandbBacktester(
        'BYBIT_SOONUSDT.P, 3_08827.csv',
        config=config,
        wandb_tracker=tracker
    )
    
    # Test different signal combinations
    signal_strategies = [
        {
            'name': 'Bullish_BOS_Only',
            'method': 'signals',
            'params': {'signal_columns': ['Bullish BOS']}
        },
        {
            'name': 'Bearish_BOS_Only',
            'method': 'signals',
            'params': {'signal_columns': ['Bearish BOS']}
        },
        {
            'name': 'All_BOS_Signals',
            'method': 'signals',
            'params': {'signal_columns': ['Bullish BOS', 'Bearish BOS']}
        },
        {
            'name': 'All_Signals',
            'method': 'signals',
            'params': {'signal_columns': ['Bullish BOS', 'Bearish BOS', 'Bullish CHOCH', 'Bearish CHOCH']}
        }
    ]
    
    # Run signal analysis experiment
    results = backtester.run_wandb_experiment(
        experiment_name="SOONUSDT_Signal_Analysis",
        chunking_strategies=signal_strategies,
        tags=["signal-analysis", "bos", "choch"],
        notes="Detailed analysis of different signal combinations"
    )
    
    # Print signal analysis
    print("\n🎯 Signal Analysis Results:")
    for strategy_name, data in results.items():
        analysis = data['analysis']
        print(f"\n  {strategy_name}:")
        print(f"    Total PnL: ${analysis['total_pnl']:.2f}")
        print(f"    Win Rate: {analysis['overall_win_rate']:.1%}")
        print(f"    Total Trades: {analysis['total_trades']}")
        print(f"    Chunks: {analysis['total_chunks']}")
        
        # Analyze individual trades
        trades = data['trades']
        if trades:
            bullish_trades = [t for t in trades if 'Bullish' in t.get('entry_signal', '')]
            bearish_trades = [t for t in trades if 'Bearish' in t.get('entry_signal', '')]
            
            if bullish_trades:
                bullish_wr = len([t for t in bullish_trades if t['pnl'] > 0]) / len(bullish_trades)
                print(f"    Bullish signals: {len(bullish_trades)} trades, {bullish_wr:.1%} win rate")
            
            if bearish_trades:
                bearish_wr = len([t for t in bearish_trades if t['pnl'] > 0]) / len(bearish_trades)
                print(f"    Bearish signals: {len(bearish_trades)} trades, {bearish_wr:.1%} win rate")
    
    tracker.finish_experiment()
    print("✅ Signal analysis completed!")

def main():
    """Main function to run all examples"""
    print("🚀 wandb Trading Bot Integration Examples")
    print("=" * 60)
    
    # Check if wandb is available
    try:
        import wandb
        print("✅ wandb is available")
    except ImportError:
        print("❌ wandb not installed. Run 'python setup_wandb.py' first")
        return
    
    # Check if data file exists
    import os
    if not os.path.exists('BYBIT_SOONUSDT.P, 3_08827.csv'):
        print("❌ Data file not found. Please ensure 'BYBIT_SOONUSDT.P, 3_08827.csv' exists")
        return
    
    print("📊 Available examples:")
    print("1. Basic experiment with strategy comparison")
    print("2. Parameter configuration comparison")
    print("3. Detailed signal analysis")
    print("4. Run all examples")
    
    choice = input("\nSelect example to run (1-4): ").strip()
    
    if choice == '1':
        run_basic_experiment()
    elif choice == '2':
        run_parameter_comparison()
    elif choice == '3':
        run_signal_analysis()
    elif choice == '4':
        print("🔄 Running all examples...")
        run_basic_experiment()
        print("\n" + "="*60)
        run_parameter_comparison()
        print("\n" + "="*60)
        run_signal_analysis()
    else:
        print("❌ Invalid choice")
        return
    
    print("\n🎉 All examples completed!")
    print("📊 Check your wandb dashboard at https://wandb.ai for detailed results")
    print("💡 Tip: Use 'python run_hyperparameter_sweep.py' for automated optimization")

if __name__ == "__main__":
    main()