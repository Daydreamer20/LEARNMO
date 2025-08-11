#!/usr/bin/env python3
"""
Run comprehensive wandb experiment with multiple strategies
"""

import os
os.environ['WANDB_API_KEY'] = 'acff3ce9376b1acc92a538394739adc9a19b1c99'

from wandb_integration import WandbTradingTracker, WandbBacktester

def main():
    print('🚀 Running Comprehensive wandb Experiment')
    print('=' * 50)
    
    # Initialize tracker
    tracker = WandbTradingTracker(
        project_name='soonusdt-comprehensive-analysis',
        entity='elytuyor-department-of-education'
    )
    
    # Configuration
    config = {
        'initial_balance': 1000,
        'leverage': 10,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.03,
        'position_size_pct': 0.1
    }
    
    # Initialize backtester
    backtester = WandbBacktester(
        'BYBIT_SOONUSDT.P, 3_08827.csv',
        config=config,
        wandb_tracker=tracker
    )
    
    # Define comprehensive strategy comparison
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
            'name': 'Time_6H_Windows',
            'method': 'time',
            'params': {'time_window': '6H'}
        },
        {
            'name': 'Time_12H_Windows',
            'method': 'time',
            'params': {'time_window': '12H'}
        },
        {
            'name': 'BOS_Signals_Only',
            'method': 'signals',
            'params': {'signal_columns': ['Bullish BOS', 'Bearish BOS']}
        },
        {
            'name': 'Volatility_Based',
            'method': 'volatility',
            'params': {}
        }
    ]
    
    print(f'🧪 Testing {len(strategies)} different strategies...')
    
    # Run comprehensive experiment
    results = backtester.run_wandb_experiment(
        experiment_name='SOONUSDT_Comprehensive_Strategy_Analysis',
        chunking_strategies=strategies,
        tags=['comprehensive', 'soonusdt', 'strategy-comparison', 'production'],
        notes='Comprehensive analysis of all chunking strategies for SOONUSDT trading with detailed wandb tracking'
    )
    
    # Print comprehensive results
    print('\n📊 COMPREHENSIVE RESULTS SUMMARY:')
    print('=' * 80)
    print(f"{'Strategy':<25} {'PnL':<12} {'Win Rate':<10} {'Trades':<8} {'Chunks':<8} {'Sharpe':<8}")
    print('-' * 80)
    
    best_strategy = None
    best_pnl = float('-inf')
    
    for strategy_name, data in results.items():
        analysis = data['analysis']
        sharpe = analysis['avg_chunk_return'] / max(analysis['return_std'], 0.01)
        
        print(f"{strategy_name:<25} ${analysis['total_pnl']:<11.2f} "
              f"{analysis['overall_win_rate']:<9.1%} {analysis['total_trades']:<8} "
              f"{analysis['total_chunks']:<8} {sharpe:<8.2f}")
        
        if analysis['total_pnl'] > best_pnl:
            best_pnl = analysis['total_pnl']
            best_strategy = strategy_name
    
    print('-' * 80)
    print(f'🏆 BEST STRATEGY: {best_strategy} with ${best_pnl:.2f} profit')
    
    # Signal analysis summary
    print('\n🎯 SIGNAL PERFORMANCE SUMMARY:')
    all_trades = []
    for data in results.values():
        all_trades.extend(data['trades'])
    
    if all_trades:
        signal_stats = {}
        for trade in all_trades:
            signal = trade.get('entry_signal', 'Unknown')
            if signal not in signal_stats:
                signal_stats[signal] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
            
            signal_stats[signal]['trades'] += 1
            signal_stats[signal]['total_pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                signal_stats[signal]['wins'] += 1
        
        print(f"{'Signal':<15} {'Trades':<8} {'Win Rate':<10} {'Avg PnL':<10} {'Total PnL':<12}")
        print('-' * 65)
        for signal, stats in sorted(signal_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            avg_pnl = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            print(f"{signal:<15} {stats['trades']:<8} {win_rate:<9.1%} "
                  f"${avg_pnl:<9.2f} ${stats['total_pnl']:<11.2f}")
    
    print(f'\n🔗 View complete results at:')
    print(f'   https://wandb.ai/elytuyor-department-of-education/soonusdt-comprehensive-analysis')
    
    # Finish experiment
    tracker.finish_experiment()
    
    print('\n✅ Comprehensive experiment completed!')
    print('📊 All data has been logged to wandb for detailed analysis')
    
    return results

if __name__ == "__main__":
    main()