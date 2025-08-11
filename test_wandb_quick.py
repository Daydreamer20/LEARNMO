#!/usr/bin/env python3
"""
Quick test of wandb integration with SOONUSDT data
"""

import os
os.environ['WANDB_API_KEY'] = 'acff3ce9376b1acc92a538394739adc9a19b1c99'

from wandb_integration import WandbTradingTracker, WandbBacktester

def main():
    print('🧪 Running Quick wandb Test with SOONUSDT Data')
    
    # Initialize tracker
    tracker = WandbTradingTracker(
        project_name='soonusdt-test-experiment',
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
    
    # Test with the best performing strategy (6H time windows)
    strategy = {
        'name': 'Time_6H_Test',
        'method': 'time',
        'params': {'time_window': '6H'}
    }
    
    # Run experiment
    results = backtester.run_wandb_experiment(
        experiment_name='SOONUSDT_Quick_Test',
        chunking_strategies=[strategy],
        tags=['test', 'soonusdt', '6h-time'],
        notes='Quick test of wandb integration with SOONUSDT data'
    )
    
    # Print results
    analysis = results[strategy['name']]['analysis']
    print(f'✅ Test completed!')
    print(f'💰 Total PnL: ${analysis["total_pnl"]:.2f}')
    print(f'📊 Win Rate: {analysis["overall_win_rate"]:.1%}')
    print(f'📈 Avg Return: {analysis["avg_chunk_return"]:.2f}%')
    print(f'🎯 Total Trades: {analysis["total_trades"]}')
    print(f'📦 Chunks: {analysis["total_chunks"]}')
    
    print(f'\n🔗 Check your dashboard at:')
    print(f'   https://wandb.ai/elytuyor-department-of-education/soonusdt-test-experiment')
    
    # Finish experiment
    tracker.finish_experiment()
    
    return True

if __name__ == "__main__":
    main()