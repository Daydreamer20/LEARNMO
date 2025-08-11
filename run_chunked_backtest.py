#!/usr/bin/env python3
"""
Quick script to run chunked backtests with different strategies
"""

from chunked_backtester import ChunkedBacktester
import sys

def main():
    # Configuration
    data_file = 'BYBIT_SIRENUSDT.P, 5_e106e.csv'
    
    config = {
        'initial_balance': 1000,
        'leverage': 10,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'position_size_pct': 0.1
    }
    
    print("=== Chunked Backtesting Analysis ===")
    print(f"Data file: {data_file}")
    print(f"Config: {config}")
    
    # Initialize backtester
    backtester = ChunkedBacktester(data_file, config)
    
    # Test different chunking approaches
    chunking_strategies = [
        {
            'name': 'Fixed 50 candles',
            'method': 'rows',
            'params': {'chunk_size': 50}
        },
        {
            'name': 'Fixed 100 candles', 
            'method': 'rows',
            'params': {'chunk_size': 100}
        },
        {
            'name': 'Signal-based chunks',
            'method': 'signals',
            'params': {}
        },
        {
            'name': 'Daily time windows',
            'method': 'time',
            'params': {'time_window': '1D'}
        }
    ]
    
    results_summary = []
    
    for strategy in chunking_strategies:
        print(f"\n--- {strategy['name']} ---")
        
        try:
            # Run backtest
            results = backtester.run_chunked_backtest(
                strategy['method'], 
                **strategy['params']
            )
            
            # Analyze results
            analysis = backtester.analyze_results()
            
            # Print summary
            print(f"Chunks: {analysis['total_chunks']}")
            print(f"Profitable chunks: {analysis['profitable_chunks']} ({analysis['profitable_chunks']/analysis['total_chunks']*100:.1f}%)")
            print(f"Total trades: {analysis['total_trades']}")
            print(f"Win rate: {analysis['overall_win_rate']:.1%}")
            print(f"Total PnL: ${analysis['total_pnl']:.2f}")
            print(f"Avg chunk return: {analysis['avg_chunk_return']:.2f}%")
            print(f"Best chunk: {analysis['best_chunk_return']:.2f}%")
            print(f"Worst chunk: {analysis['worst_chunk_return']:.2f}%")
            print(f"Return volatility: {analysis['return_std']:.2f}%")
            
            # Save detailed results
            filename = f"results_{strategy['name'].lower().replace(' ', '_')}.json"
            backtester.save_results(filename)
            
            # Save performance chart
            chart_name = f"performance_{strategy['name'].lower().replace(' ', '_')}.png"
            backtester.plot_chunk_performance(chart_name)
            
            results_summary.append({
                'strategy': strategy['name'],
                'analysis': analysis
            })
            
        except Exception as e:
            print(f"Error with {strategy['name']}: {e}")
    
    # Compare strategies
    print("\n=== Strategy Comparison ===")
    print(f"{'Strategy':<20} {'Chunks':<8} {'Trades':<8} {'Win Rate':<10} {'Avg Return':<12} {'Volatility':<12}")
    print("-" * 80)
    
    for result in results_summary:
        name = result['strategy']
        analysis = result['analysis']
        print(f"{name:<20} {analysis['total_chunks']:<8} {analysis['total_trades']:<8} "
              f"{analysis['overall_win_rate']:<10.1%} {analysis['avg_chunk_return']:<12.2f}% "
              f"{analysis['return_std']:<12.2f}%")
    
    print("\nChunked backtest analysis complete!")
    print("Check the generated JSON files and PNG charts for detailed results.")

if __name__ == "__main__":
    main()