#!/usr/bin/env python3
"""
Specialized chunked backtesting for SOONUSDT data with BOS and CHOCH signals
"""

from chunked_backtester import ChunkedBacktester
import pandas as pd
import sys

def main():
    # Configuration for SOONUSDT
    data_file = 'BYBIT_SOONUSDT.P, 3_08827.csv'
    
    config = {
        'initial_balance': 1000,
        'leverage': 10,
        'stop_loss_pct': 0.015,  # Tighter stop loss for crypto
        'take_profit_pct': 0.03,  # 2:1 risk reward
        'position_size_pct': 0.1
    }
    
    print("=== SOONUSDT Chunked Backtesting Analysis ===")
    print(f"Data file: {data_file}")
    print(f"Config: {config}")
    
    # Load and inspect data first
    data = pd.read_csv(data_file)
    print(f"\nDataset Overview:")
    print(f"  Total rows: {len(data)}")
    print(f"  Time range: {pd.to_datetime(data['time'], unit='s').min()} to {pd.to_datetime(data['time'], unit='s').max()}")
    print(f"  Signal counts:")
    print(f"    Bullish BOS: {data['Bullish BOS'].sum()}")
    print(f"    Bearish BOS: {data['Bearish BOS'].sum()}")
    print(f"    Bullish CHOCH: {data['Bullish CHOCH'].sum()}")
    print(f"    Bearish CHOCH: {data['Bearish CHOCH'].sum()}")
    
    # Initialize backtester
    backtester = ChunkedBacktester(data_file, config)
    
    # Test different chunking approaches optimized for this dataset
    chunking_strategies = [
        {
            'name': 'Fixed 75 candles',
            'method': 'rows',
            'params': {'chunk_size': 75}
        },
        {
            'name': 'Fixed 150 candles', 
            'method': 'rows',
            'params': {'chunk_size': 150}
        },
        {
            'name': 'BOS + CHOCH signals',
            'method': 'signals',
            'params': {'signal_columns': ['Bullish BOS', 'Bearish BOS', 'Bullish CHOCH', 'Bearish CHOCH']}
        },
        {
            'name': 'BOS signals only',
            'method': 'signals',
            'params': {'signal_columns': ['Bullish BOS', 'Bearish BOS']}
        },
        {
            'name': '6-hour time windows',
            'method': 'time',
            'params': {'time_window': '6H'}
        },
        {
            'name': '12-hour time windows',
            'method': 'time',
            'params': {'time_window': '12H'}
        },
        {
            'name': 'Volatility-based',
            'method': 'volatility',
            'params': {}
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
            
            # Analyze signal performance if we have trade details
            if results and len(results) > 0:
                all_trades = []
                for chunk_result in results:
                    all_trades.extend(chunk_result.get('trade_details', []))
                
                if all_trades:
                    # Group by entry signal
                    signal_performance = {}
                    for trade in all_trades:
                        signal = trade.get('entry_signal', 'Unknown')
                        if signal not in signal_performance:
                            signal_performance[signal] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
                        
                        signal_performance[signal]['trades'] += 1
                        signal_performance[signal]['total_pnl'] += trade['pnl']
                        if trade['pnl'] > 0:
                            signal_performance[signal]['wins'] += 1
                    
                    print(f"Signal Performance:")
                    for signal, perf in signal_performance.items():
                        win_rate = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
                        avg_pnl = perf['total_pnl'] / perf['trades'] if perf['trades'] > 0 else 0
                        print(f"  {signal}: {perf['trades']} trades, {win_rate:.1%} win rate, ${avg_pnl:.2f} avg PnL")
            
            # Save detailed results
            filename = f"soonusdt_{strategy['name'].lower().replace(' ', '_').replace('-', '_')}.json"
            backtester.save_results(filename)
            
            # Save performance chart
            chart_name = f"soonusdt_{strategy['name'].lower().replace(' ', '_').replace('-', '_')}.png"
            backtester.plot_chunk_performance(chart_name)
            
            results_summary.append({
                'strategy': strategy['name'],
                'analysis': analysis
            })
            
        except Exception as e:
            print(f"Error with {strategy['name']}: {e}")
    
    # Compare strategies
    print("\n=== Strategy Comparison ===")
    print(f"{'Strategy':<25} {'Chunks':<8} {'Trades':<8} {'Win Rate':<10} {'Avg Return':<12} {'Total PnL':<12} {'Volatility':<12}")
    print("-" * 100)
    
    for result in results_summary:
        name = result['strategy']
        analysis = result['analysis']
        print(f"{name:<25} {analysis['total_chunks']:<8} {analysis['total_trades']:<8} "
              f"{analysis['overall_win_rate']:<10.1%} {analysis['avg_chunk_return']:<12.2f}% "
              f"${analysis['total_pnl']:<11.2f} {analysis['return_std']:<12.2f}%")
    
    # Find best strategy
    if results_summary:
        best_strategy = max(results_summary, key=lambda x: x['analysis']['total_pnl'])
        print(f"\n🏆 Best Strategy by Total PnL: {best_strategy['strategy']}")
        print(f"   Total PnL: ${best_strategy['analysis']['total_pnl']:.2f}")
        print(f"   Win Rate: {best_strategy['analysis']['overall_win_rate']:.1%}")
        print(f"   Avg Return: {best_strategy['analysis']['avg_chunk_return']:.2f}%")
        
        best_consistency = min(results_summary, key=lambda x: x['analysis']['return_std'])
        print(f"\n🎯 Most Consistent Strategy: {best_consistency['strategy']}")
        print(f"   Return Volatility: {best_consistency['analysis']['return_std']:.2f}%")
        print(f"   Total PnL: ${best_consistency['analysis']['total_pnl']:.2f}")
    
    print("\nSOONUSDT chunked backtest analysis complete!")
    print("Check the generated JSON files and PNG charts for detailed results.")

if __name__ == "__main__":
    main()