#!/usr/bin/env python3
"""
Comprehensive analysis summary for SOONUSDT chunked backtesting results
"""

import json
import glob
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

def load_soonusdt_results() -> Dict:
    """Load all SOONUSDT chunked backtest results"""
    results_files = glob.glob("soonusdt_*.json")
    all_results = {}
    
    for file in results_files:
        strategy_name = file.replace('soonusdt_', '').replace('.json', '').replace('_', ' ').title()
        
        with open(file, 'r') as f:
            data = json.load(f)
            all_results[strategy_name] = data
    
    return all_results

def analyze_signal_performance(results: Dict) -> Dict:
    """Analyze performance by signal type across all strategies"""
    signal_stats = {}
    
    for strategy_name, data in results.items():
        for chunk_result in data['results']:
            for trade in chunk_result.get('trade_details', []):
                signal = trade.get('entry_signal', 'Unknown')
                
                if signal not in signal_stats:
                    signal_stats[signal] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_pnl': 0,
                        'strategies': set()
                    }
                
                signal_stats[signal]['total_trades'] += 1
                signal_stats[signal]['total_pnl'] += trade['pnl']
                signal_stats[signal]['strategies'].add(strategy_name)
                
                if trade['pnl'] > 0:
                    signal_stats[signal]['winning_trades'] += 1
    
    # Calculate derived metrics
    for signal, stats in signal_stats.items():
        stats['win_rate'] = stats['winning_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        stats['avg_pnl'] = stats['total_pnl'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        stats['strategies'] = list(stats['strategies'])
    
    return signal_stats

def create_performance_comparison(results: Dict) -> pd.DataFrame:
    """Create detailed performance comparison table"""
    comparison_data = []
    
    for strategy, data in results.items():
        analysis = data['analysis']
        
        # Calculate additional metrics
        profitable_rate = analysis['profitable_chunks'] / analysis['total_chunks'] * 100
        trades_per_chunk = analysis['total_trades'] / analysis['total_chunks']
        
        comparison_data.append({
            'Strategy': strategy,
            'Chunks': analysis['total_chunks'],
            'Profitable Rate': f"{profitable_rate:.1f}%",
            'Total Trades': analysis['total_trades'],
            'Trades/Chunk': f"{trades_per_chunk:.1f}",
            'Win Rate': f"{analysis['overall_win_rate']:.1%}",
            'Total PnL': f"${analysis['total_pnl']:.2f}",
            'Avg Return': f"{analysis['avg_chunk_return']:.2f}%",
            'Best Chunk': f"{analysis['best_chunk_return']:.2f}%",
            'Worst Chunk': f"{analysis['worst_chunk_return']:.2f}%",
            'Volatility': f"{analysis['return_std']:.2f}%",
            'Risk-Adj Return': f"{analysis['avg_chunk_return'] / max(analysis['return_std'], 0.1):.2f}"
        })
    
    return pd.DataFrame(comparison_data)

def generate_insights(results: Dict, signal_stats: Dict) -> List[str]:
    """Generate key insights from the analysis"""
    insights = []
    
    # Overall performance insights
    profitable_strategies = [name for name, data in results.items() if data['analysis']['total_pnl'] > 0]
    if profitable_strategies:
        insights.append(f"✅ {len(profitable_strategies)} out of {len(results)} strategies were profitable")
        best_strategy = max(results.items(), key=lambda x: x[1]['analysis']['total_pnl'])
        insights.append(f"🏆 Best strategy: {best_strategy[0]} with ${best_strategy[1]['analysis']['total_pnl']:.2f} profit")
    else:
        insights.append("❌ No strategies were profitable overall")
    
    # Signal performance insights
    if signal_stats:
        best_signal = max(signal_stats.items(), key=lambda x: x[1]['win_rate'])
        worst_signal = min(signal_stats.items(), key=lambda x: x[1]['win_rate'])
        
        insights.append(f"📈 Best signal: {best_signal[0]} with {best_signal[1]['win_rate']:.1%} win rate")
        insights.append(f"📉 Worst signal: {worst_signal[0]} with {worst_signal[1]['win_rate']:.1%} win rate")
        
        # Bullish vs Bearish performance
        bullish_signals = {k: v for k, v in signal_stats.items() if 'Bullish' in k}
        bearish_signals = {k: v for k, v in signal_stats.items() if 'Bearish' in k}
        
        if bullish_signals and bearish_signals:
            bullish_avg_wr = np.mean([s['win_rate'] for s in bullish_signals.values()])
            bearish_avg_wr = np.mean([s['win_rate'] for s in bearish_signals.values()])
            
            if bullish_avg_wr > bearish_avg_wr:
                insights.append(f"🐂 Bullish signals outperform bearish ({bullish_avg_wr:.1%} vs {bearish_avg_wr:.1%})")
            else:
                insights.append(f"🐻 Bearish signals outperform bullish ({bearish_avg_wr:.1%} vs {bullish_avg_wr:.1%})")
    
    # Chunking method insights
    time_based = [name for name in results.keys() if 'hour' in name.lower() or 'time' in name.lower()]
    signal_based = [name for name in results.keys() if 'bos' in name.lower() or 'choch' in name.lower()]
    fixed_size = [name for name in results.keys() if 'fixed' in name.lower()]
    
    if time_based:
        time_avg_pnl = np.mean([results[name]['analysis']['total_pnl'] for name in time_based])
        insights.append(f"⏰ Time-based chunking avg PnL: ${time_avg_pnl:.2f}")
    
    if signal_based:
        signal_avg_pnl = np.mean([results[name]['analysis']['total_pnl'] for name in signal_based])
        insights.append(f"🎯 Signal-based chunking avg PnL: ${signal_avg_pnl:.2f}")
    
    if fixed_size:
        fixed_avg_pnl = np.mean([results[name]['analysis']['total_pnl'] for name in fixed_size])
        insights.append(f"📏 Fixed-size chunking avg PnL: ${fixed_avg_pnl:.2f}")
    
    # Risk insights
    high_vol_strategies = [name for name, data in results.items() if data['analysis']['return_std'] > 8]
    if high_vol_strategies:
        insights.append(f"⚠️  High volatility strategies: {', '.join(high_vol_strategies)}")
    
    return insights

def create_visualization(results: Dict):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total PnL comparison
    strategies = list(results.keys())
    pnls = [results[s]['analysis']['total_pnl'] for s in strategies]
    
    colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
    axes[0, 0].bar(range(len(strategies)), pnls, color=colors, alpha=0.7)
    axes[0, 0].set_title('Total PnL by Strategy')
    axes[0, 0].set_ylabel('PnL ($)')
    axes[0, 0].set_xticks(range(len(strategies)))
    axes[0, 0].set_xticklabels(strategies, rotation=45, ha='right')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Win Rate vs Return Volatility
    win_rates = [results[s]['analysis']['overall_win_rate'] * 100 for s in strategies]
    volatilities = [results[s]['analysis']['return_std'] for s in strategies]
    
    scatter = axes[0, 1].scatter(volatilities, win_rates, c=pnls, cmap='RdYlGn', s=100, alpha=0.7)
    axes[0, 1].set_title('Win Rate vs Volatility')
    axes[0, 1].set_xlabel('Return Volatility (%)')
    axes[0, 1].set_ylabel('Win Rate (%)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Total PnL ($)')
    
    # Add strategy labels
    for i, strategy in enumerate(strategies):
        axes[0, 1].annotate(strategy[:10], (volatilities[i], win_rates[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Chunk profitability rates
    profitable_rates = [results[s]['analysis']['profitable_chunks'] / results[s]['analysis']['total_chunks'] * 100 
                       for s in strategies]
    
    axes[1, 0].bar(range(len(strategies)), profitable_rates, alpha=0.7, color='blue')
    axes[1, 0].set_title('Profitable Chunk Rate by Strategy')
    axes[1, 0].set_ylabel('Profitable Chunks (%)')
    axes[1, 0].set_xticks(range(len(strategies)))
    axes[1, 0].set_xticklabels(strategies, rotation=45, ha='right')
    
    # 4. Risk-adjusted returns
    risk_adj_returns = [results[s]['analysis']['avg_chunk_return'] / max(results[s]['analysis']['return_std'], 0.1) 
                       for s in strategies]
    
    axes[1, 1].bar(range(len(strategies)), risk_adj_returns, alpha=0.7, color='purple')
    axes[1, 1].set_title('Risk-Adjusted Returns (Return/Volatility)')
    axes[1, 1].set_ylabel('Risk-Adjusted Return')
    axes[1, 1].set_xticks(range(len(strategies)))
    axes[1, 1].set_xticklabels(strategies, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('soonusdt_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive analysis chart saved to 'soonusdt_comprehensive_analysis.png'")

def main():
    """Generate comprehensive SOONUSDT analysis"""
    print("=== SOONUSDT Comprehensive Analysis ===\n")
    
    # Load results
    results = load_soonusdt_results()
    
    if not results:
        print("No SOONUSDT results found. Run the backtesting first.")
        return
    
    print(f"Analyzing {len(results)} chunking strategies...\n")
    
    # Create performance comparison
    comparison_df = create_performance_comparison(results)
    print("PERFORMANCE COMPARISON:")
    print(comparison_df.to_string(index=False))
    print()
    
    # Analyze signal performance
    signal_stats = analyze_signal_performance(results)
    if signal_stats:
        print("SIGNAL PERFORMANCE ANALYSIS:")
        for signal, stats in sorted(signal_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True):
            print(f"  {signal}:")
            print(f"    Trades: {stats['total_trades']}")
            print(f"    Win Rate: {stats['win_rate']:.1%}")
            print(f"    Avg PnL: ${stats['avg_pnl']:.2f}")
            print(f"    Total PnL: ${stats['total_pnl']:.2f}")
            print(f"    Used in: {', '.join(stats['strategies'])}")
            print()
    
    # Generate insights
    insights = generate_insights(results, signal_stats)
    print("KEY INSIGHTS:")
    for insight in insights:
        print(f"  {insight}")
    print()
    
    # Create visualization
    create_visualization(results)
    
    # Save detailed report
    with open('soonusdt_comprehensive_report.txt', 'w', encoding='utf-8') as f:
        f.write("SOONUSDT COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        if signal_stats:
            f.write("SIGNAL PERFORMANCE:\n")
            for signal, stats in signal_stats.items():
                f.write(f"{signal}: {stats['total_trades']} trades, {stats['win_rate']:.1%} win rate, ${stats['avg_pnl']:.2f} avg PnL\n")
            f.write("\n")
        
        f.write("KEY INSIGHTS:\n")
        for insight in insights:
            f.write(f"{insight}\n")
    
    print("📄 Detailed report saved to 'soonusdt_comprehensive_report.txt'")
    print("📊 Analysis complete!")

if __name__ == "__main__":
    main()