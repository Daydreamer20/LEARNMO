#!/usr/bin/env python3
"""
Summary and recommendations for chunking strategies
"""

import json
import glob
import pandas as pd
from typing import Dict, List

def load_all_results() -> Dict:
    """Load all chunked backtest results"""
    results_files = glob.glob("results_*.json")
    all_results = {}
    
    for file in results_files:
        strategy_name = file.replace('results_', '').replace('.json', '').replace('_', ' ').title()
        
        with open(file, 'r') as f:
            data = json.load(f)
            all_results[strategy_name] = data['analysis']
    
    return all_results

def compare_strategies(results: Dict) -> pd.DataFrame:
    """Create comparison table of all strategies"""
    comparison_data = []
    
    for strategy, analysis in results.items():
        comparison_data.append({
            'Strategy': strategy,
            'Total Chunks': analysis['total_chunks'],
            'Profitable Chunks': analysis['profitable_chunks'],
            'Profitability Rate': f"{analysis['profitable_chunks']/analysis['total_chunks']*100:.1f}%",
            'Total Trades': analysis['total_trades'],
            'Win Rate': f"{analysis['overall_win_rate']:.1%}",
            'Total PnL': f"${analysis['total_pnl']:.2f}",
            'Avg Chunk Return': f"{analysis['avg_chunk_return']:.2f}%",
            'Best Chunk': f"{analysis['best_chunk_return']:.2f}%",
            'Worst Chunk': f"{analysis['worst_chunk_return']:.2f}%",
            'Return Volatility': f"{analysis['return_std']:.2f}%"
        })
    
    return pd.DataFrame(comparison_data)

def rank_strategies(results: Dict) -> List[Dict]:
    """Rank strategies by multiple criteria"""
    rankings = []
    
    for strategy, analysis in results.items():
        # Calculate composite score
        profitability_score = analysis['profitable_chunks'] / analysis['total_chunks']
        win_rate_score = analysis['overall_win_rate']
        return_score = max(0, analysis['avg_chunk_return'] / 10)  # Normalize to 0-1 scale
        consistency_score = max(0, 1 - (analysis['return_std'] / 10))  # Lower volatility = higher score
        
        composite_score = (
            profitability_score * 0.3 +
            win_rate_score * 0.3 +
            return_score * 0.25 +
            consistency_score * 0.15
        )
        
        rankings.append({
            'strategy': strategy,
            'composite_score': composite_score,
            'profitability_score': profitability_score,
            'win_rate_score': win_rate_score,
            'return_score': return_score,
            'consistency_score': consistency_score,
            'analysis': analysis
        })
    
    return sorted(rankings, key=lambda x: x['composite_score'], reverse=True)

def generate_recommendations(rankings: List[Dict]) -> str:
    """Generate strategy recommendations"""
    recommendations = []
    recommendations.append("=== CHUNKING STRATEGY RECOMMENDATIONS ===\n")
    
    if not rankings:
        return "No results available for analysis."
    
    best_strategy = rankings[0]
    recommendations.append(f"🏆 BEST OVERALL STRATEGY: {best_strategy['strategy']}")
    recommendations.append(f"   Composite Score: {best_strategy['composite_score']:.3f}")
    recommendations.append(f"   Profitability Rate: {best_strategy['analysis']['profitable_chunks']/best_strategy['analysis']['total_chunks']*100:.1f}%")
    recommendations.append(f"   Average Return: {best_strategy['analysis']['avg_chunk_return']:.2f}%")
    recommendations.append(f"   Win Rate: {best_strategy['analysis']['overall_win_rate']:.1%}")
    recommendations.append("")
    
    # Specific use case recommendations
    recommendations.append("📊 STRATEGY-SPECIFIC RECOMMENDATIONS:")
    recommendations.append("")
    
    # Find best for different criteria
    best_profitability = max(rankings, key=lambda x: x['profitability_score'])
    best_returns = max(rankings, key=lambda x: x['analysis']['avg_chunk_return'])
    best_consistency = max(rankings, key=lambda x: x['consistency_score'])
    most_trades = max(rankings, key=lambda x: x['analysis']['total_trades'])
    
    recommendations.append(f"💰 For Maximum Profitability: {best_profitability['strategy']}")
    recommendations.append(f"   {best_profitability['analysis']['profitable_chunks']}/{best_profitability['analysis']['total_chunks']} chunks profitable ({best_profitability['profitability_score']*100:.1f}%)")
    recommendations.append("")
    
    recommendations.append(f"📈 For Highest Returns: {best_returns['strategy']}")
    recommendations.append(f"   Average chunk return: {best_returns['analysis']['avg_chunk_return']:.2f}%")
    recommendations.append("")
    
    recommendations.append(f"🎯 For Most Consistency: {best_consistency['strategy']}")
    recommendations.append(f"   Return volatility: {best_consistency['analysis']['return_std']:.2f}%")
    recommendations.append("")
    
    recommendations.append(f"⚡ For Most Trading Activity: {most_trades['strategy']}")
    recommendations.append(f"   Total trades: {most_trades['analysis']['total_trades']}")
    recommendations.append("")
    
    # General insights
    recommendations.append("💡 KEY INSIGHTS:")
    
    # Analyze chunk size impact
    fixed_strategies = [r for r in rankings if 'Fixed' in r['strategy']]
    if len(fixed_strategies) >= 2:
        smaller_chunks = min(fixed_strategies, key=lambda x: int(x['strategy'].split()[1]))
        larger_chunks = max(fixed_strategies, key=lambda x: int(x['strategy'].split()[1]))
        
        if smaller_chunks['analysis']['avg_chunk_return'] > larger_chunks['analysis']['avg_chunk_return']:
            recommendations.append("   • Smaller chunks tend to perform better")
        else:
            recommendations.append("   • Larger chunks tend to perform better")
    
    # Signal-based analysis
    signal_strategy = next((r for r in rankings if 'Signal' in r['strategy']), None)
    if signal_strategy:
        if signal_strategy['analysis']['avg_chunk_return'] < 0:
            recommendations.append("   • Signal-based chunking shows poor performance - signals may not be reliable")
        else:
            recommendations.append("   • Signal-based chunking shows good performance - signals are reliable")
    
    # Time-based analysis
    time_strategy = next((r for r in rankings if 'Daily' in r['strategy'] or 'Time' in r['strategy']), None)
    if time_strategy:
        if time_strategy['composite_score'] > 0.5:
            recommendations.append("   • Time-based chunking works well - market has daily patterns")
        else:
            recommendations.append("   • Time-based chunking is mediocre - limited daily patterns")
    
    recommendations.append("")
    recommendations.append("🚀 NEXT STEPS:")
    recommendations.append(f"   1. Use {best_strategy['strategy']} for your main backtesting")
    recommendations.append("   2. Consider optimizing parameters for the best-performing strategy")
    recommendations.append("   3. Test with different market conditions or time periods")
    recommendations.append("   4. Combine insights from multiple chunking approaches")
    
    return "\n".join(recommendations)

def main():
    """Generate comprehensive chunking analysis summary"""
    print("Loading chunked backtest results...")
    
    # Load all results
    results = load_all_results()
    
    if not results:
        print("No results files found. Run chunked backtest first.")
        return
    
    print(f"Found results for {len(results)} strategies\n")
    
    # Create comparison table
    comparison_df = compare_strategies(results)
    print("=== STRATEGY COMPARISON TABLE ===")
    print(comparison_df.to_string(index=False))
    print()
    
    # Rank strategies
    rankings = rank_strategies(results)
    
    # Generate recommendations
    recommendations = generate_recommendations(rankings)
    print(recommendations)
    
    # Save summary to file
    with open('chunking_analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("CHUNKED BACKTESTING ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write("COMPARISON TABLE:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        f.write(recommendations)
    
    print("\n📄 Full analysis saved to 'chunking_analysis_summary.txt'")

if __name__ == "__main__":
    main()