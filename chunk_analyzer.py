#!/usr/bin/env python3
"""
Analyze chunked backtest results to identify patterns and optimal conditions
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List

class ChunkAnalyzer:
    """
    Analyze chunked backtest results to find patterns
    """
    
    def __init__(self, results_file: str):
        """
        Load results from JSON file
        
        Args:
            results_file: Path to results JSON file
        """
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.results = self.data['results']
        self.analysis = self.data['analysis']
        
    def find_best_chunks(self, top_n: int = 3) -> List[Dict]:
        """
        Find the best performing chunks
        
        Args:
            top_n: Number of top chunks to return
            
        Returns:
            List of best chunk results
        """
        # Sort chunks by return percentage
        sorted_chunks = sorted(
            [r for r in self.results if r['trades'] > 0],
            key=lambda x: x['return_pct'],
            reverse=True
        )
        
        return sorted_chunks[:top_n]
        
    def find_worst_chunks(self, bottom_n: int = 3) -> List[Dict]:
        """
        Find the worst performing chunks
        
        Args:
            bottom_n: Number of bottom chunks to return
            
        Returns:
            List of worst chunk results
        """
        # Sort chunks by return percentage (ascending)
        sorted_chunks = sorted(
            [r for r in self.results if r['trades'] > 0],
            key=lambda x: x['return_pct']
        )
        
        return sorted_chunks[:bottom_n]
        
    def analyze_chunk_characteristics(self) -> Dict:
        """
        Analyze characteristics of profitable vs unprofitable chunks
        
        Returns:
            Dictionary with analysis results
        """
        profitable_chunks = [r for r in self.results if r['return_pct'] > 0]
        unprofitable_chunks = [r for r in self.results if r['return_pct'] < 0]
        
        if not profitable_chunks or not unprofitable_chunks:
            return {"error": "Need both profitable and unprofitable chunks for comparison"}
        
        analysis = {
            'profitable': {
                'count': len(profitable_chunks),
                'avg_return': np.mean([c['return_pct'] for c in profitable_chunks]),
                'avg_trades': np.mean([c['trades'] for c in profitable_chunks]),
                'avg_win_rate': np.mean([c['win_rate'] for c in profitable_chunks]),
                'avg_chunk_size': np.mean([c['rows'] for c in profitable_chunks])
            },
            'unprofitable': {
                'count': len(unprofitable_chunks),
                'avg_return': np.mean([c['return_pct'] for c in unprofitable_chunks]),
                'avg_trades': np.mean([c['trades'] for c in unprofitable_chunks]),
                'avg_win_rate': np.mean([c['win_rate'] for c in unprofitable_chunks]),
                'avg_chunk_size': np.mean([c['rows'] for c in unprofitable_chunks])
            }
        }
        
        return analysis
        
    def plot_chunk_timeline(self, save_path: str = None):
        """
        Plot chunk performance over time
        
        Args:
            save_path: Optional path to save plot
        """
        # Extract data for plotting
        chunk_data = []
        for chunk in self.results:
            if chunk['trades'] > 0:
                chunk_data.append({
                    'start_time': pd.to_datetime(chunk['start_time']),
                    'return_pct': chunk['return_pct'],
                    'trades': chunk['trades'],
                    'win_rate': chunk['win_rate']
                })
        
        if not chunk_data:
            print("No trading chunks to plot")
            return
            
        df = pd.DataFrame(chunk_data)
        df = df.sort_values('start_time')
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Returns over time
        axes[0].plot(df['start_time'], df['return_pct'], marker='o')
        axes[0].set_title('Chunk Returns Over Time')
        axes[0].set_ylabel('Return %')
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].grid(True, alpha=0.3)
        
        # Number of trades over time
        axes[1].bar(df['start_time'], df['trades'], alpha=0.7)
        axes[1].set_title('Trades per Chunk Over Time')
        axes[1].set_ylabel('Number of Trades')
        axes[1].grid(True, alpha=0.3)
        
        # Win rate over time
        axes[2].plot(df['start_time'], df['win_rate'] * 100, marker='s', color='green')
        axes[2].set_title('Win Rate Over Time')
        axes[2].set_ylabel('Win Rate %')
        axes[2].set_xlabel('Time')
        axes[2].axhline(y=50, color='r', linestyle='--', alpha=0.5)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timeline plot saved to {save_path}")
        else:
            plt.show()
            
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=== CHUNK ANALYSIS REPORT ===")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data file: {self.data.get('data_file', 'Unknown')}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  Total chunks: {self.analysis['total_chunks']}")
        report.append(f"  Profitable chunks: {self.analysis['profitable_chunks']} ({self.analysis['profitable_chunks']/self.analysis['total_chunks']*100:.1f}%)")
        report.append(f"  Total trades: {self.analysis['total_trades']}")
        report.append(f"  Overall win rate: {self.analysis['overall_win_rate']:.1%}")
        report.append(f"  Total PnL: ${self.analysis['total_pnl']:.2f}")
        report.append(f"  Average chunk return: {self.analysis['avg_chunk_return']:.2f}%")
        report.append("")
        
        # Best performing chunks
        best_chunks = self.find_best_chunks(3)
        report.append("TOP PERFORMING CHUNKS:")
        for i, chunk in enumerate(best_chunks, 1):
            report.append(f"  #{i}: {chunk['return_pct']:.2f}% return, {chunk['trades']} trades, {chunk['win_rate']:.1%} win rate")
        report.append("")
        
        # Worst performing chunks
        worst_chunks = self.find_worst_chunks(3)
        report.append("WORST PERFORMING CHUNKS:")
        for i, chunk in enumerate(worst_chunks, 1):
            report.append(f"  #{i}: {chunk['return_pct']:.2f}% return, {chunk['trades']} trades, {chunk['win_rate']:.1%} win rate")
        report.append("")
        
        # Characteristics analysis
        char_analysis = self.analyze_chunk_characteristics()
        if 'error' not in char_analysis:
            report.append("PROFITABLE vs UNPROFITABLE CHUNKS:")
            report.append("  Profitable chunks:")
            report.append(f"    Count: {char_analysis['profitable']['count']}")
            report.append(f"    Avg return: {char_analysis['profitable']['avg_return']:.2f}%")
            report.append(f"    Avg trades: {char_analysis['profitable']['avg_trades']:.1f}")
            report.append(f"    Avg win rate: {char_analysis['profitable']['avg_win_rate']:.1%}")
            report.append(f"    Avg chunk size: {char_analysis['profitable']['avg_chunk_size']:.0f} rows")
            report.append("")
            report.append("  Unprofitable chunks:")
            report.append(f"    Count: {char_analysis['unprofitable']['count']}")
            report.append(f"    Avg return: {char_analysis['unprofitable']['avg_return']:.2f}%")
            report.append(f"    Avg trades: {char_analysis['unprofitable']['avg_trades']:.1f}")
            report.append(f"    Avg win rate: {char_analysis['unprofitable']['avg_win_rate']:.1%}")
            report.append(f"    Avg chunk size: {char_analysis['unprofitable']['avg_chunk_size']:.0f} rows")
        
        return "\n".join(report)
        
    def save_report(self, filename: str):
        """
        Save analysis report to file
        
        Args:
            filename: Output filename
        """
        report = self.generate_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Analysis report saved to {filename}")

def main():
    """
    Analyze all available chunk results
    """
    import glob
    
    # Find all results files
    results_files = glob.glob("results_*.json")
    
    if not results_files:
        print("No results files found. Run chunked backtest first.")
        return
    
    print(f"Found {len(results_files)} results files to analyze:")
    for file in results_files:
        print(f"  - {file}")
    
    # Analyze each results file
    for results_file in results_files:
        print(f"\n=== Analyzing {results_file} ===")
        
        try:
            analyzer = ChunkAnalyzer(results_file)
            
            # Generate and print report
            report = analyzer.generate_report()
            print(report)
            
            # Save detailed report
            report_filename = results_file.replace('.json', '_analysis.txt')
            analyzer.save_report(report_filename)
            
            # Generate timeline plot
            timeline_filename = results_file.replace('.json', '_timeline.png')
            analyzer.plot_chunk_timeline(timeline_filename)
            
        except Exception as e:
            print(f"Error analyzing {results_file}: {e}")

if __name__ == "__main__":
    main()