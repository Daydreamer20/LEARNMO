# Data Chunking for Backtesting Guide

## Overview
This toolkit provides comprehensive data chunking capabilities for backtesting trading strategies across different market segments and conditions.

## Files Created
- `data_chunker.py` - Core chunking utility with multiple strategies
- `chunked_backtester.py` - Backtesting engine for chunked data
- `run_chunked_backtest.py` - Quick script to test all chunking methods
- `chunk_analyzer.py` - Detailed analysis of chunk results
- `chunking_summary.py` - Strategy comparison and recommendations

## Chunking Strategies

### 1. Fixed Row Chunks
```python
chunks = list(chunker.chunk_by_rows(chunk_size=100, overlap=0))
```
- **Best for**: Consistent sample sizes, statistical analysis
- **Pros**: Predictable chunk sizes, easy to compare
- **Cons**: May split important market events

### 2. Time-Based Chunks
```python
chunks = list(chunker.chunk_by_time(time_window='1D', overlap_hours=0))
```
- **Best for**: Daily/weekly pattern analysis
- **Pros**: Respects market sessions, natural boundaries
- **Cons**: Variable chunk sizes

### 3. Signal-Based Chunks
```python
chunks = list(chunker.chunk_by_signals(['Bullish BOS', 'Bearish BOS']))
```
- **Best for**: Event-driven analysis, signal validation
- **Pros**: Focuses on important market events
- **Cons**: Highly variable chunk sizes

### 4. Volatility-Based Chunks
```python
chunks = list(chunker.chunk_by_volatility(volatility_window=20))
```
- **Best for**: Regime-based analysis
- **Pros**: Adapts to market conditions
- **Cons**: Complex to interpret

## Quick Start

### 1. Run All Chunking Methods
```bash
python run_chunked_backtest.py
```

### 2. Analyze Results
```bash
python chunk_analyzer.py
```

### 3. Get Recommendations
```bash
python chunking_summary.py
```

## Results from Your Data

Based on your SIRENUSDT data analysis:

### 🏆 Best Strategy: Daily Time Windows
- **100% profitability rate** (2/2 chunks profitable)
- **6.66% average return** per chunk
- **54.5% win rate** across all trades
- **$133.23 total PnL**

### Key Insights
1. **Time-based chunking works best** - Your market shows strong daily patterns
2. **Larger chunks perform better** - 100-candle chunks outperform 50-candle chunks
3. **Signal-based chunking failed** - BOS signals may not be reliable for this timeframe
4. **Consistency matters** - Fixed 100-candle chunks had lowest volatility (0.70%)

## Customization

### Modify Backtesting Logic
Edit `chunked_backtester.py` in the `run_chunk_backtest()` method:
```python
# Your custom entry/exit logic here
if your_entry_condition:
    # Enter position
if your_exit_condition:
    # Exit position
```

### Add New Chunking Methods
Extend `DataChunker` class in `data_chunker.py`:
```python
def chunk_by_custom_method(self, **kwargs):
    # Your custom chunking logic
    yield chunk
```

### Adjust Configuration
Modify config in `run_chunked_backtest.py`:
```python
config = {
    'initial_balance': 1000,
    'leverage': 10,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    'position_size_pct': 0.1
}
```

## Advanced Usage

### Test Different Parameters
```python
from chunked_backtester import ChunkedBacktester

backtester = ChunkedBacktester('your_data.csv')

# Test different chunk sizes
for size in [50, 100, 200]:
    results = backtester.run_chunked_backtest('rows', chunk_size=size)
    analysis = backtester.analyze_results()
    print(f"Size {size}: {analysis['avg_chunk_return']:.2f}% avg return")
```

### Custom Analysis
```python
from chunk_analyzer import ChunkAnalyzer

analyzer = ChunkAnalyzer('results_daily_time_windows.json')
best_chunks = analyzer.find_best_chunks(5)
worst_chunks = analyzer.find_worst_chunks(5)
```

## Output Files Generated
- `results_*.json` - Detailed backtest results
- `performance_*.png` - Performance charts
- `*_analysis.txt` - Detailed analysis reports
- `*_timeline.png` - Timeline performance plots
- `chunking_analysis_summary.txt` - Overall summary and recommendations

## Recommendations for Your Strategy

1. **Use Daily Time Windows** for primary backtesting
2. **Consider 100-candle fixed chunks** for consistency testing
3. **Avoid signal-based chunking** until signals are improved
4. **Test with different market conditions** to validate robustness
5. **Optimize parameters** within the best-performing chunking method

## Next Steps
1. Apply the recommended chunking strategy to your main backtesting
2. Test with different symbols and timeframes
3. Optimize your trading signals based on chunk analysis insights
4. Consider combining multiple chunking approaches for robust validation