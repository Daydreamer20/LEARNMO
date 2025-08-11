# SOONUSDT Chunked Backtesting Results

## Dataset Overview
- **Symbol**: BYBIT_SOONUSDT.P (3-minute timeframe)
- **Total Rows**: 1,029 candles
- **Time Range**: August 8, 2025 22:09 to August 11, 2025 01:33 (2.5 days)
- **Signal Counts**:
  - Bullish BOS: 185 signals
  - Bearish BOS: 138 signals
  - Bullish CHOCH: 0 signals
  - Bearish CHOCH: 2 signals

## Key Findings

### 🏆 Best Performing Strategy: Time-Based Chunking (6H/12H)
- **Total PnL**: +$48.04 (4.8% return)
- **Win Rate**: 37.3%
- **Profitable Chunks**: 1/3 (33.3%)
- **Best Chunk**: +14.16% return
- **Risk**: High volatility (9.54%)

### 📊 Signal Performance Analysis

#### Bullish BOS (Best Signal)
- **197 trades** across all strategies
- **39.6% win rate** - significantly above random
- **+$1.35 average PnL** per trade
- **Total contribution**: +$265.56

#### Bearish BOS (Problematic Signal)
- **113 trades** across all strategies
- **23.0% win rate** - below breakeven
- **-$7.73 average PnL** per trade
- **Total contribution**: -$873.69

#### Bearish CHOCH (Avoid)
- **5 trades** total
- **0% win rate** - completely unreliable
- **-$4.61 average loss** per trade

### 🎯 Strategy Comparison

| Strategy | Total PnL | Win Rate | Profitable Chunks | Volatility | Recommendation |
|----------|-----------|----------|-------------------|------------|----------------|
| **6H/12H Time Windows** | **+$48.04** | **37.3%** | **33.3%** | 9.54% | ✅ **BEST** |
| Fixed 150 Candles | -$16.84 | 35.1% | 42.9% | 8.48% | ⚠️ Marginal |
| Fixed 75 Candles | -$66.24 | 33.9% | 35.7% | 5.63% | ❌ Poor |
| Volatility-Based | -$123.96 | 33.9% | 19.4% | 6.29% | ❌ Poor |
| Signal-Based (BOS) | -$260.12 | 7.1% | 6.2% | 1.61% | ❌ **WORST** |

## Critical Insights

### ✅ What Works
1. **Time-based chunking** significantly outperforms other methods
2. **Bullish BOS signals** are reliable with 39.6% win rate
3. **Larger chunk sizes** (150 vs 75 candles) perform better
4. **6-12 hour time windows** capture natural market cycles

### ❌ What Doesn't Work
1. **Signal-based chunking** performs terribly (-$260 loss)
2. **Bearish signals** are unreliable (23% win rate for BOS, 0% for CHOCH)
3. **Small chunk sizes** increase noise and reduce performance
4. **CHOCH signals** should be avoided entirely

### 🔍 Market Behavior Analysis
- **Bullish bias**: Market shows strong upward momentum during this period
- **Signal quality**: BOS signals are much more reliable than CHOCH
- **Timeframe sensitivity**: 3-minute data benefits from longer aggregation periods
- **Volatility impact**: Higher volatility strategies can be profitable but risky

## Recommendations

### 🚀 For Live Trading
1. **Use 6-12 hour time windows** for backtesting and strategy validation
2. **Focus on Bullish BOS signals** - avoid bearish signals in trending markets
3. **Implement position sizing** - reduce size on bearish signals if used
4. **Consider market regime** - this analysis shows bullish market conditions

### 📈 For Strategy Optimization
1. **Test different stop-loss/take-profit ratios** for bullish signals
2. **Add trend filters** to improve bearish signal reliability
3. **Implement dynamic position sizing** based on signal strength
4. **Consider longer timeframes** (5m, 15m) for more stable signals

### ⚠️ Risk Management
1. **High volatility warning**: Time-based strategies show 9.54% volatility
2. **Drawdown potential**: Worst chunk showed -8.96% loss
3. **Signal reliability**: Only 37.3% win rate even for best strategy
4. **Market dependency**: Results may not generalize to different market conditions

## Technical Configuration Used
```python
config = {
    'initial_balance': 1000,
    'leverage': 10,
    'stop_loss_pct': 0.015,  # 1.5%
    'take_profit_pct': 0.03,  # 3% (2:1 R:R)
    'position_size_pct': 0.1  # 10% of balance
}
```

## Files Generated
- `soonusdt_*.json` - Detailed backtest results for each strategy
- `soonusdt_*.png` - Performance charts for each strategy
- `soonusdt_comprehensive_analysis.png` - Overall comparison chart
- `soonusdt_comprehensive_report.txt` - Detailed text report

## Next Steps
1. **Validate on different time periods** to confirm strategy robustness
2. **Test on other crypto pairs** to check generalizability
3. **Implement live paper trading** with the best strategy (6H time windows)
4. **Optimize parameters** (stop loss, take profit, position size) for the winning strategy
5. **Add additional filters** to improve bearish signal reliability

---

**Note**: This analysis is based on a 2.5-day period during bullish market conditions. Results may vary significantly in different market regimes. Always validate strategies across multiple time periods and market conditions before live implementation.