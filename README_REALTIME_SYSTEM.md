# Real-Time Trading Bot with Continuous Optimization

A complete real-time trading system that connects to Bybit, continuously optimizes parameters using WANDB, and executes trades automatically.

## 🚀 Quick Start

### 1. Setup the System
```bash
python setup_realtime_system.py
```

### 2. Configure (Optional)
Edit `config.py` with your preferences:
- API keys (not needed for testnet)
- Trading parameters
- Risk management settings

### 3. Start Trading
```bash
python start_trading_system.py
```

## 📁 System Components

### Core Files
- **`bybit_realtime_data.py`** - Real-time data collection from Bybit WebSocket
- **`realtime_training_manager.py`** - Continuous parameter optimization
- **`live_trading_bot.py`** - Live trading execution engine
- **`setup_realtime_system.py`** - One-click system setup
- **`start_trading_system.py`** - System startup script

### Data Flow
```
Bybit WebSocket → Data Collector → Training Manager → Optimized Parameters → Trading Bot → Trades
                                ↓
                            WANDB Logging ← Performance Tracking
```

## 🔄 How It Works

### 1. Real-Time Data Collection
- Connects to Bybit WebSocket API
- Streams 3-minute klines, trades, and orderbook data
- Stores data in SQLite database for analysis
- Triggers callbacks for real-time processing

### 2. Continuous Parameter Optimization
- Runs optimization every 30 minutes (configurable)
- Tests 50+ parameter combinations using recent data
- Uses weighted scoring: PnL (40%) + Win Rate (20%) + Sharpe (20%) + Consistency (20%)
- Updates parameters only if significant improvement (>10%)

### 3. Live Trading Execution
- Monitors real-time data for trading signals
- Uses continuously optimized parameters
- Implements risk management (stop loss, take profit, position limits)
- Logs all trades to WANDB for analysis

### 4. Performance Tracking
- Real-time performance metrics in WANDB
- Trade-by-trade analysis
- Parameter optimization history
- Risk management alerts

## 📊 Key Features

### Adaptive Parameter Optimization
- **Stop Loss**: 0.5% - 3.0% (optimized)
- **Take Profit**: 1.0% - 8.0% (optimized)
- **Position Size**: 5% - 25% of balance (optimized)
- **Leverage**: 5x - 20x (optimized)
- **Chunking Method**: Time-based or row-based (optimized)

### Risk Management
- Maximum daily loss limit (-$50 default)
- Maximum concurrent positions (3 default)
- Position sizing based on account balance
- Automatic stop loss and take profit

### Signal Generation
- Technical indicators: SMA, RSI, MACD, Bollinger Bands
- Volume analysis and momentum detection
- Market structure signals (BOS, CHOCH)
- Customizable signal logic

## 🎯 Trading Signals

### Current Signal Logic
```python
# Bullish Signal
- Price above 10-period SMA
- 10-SMA above 20-SMA (trend confirmation)
- RSI between 40-70 (not overbought)
- Positive 3-period momentum (>0.2%)
- Volume spike (>1.5x average)

# Bearish Signal  
- Price below 10-period SMA
- 10-SMA below 20-SMA (downtrend)
- RSI between 30-60 (not oversold)
- Negative 3-period momentum (<-0.2%)
- Volume spike (>1.5x average)
```

## 📈 Performance Metrics

### Real-Time Tracking
- **Balance**: Current account balance
- **Daily PnL**: Profit/loss for current day
- **Total Return**: Overall percentage return
- **Win Rate**: Percentage of profitable trades
- **Active Positions**: Number of open positions
- **Parameter Updates**: When optimization improves performance

### WANDB Dashboard
- Live performance charts
- Parameter optimization history
- Trade analysis and patterns
- Risk metrics and drawdowns
- Signal effectiveness analysis

## ⚙️ Configuration Options

### Trading Parameters
```python
INITIAL_BALANCE = 1000.0      # Starting balance
MAX_DAILY_LOSS = -50.0        # Stop trading if daily loss exceeds
MAX_POSITIONS = 3             # Maximum concurrent positions
```

### Optimization Settings
```python
OPTIMIZATION_INTERVAL = 1800  # Optimize every 30 minutes
MIN_DATA_POINTS = 200         # Minimum data points for optimization
```

### Bybit Connection
```python
BYBIT_TESTNET = True          # Use testnet (recommended for testing)
SYMBOLS = ["SOONUSDT"]        # Symbols to trade
```

## 🔧 Advanced Usage

### Custom Signal Logic
Modify `_add_trading_signals()` in `realtime_training_manager.py`:

```python
def _add_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
    # Add your custom signal logic here
    df['My_Custom_Signal'] = your_signal_logic(df)
    return df
```

### Parameter Ranges
Adjust optimization ranges in `RealtimeTrainingManager`:

```python
self.param_ranges = {
    'stop_loss_pct': (0.01, 0.02),  # Tighter range
    'take_profit_pct': (0.02, 0.04),
    # ... other parameters
}
```

### Risk Management
Customize risk rules in `LiveTradingBot`:

```python
self.max_daily_loss = -100.0    # Higher loss tolerance
self.max_positions = 5          # More concurrent positions
```

## 📊 Monitoring & Analysis

### Real-Time Logs
```bash
tail -f trading_bot.log
```

### Database Queries
```sql
-- View recent trades
SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;

-- View parameter optimization history
SELECT * FROM parameter_optimization ORDER BY timestamp DESC LIMIT 5;

-- View performance over time
SELECT DATE(created_at) as date, COUNT(*) as trades, 
       AVG(pnl_amount) as avg_pnl
FROM trades GROUP BY DATE(created_at);
```

### WANDB Analysis
- Go to your WANDB dashboard
- View real-time performance metrics
- Analyze parameter optimization trends
- Compare different time periods

## 🚨 Safety Features

### Testnet Mode
- System starts in testnet mode by default
- No real money at risk during testing
- Full functionality for testing strategies

### Risk Limits
- Daily loss limits prevent catastrophic losses
- Position limits prevent overexposure
- Automatic stop losses on all trades

### Parameter Validation
- Only updates parameters if significantly better
- Validates parameter ranges before applying
- Logs all parameter changes for audit trail

## 🔍 Troubleshooting

### Common Issues

**WebSocket Connection Failed**
```bash
# Check internet connection
ping stream-testnet.bybit.com

# Verify WebSocket URL in logs
```

**No Trading Signals**
```bash
# Check if data is being received
sqlite3 realtime_trading.db "SELECT COUNT(*) FROM klines;"

# Verify signal logic in logs
```

**Parameter Optimization Not Running**
```bash
# Check if enough data points
# Verify optimization interval settings
# Look for errors in logs
```

### Debug Mode
Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Next Steps

### 1. Testing Phase
- Run system in testnet mode for several days
- Monitor performance and parameter changes
- Analyze signal effectiveness

### 2. Strategy Refinement
- Customize signal logic based on results
- Adjust parameter ranges for your risk tolerance
- Add additional technical indicators

### 3. Live Trading (Advanced)
- Set `BYBIT_TESTNET = False` in config
- Add real API credentials
- Start with small position sizes
- Monitor closely for first few days

## ⚠️ Important Disclaimers

- **This is experimental software** - test thoroughly before live trading
- **Past performance doesn't guarantee future results**
- **Always use proper risk management**
- **Start with small amounts** when going live
- **Monitor the system actively** especially initially

## 🤝 Support

For issues or questions:
1. Check the logs first: `trading_bot.log`
2. Review WANDB dashboard for performance issues
3. Check database for data collection problems
4. Verify network connectivity to Bybit

---

**Happy Trading! 🚀📈**

Remember: The best trading system is one that's thoroughly tested and properly managed. Start small, learn continuously, and always prioritize risk management over profits.