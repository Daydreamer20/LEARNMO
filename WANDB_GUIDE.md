# Weights & Biases Integration for Trading Bot Training

## Overview
This integration provides comprehensive experiment tracking, hyperparameter optimization, and model training capabilities for your trading bot using Weights & Biases (wandb).

## 🚀 Quick Start

### 1. Setup wandb
```bash
# Install and configure wandb
python setup_wandb.py

# Or manually:
pip install wandb
wandb login
```

### 2. Run Basic Example
```bash
python wandb_example.py
```

### 3. Start Hyperparameter Optimization
```bash
python run_hyperparameter_sweep.py --sweep-type basic --count 50
```

## 📁 Files Overview

### Core Integration Files
- **`wandb_integration.py`** - Main wandb integration classes
- **`setup_wandb.py`** - Setup and configuration script
- **`wandb_example.py`** - Complete usage examples
- **`run_hyperparameter_sweep.py`** - Automated parameter optimization

### Key Classes
- **`WandbTradingTracker`** - Core wandb tracking functionality
- **`WandbBacktester`** - Enhanced backtester with wandb integration

## 🎯 Features

### 1. Experiment Tracking
- **Strategy Comparison** - Compare different chunking strategies
- **Performance Metrics** - Track PnL, win rates, Sharpe ratios
- **Visual Charts** - Automatic chart generation and logging
- **Trade Analysis** - Detailed trade-by-trade tracking

### 2. Hyperparameter Optimization
- **Bayesian Optimization** - Intelligent parameter search
- **Multiple Objectives** - Optimize for PnL, Sharpe ratio, or consistency
- **Parameter Ranges** - Configurable search spaces
- **Early Stopping** - Automatic termination of poor runs

### 3. Signal Analysis
- **Signal Performance** - Track individual signal effectiveness
- **Entry/Exit Analysis** - Analyze entry and exit signal quality
- **Market Regime Detection** - Identify optimal market conditions

## 📊 Tracked Metrics

### Performance Metrics
- `total_pnl` - Total profit/loss
- `win_rate` - Percentage of winning trades
- `avg_chunk_return` - Average return per chunk
- `return_std` - Return volatility
- `sharpe_ratio` - Risk-adjusted returns
- `profit_factor` - Gross profit / gross loss
- `max_drawdown` - Maximum drawdown

### Risk Metrics
- `calmar_ratio` - Return / max drawdown
- `risk_adjusted_return` - PnL / volatility
- `return_range` - Best chunk - worst chunk
- `consecutive_wins/losses` - Maximum consecutive streaks

### Trade Metrics
- `avg_win` - Average winning trade
- `avg_loss` - Average losing trade
- `largest_win/loss` - Best and worst individual trades
- `win_loss_ratio` - Average win / average loss

## 🔧 Configuration Examples

### Basic Configuration
```python
config = {
    'initial_balance': 1000,
    'leverage': 10,
    'stop_loss_pct': 0.015,
    'take_profit_pct': 0.03,
    'position_size_pct': 0.1
}
```

### Conservative Configuration
```python
config = {
    'initial_balance': 1000,
    'leverage': 5,
    'stop_loss_pct': 0.01,
    'take_profit_pct': 0.02,
    'position_size_pct': 0.08
}
```

### Aggressive Configuration
```python
config = {
    'initial_balance': 1000,
    'leverage': 20,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    'position_size_pct': 0.15
}
```

## 🎛️ Hyperparameter Sweep Types

### 1. Basic Optimization
- **Objective**: Maximize total PnL
- **Parameters**: All trading parameters + chunking methods
- **Use Case**: General strategy optimization

```bash
python run_hyperparameter_sweep.py --sweep-type basic --count 50
```

### 2. Risk-Focused Optimization
- **Objective**: Maximize Sharpe ratio
- **Parameters**: Conservative parameter ranges
- **Use Case**: Risk-adjusted strategy development

```bash
python run_hyperparameter_sweep.py --sweep-type risk_focused --count 30
```

### 3. Consistency Optimization
- **Objective**: Maximize profitable chunk rate
- **Parameters**: Stability-focused ranges
- **Use Case**: Consistent performance strategies

```bash
python run_hyperparameter_sweep.py --sweep-type consistency --count 40
```

## 📈 Usage Examples

### Simple Experiment
```python
from wandb_integration import WandbTradingTracker, WandbBacktester

# Initialize tracker
tracker = WandbTradingTracker(project_name="my-trading-bot")

# Initialize backtester
backtester = WandbBacktester(
    'your_data.csv',
    config=your_config,
    wandb_tracker=tracker
)

# Define strategies
strategies = [
    {'name': 'Strategy_A', 'method': 'rows', 'params': {'chunk_size': 100}},
    {'name': 'Strategy_B', 'method': 'time', 'params': {'time_window': '6H'}}
]

# Run experiment
results = backtester.run_wandb_experiment(
    experiment_name="Strategy_Comparison",
    chunking_strategies=strategies,
    tags=["comparison", "test"],
    notes="Testing different strategies"
)
```

### Manual Logging
```python
import wandb

# Initialize run
wandb.init(project="trading-bot", name="manual-experiment")

# Log metrics
wandb.log({
    'total_pnl': 150.25,
    'win_rate': 0.65,
    'sharpe_ratio': 1.8
})

# Log charts
wandb.log({"performance_chart": wandb.Image("chart.png")})

# Finish run
wandb.finish()
```

## 🔍 Analysis and Insights

### Dashboard Features
- **Run Comparison** - Compare multiple experiments side-by-side
- **Parameter Importance** - Identify most impactful parameters
- **Performance Trends** - Track improvement over time
- **Custom Charts** - Create custom visualizations

### Key Insights to Track
1. **Parameter Sensitivity** - Which parameters affect performance most?
2. **Market Regime Performance** - How does strategy perform in different conditions?
3. **Signal Quality** - Which signals are most reliable?
4. **Risk-Return Trade-offs** - Optimal risk/reward configurations
5. **Consistency Patterns** - What makes strategies more consistent?

## 🚨 Best Practices

### Experiment Organization
- Use **descriptive names** for experiments
- Add **relevant tags** for easy filtering
- Include **detailed notes** about hypotheses
- **Group related experiments** in same project

### Parameter Optimization
- Start with **broad ranges** then narrow down
- Use **Bayesian optimization** for efficiency
- Set **realistic objectives** (don't just maximize PnL)
- **Validate results** on different time periods

### Data Management
- **Version your data** when making changes
- **Document data preprocessing** steps
- **Track data quality** metrics
- **Separate train/validation** periods

## 🔧 Advanced Features

### Custom Metrics
```python
# Add custom metrics to tracking
def custom_metric(results):
    # Your custom calculation
    return metric_value

wandb.log({"custom_metric": custom_metric(results)})
```

### Artifact Tracking
```python
# Track model artifacts
artifact = wandb.Artifact('trading-model', type='model')
artifact.add_file('model.pkl')
wandb.log_artifact(artifact)
```

### Sweep Configuration
```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'total_pnl', 'goal': 'maximize'},
    'parameters': {
        'stop_loss_pct': {'min': 0.01, 'max': 0.03},
        'take_profit_pct': {'min': 0.02, 'max': 0.06}
    }
}
```

## 🐛 Troubleshooting

### Common Issues
1. **Login Problems** - Run `wandb login` manually
2. **API Key Issues** - Set `WANDB_API_KEY` environment variable
3. **Network Issues** - Use `wandb offline` for local logging
4. **Large Logs** - Reduce logging frequency or use sampling

### Performance Tips
- **Batch logging** - Log multiple metrics at once
- **Reduce chart frequency** - Don't log charts every iteration
- **Use sampling** - Log subset of trades for large datasets
- **Optimize data types** - Use appropriate numeric types

## 📚 Resources

### Documentation
- [wandb Documentation](https://docs.wandb.ai/)
- [Hyperparameter Sweeps](https://docs.wandb.ai/guides/sweeps)
- [Experiment Tracking](https://docs.wandb.ai/guides/track)

### Examples
- Run `python wandb_example.py` for interactive examples
- Check wandb dashboard for experiment results
- Review sweep results for optimization insights

## 🎯 Next Steps

1. **Run Initial Experiments** - Start with basic strategy comparison
2. **Optimize Parameters** - Use hyperparameter sweeps
3. **Analyze Results** - Identify best performing configurations
4. **Validate Strategies** - Test on different time periods
5. **Deploy Best Strategy** - Implement optimized parameters

---

**Note**: Always validate your optimized parameters on out-of-sample data before live trading. Hyperparameter optimization can lead to overfitting if not properly validated.