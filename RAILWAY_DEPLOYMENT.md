# Railway.com Deployment Guide - Continuous Learning Trading Bot

## 🚀 Features

This bot runs 24/7 on Railway.com with:
- **Continuous Learning**: Automatically optimizes parameters every hour
- **WANDB Integration**: Tracks all experiments and performance metrics
- **Data Persistence**: SQLite database stores all trades and market data
- **Auto-Recovery**: Reconnects automatically if connection drops
- **Health Monitoring**: Built-in health check endpoint for Railway
- **State Persistence**: Saves and restores bot state across restarts

## 📋 Prerequisites

1. **Railway.com Account**: Sign up at [railway.app](https://railway.app)
2. **WANDB Account** (optional): Sign up at [wandb.ai](https://wandb.ai) for experiment tracking
3. **GitHub Repository**: Push your code to GitHub

## 🛠️ Deployment Steps

### 1. Prepare Your Repository

```bash
# Clone or create your repository
git init
git add .
git commit -m "Initial commit - Railway trading bot"
git branch -M main
git remote add origin https://github.com/yourusername/trading-bot.git
git push -u origin main
```

### 2. Deploy to Railway

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will automatically detect the Python app and deploy

### 3. Configure Environment Variables

In Railway dashboard, go to your project → Variables tab and add:

```
WANDB_API_KEY=your_wandb_api_key_here
INITIAL_BALANCE=1000
PAPER_TRADING=true
OPTIMIZATION_INTERVAL=3600
```

### 4. Monitor Deployment

- Railway will build and deploy automatically
- Check the "Deployments" tab for build logs
- The bot will start trading immediately after deployment

## 📊 Monitoring & Tracking

### Railway Dashboard
- **Logs**: Real-time bot activity and trading decisions
- **Metrics**: CPU, memory, and network usage
- **Health**: Built-in health check at `/health` endpoint

### WANDB Dashboard
- **Real-time Metrics**: Price, volume, balance, RSI
- **Trade Analytics**: PnL, win rate, trade duration
- **Parameter Evolution**: How the bot learns and adapts
- **Performance Comparison**: Compare different optimization runs

## 🔧 Configuration Options

### Trading Parameters (Auto-optimized)
```python
leverage: 15.0              # Leverage multiplier
position_size_pct: 0.15     # 15% of balance per trade
stop_loss_pct: 0.020        # 2% stop loss
take_profit_pct: 0.035      # 3.5% take profit
momentum_threshold: 0.002   # Minimum momentum for signals
volume_multiplier: 1.4      # Volume spike threshold
```

### Learning Configuration
```python
optimization_interval: 3600  # Optimize every hour (seconds)
max_positions: 1            # Maximum concurrent positions
min_balance: 100            # Emergency stop threshold
```

## 📈 How the Learning Works

### 1. **Data Collection**
- Collects real-time 3-minute SOONUSDT candles
- Stores all market data in SQLite database
- Tracks every trade with full context

### 2. **Performance Evaluation**
- Calculates performance score every hour
- Considers: PnL, win rate, trade duration
- Compares against historical performance

### 3. **Parameter Optimization**
- **Poor Performance** (score < 0):
  - Reduces leverage by 10%
  - Reduces position size by 10%
  - Increases stop loss by 10%
  
- **Good Performance** (score > 50):
  - Increases leverage by 5%
  - Increases position size by 5%
  - Maintains current stop loss

### 4. **Continuous Adaptation**
- Parameters evolve based on market conditions
- Learns from both winning and losing trades
- Maintains performance history for analysis

## 🛡️ Safety Features

### Risk Management
- **Paper Trading Mode**: No real money at risk
- **Position Limits**: Maximum 1 position at a time
- **Emergency Stop**: Stops trading if balance drops too low
- **Auto-Recovery**: Reconnects if connection drops

### Data Persistence
- **State Saving**: Bot state saved every 20 candles
- **Trade History**: All trades stored in database
- **Parameter History**: Optimization results tracked
- **Crash Recovery**: Restores state after restarts

## 📊 Expected Performance

Based on backtesting results:
- **Target Return**: 10-20% monthly (paper trading)
- **Win Rate**: 35-45%
- **Trade Frequency**: 1-3 trades per day
- **Learning Curve**: Performance improves over 1-2 weeks

## 🔍 Monitoring Commands

### Check Bot Status
```bash
# Railway CLI
railway logs

# Or check health endpoint
curl https://your-app.railway.app/health
```

### WANDB Monitoring
1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to "railway-trading-bot" project
3. View real-time metrics and trade analytics

## 🚨 Troubleshooting

### Common Issues

1. **Connection Failures**
   - Bot automatically retries with exponential backoff
   - Check Railway logs for connection errors
   - Verify Bybit API is accessible

2. **WANDB Not Logging**
   - Check WANDB_API_KEY environment variable
   - Bot continues without WANDB if key is missing

3. **Database Errors**
   - SQLite database is created automatically
   - Check Railway storage limits

4. **Memory Issues**
   - Bot limits history to last 200 candles
   - Automatically cleans up old data

### Support
- **Railway Issues**: Check [Railway docs](https://docs.railway.app)
- **Bot Issues**: Check logs in Railway dashboard
- **WANDB Issues**: Check [WANDB docs](https://docs.wandb.ai)

## 💡 Advanced Features

### Custom Optimization
Modify `_run_optimization_sync()` to implement:
- **Bayesian Optimization**: More sophisticated parameter tuning
- **Multi-objective Optimization**: Balance return vs. risk
- **Market Regime Detection**: Adapt to different market conditions

### Additional Symbols
Add more trading pairs by modifying:
```python
self.symbols = ["SOONUSDT", "BTCUSDT", "ETHUSDT"]
```

### Real Trading
⚠️ **WARNING**: Only enable real trading after extensive testing
```python
# Set environment variables
PAPER_TRADING=false
BYBIT_API_KEY=your_real_api_key
BYBIT_API_SECRET=your_real_api_secret
```

## 🎯 Next Steps

1. **Deploy and Monitor**: Let it run for 24-48 hours
2. **Analyze Performance**: Check WANDB dashboard
3. **Optimize Further**: Adjust learning parameters if needed
4. **Scale Up**: Add more symbols or increase position sizes
5. **Go Live**: Switch to real trading (with caution!)

The bot is designed to learn and improve continuously. Give it time to adapt to current market conditions!