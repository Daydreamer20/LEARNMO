# Railway Trading Bot Deployment

## 🚀 Deployed: 2025-08-11 18:47:29

This is a continuous learning trading bot that:
- Trades SOONUSDT on Bybit
- Optimizes parameters every hour using WANDB
- Runs 24/7 on Railway.com
- Stores all data in SQLite database

## Environment Variables Needed:
- WANDB_API_KEY (optional but recommended)
- INITIAL_BALANCE=1000
- PAPER_TRADING=true

## Health Check:
- Endpoint: /health
- Port: Automatically set by Railway

## Monitoring:
- Railway logs for bot activity
- WANDB dashboard for performance metrics
