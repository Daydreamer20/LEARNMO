# Ollama-Powered Scalping Trading Bot

A sophisticated trading bot that uses Ollama AI for market analysis and executes scalping trades on Bybit perpetual futures.

## Features

- **AI-Powered Analysis**: Uses Ollama for intelligent market analysis and trade signals
- **Risk Management**: Advanced risk management with liquidation-safe stop losses
- **Scalping Strategy**: Optimized for quick in/out trades with small profits
- **Multi-Symbol Support**: Trades multiple perpetual contracts simultaneously
- **Real-time Monitoring**: Continuous market monitoring and position management

## Safety Features

- Maximum $2 USD per trade
- Stop loss at 0.1% before liquidation price
- Leverage optimization based on risk parameters
- Cooldown periods to prevent overtrading
- Comprehensive validation and error handling

## Requirements

- Python 3.8+
- Ollama installed and running locally
- Bybit account with API access
- Sufficient USDT balance for trading

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd scalping-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama:
```bash
# Install Ollama (visit https://ollama.ai for instructions)
# Pull a model (e.g., llama3.1:8b)
ollama pull llama3.1:8b
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your Bybit API credentials
```

## Configuration

Edit `.env` file with your settings:

```env
# Bybit API Configuration
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_TESTNET=False  # Live trading enabled

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Trading Configuration
MAX_TRADE_SIZE_USD=2.0
STOP_LOSS_PERCENT=0.1
MAX_LEVERAGE=100
RISK_PER_TRADE=0.01
```

## Usage

1. Start Ollama service:
```bash
ollama serve
```

2. Run the trading bot:
```bash
python trading_bot.py
```

3. Monitor the logs:
```bash
tail -f trading_bot.log
```

## How It Works

### Market Analysis
- Fetches real-time market data from Bybit
- Calculates technical indicators (RSI, MACD, Bollinger Bands, EMA)
- Analyzes market sentiment and volatility
- Uses Ollama AI to generate trading signals

### Risk Management
- Calculates safe position sizes based on account balance
- Determines optimal leverage to avoid liquidation
- Sets stop losses at 0.1% before liquidation price
- Implements take profit levels with 2:1 risk-reward ratio

### Trade Execution
- Validates all trade parameters before execution
- Places market orders with automatic SL/TP
- Monitors positions for early exit opportunities
- Implements cooldown periods between trades

## Supported Symbols

- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)  
- SOLUSDT (Solana)

Additional symbols can be added to the `symbols` list in `trading_bot.py`.

## Liquidation Safety

The bot uses Bybit's liquidation formula to ensure safe trading:

| Leverage | Liquidation Distance | Stop Loss Distance |
|----------|---------------------|-------------------|
| 20x      | 4.50%              | 4.40%             |
| 50x      | 1.50%              | 1.40%             |
| 75x      | 0.83%              | 0.73%             |
| 100x     | 0.50%              | 0.40%             |

## Monitoring

The bot provides comprehensive logging:
- Trade executions with full details
- Market analysis results
- Risk management decisions
- Error handling and warnings

## Disclaimer

**This bot trades real money on live markets. Trading cryptocurrencies involves significant risk and can result in financial loss. Always:**

- Start with small amounts you can afford to lose
- Monitor the bot continuously
- Monitor the bot continuously
- Understand the risks involved
- Never invest more than you can afford to lose

## License

MIT License - see LICENSE file for details.