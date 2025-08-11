# Real-time Trading System Configuration

# Bybit API Configuration
BYBIT_TESTNET = False  # Using mainnet since you have mainnet API key
BYBIT_API_KEY = ""
BYBIT_API_SECRET = ""

# Trading Configuration
INITIAL_BALANCE = 1000.0
MAX_DAILY_LOSS = -50.0
MAX_POSITIONS = 3

# Optimization Configuration
OPTIMIZATION_INTERVAL = 1800  # 30 minutes
MIN_DATA_POINTS = 200

# Symbols to trade
SYMBOLS = ["SOONUSDT"]

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "trading_bot.log"

# Database Configuration
DATABASE_PATH = "realtime_trading.db"

# wandb Configuration
WANDB_PROJECT = "realtime-trading-bot"
WANDB_ENTITY = ""  # Your wandb username/team
