#!/usr/bin/env python3
"""
Setup script for the real-time trading and optimization system
"""

import os
import sys
import subprocess
import sqlite3
from datetime import datetime

def install_requirements():
    """Install required packages"""
    requirements = [
        'websockets',
        'asyncio',
        'pandas',
        'numpy',
        'sqlite3',
        'requests',
        'matplotlib',
        'seaborn',
        'wandb'
    ]
    
    print("Installing required packages...")
    
    for package in requirements:
        try:
            if package == 'sqlite3':
                continue  # Built-in module
            
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"{package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
            return False
    
    return True

def setup_database():
    """Initialize the database"""
    print("Setting up database...")
    
    db_path = "realtime_trading.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                initial_balance REAL,
                final_balance REAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert default configuration
        cursor.execute('''
            INSERT OR REPLACE INTO system_config (key, value) VALUES 
            ('initial_balance', '1000.0'),
            ('max_daily_loss', '-50.0'),
            ('max_positions', '3'),
            ('optimization_interval', '1800'),
            ('min_data_points', '200')
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"Database initialized: {db_path}")
        return True
        
    except Exception as e:
        print(f"Database setup failed: {e}")
        return False

def setup_wandb():
    """Setup Weights & Biases"""
    print("Setting up Weights & Biases...")
    
    try:
        import wandb
        
        # Check if already logged in
        try:
            wandb.init(project="test-connection", mode="disabled")
            wandb.finish()
            print("wandb is already configured")
            return True
        except:
            pass
        
        print("Please configure wandb:")
        print("1. Go to https://wandb.ai/authorize")
        print("2. Copy your API key")
        print("3. Run: wandb login")
        
        api_key = input("Enter your wandb API key (or press Enter to skip): ").strip()
        
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
            
            # Test connection
            wandb.init(project="realtime-trading-setup", mode="disabled")
            wandb.finish()
            
            print("wandb configured successfully")
            return True
        else:
            print("wandb setup skipped - you can configure it later")
            return True
            
    except Exception as e:
        print(f"wandb setup failed: {e}")
        return False

def create_config_file():
    """Create configuration file"""
    print("Creating configuration file...")
    
    config_content = '''# Real-time Trading System Configuration

# Bybit API Configuration
BYBIT_TESTNET = True
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
'''
    
    try:
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print("Configuration file created: config.py")
        return True
        
    except Exception as e:
        print(f"Failed to create config file: {e}")
        return False

def create_startup_script():
    """Create startup script"""
    print("Creating startup script...")
    
    startup_content = '''#!/usr/bin/env python3
"""
Startup script for the real-time trading system
"""

import asyncio
import logging
from datetime import datetime

from bybit_realtime_data import BybitRealtimeDataCollector, BybitConfig
from realtime_training_manager import RealtimeTrainingManager
from live_trading_bot import LiveTradingBot
from wandb_integration import WandbTradingTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main startup function"""
    print("REAL-TIME TRADING SYSTEM")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Configuration
    config = BybitConfig(
        testnet=True,  # Set to False for live trading
        symbols=["SOONUSDT"]
    )
    
    # Initialize components
    logger.info("Initializing system components...")
    
    data_collector = BybitRealtimeDataCollector(config, db_path="realtime_trading.db")
    wandb_tracker = WandbTradingTracker("realtime-trading-system")
    
    training_manager = RealtimeTrainingManager(
        data_collector=data_collector,
        wandb_tracker=wandb_tracker,
        optimization_interval=1800,  # 30 minutes
        min_data_points=200
    )
    
    trading_bot = LiveTradingBot(
        data_collector=data_collector,
        training_manager=training_manager,
        wandb_tracker=wandb_tracker
    )
    
    try:
        # Start the system
        logger.info("Starting real-time trading system...")
        
        await trading_bot.start_trading()
        await data_collector.start_collection()
        
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
        data_collector.stop_collection()
        
        # Final report
        summary = trading_bot.get_performance_summary()
        logger.info("FINAL SUMMARY:")
        logger.info(f"Balance: ${summary['current_balance']:.2f}")
        logger.info(f"Return: {summary['total_return_pct']:+.2f}%")
        logger.info(f"Trades: {summary['total_trades']}")
        logger.info(f"Win Rate: {summary['win_rate']:.1%}")
    
    except Exception as e:
        logger.error(f"System error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    try:
        with open('start_trading_system.py', 'w', encoding='utf-8') as f:
            f.write(startup_content)
        
        print("Startup script created: start_trading_system.py")
        return True
        
    except Exception as e:
        print(f"Failed to create startup script: {e}")
        return False

def main():
    """Main setup function"""
    print("REAL-TIME TRADING SYSTEM SETUP")
    print("=" * 50)
    print()
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    print()
    
    # Setup database
    if not setup_database():
        success = False
    
    print()
    
    # Setup wandb
    if not setup_wandb():
        success = False
    
    print()
    
    # Create config file
    if not create_config_file():
        success = False
    
    print()
    
    # Create startup script
    if not create_startup_script():
        success = False
    
    print()
    print("=" * 50)
    
    if success:
        print("SETUP COMPLETED SUCCESSFULLY!")
        print()
        print("NEXT STEPS:")
        print("1. Edit config.py with your API keys (optional for testnet)")
        print("2. Run: python start_trading_system.py")
        print("3. Monitor your wandb dashboard for real-time metrics")
        print()
        print("USEFUL COMMANDS:")
        print("   Start system: python start_trading_system.py")
        print("   View logs: type trading_bot.log")
        print("   Check database: sqlite3 realtime_trading.db")
        print()
        print("IMPORTANT:")
        print("   - System starts in TESTNET mode by default")
        print("   - Change BYBIT_TESTNET=False in config.py for live trading")
        print("   - Always test thoroughly before live trading")
        
    else:
        print("SETUP FAILED!")
        print("Please check the errors above and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())