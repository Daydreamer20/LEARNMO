#!/usr/bin/env python3
"""
Robust startup script for the real-time trading system with fallback options
"""

import asyncio
import logging
from datetime import datetime
import time
import sys

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

async def test_internet_connection():
    """Test internet connectivity"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('https://httpbin.org/get', timeout=5) as response:
                if response.status == 200:
                    logger.info("✓ Internet connection verified")
                    return True
    except Exception as e:
        logger.error(f"✗ Internet connection failed: {e}")
        return False

async def test_bybit_api():
    """Test Bybit API connectivity"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.bybit.com/v5/market/time', timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('retCode') == 0:
                        logger.info("✓ Bybit mainnet API connection verified")
                        return True
    except Exception as e:
        logger.error(f"✗ Bybit API connection failed: {e}")
        return False

async def main():
    """Main startup function with robust error handling"""
    print("🤖 ROBUST REAL-TIME TRADING SYSTEM")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Pre-flight checks
    logger.info("Running pre-flight checks...")
    
    # Check internet connection
    if not await test_internet_connection():
        logger.error("No internet connection. Please check your network.")
        return
    
    # Check Bybit API
    if not await test_bybit_api():
        logger.warning("Bybit API not accessible. Will try alternative methods.")
    
    try:
        # Import components
        from bybit_realtime_data import BybitRealtimeDataCollector, BybitConfig
        from realtime_training_manager import RealtimeTrainingManager
        from live_trading_bot import LiveTradingBot
        from wandb_integration import WandbTradingTracker
        
        # Configuration
        config = BybitConfig(
            testnet=False,  # Using mainnet with your API key
            symbols=["SOONUSDT"]
        )
        
        # Initialize components
        logger.info("Initializing system components...")
        
        data_collector = BybitRealtimeDataCollector(config, db_path="realtime_trading.db")
        
        # Initialize wandb in offline mode if needed
        import os
        if not os.environ.get('WANDB_API_KEY'):
            os.environ['WANDB_MODE'] = 'offline'
            logger.info("Running wandb in offline mode")
        
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
        
        # Start the system
        logger.info("🚀 Starting robust real-time trading system...")
        
        # Start trading bot
        await trading_bot.start_trading()
        
        # Start data collection (with fallbacks)
        await data_collector.start_collection()
        
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
        if 'data_collector' in locals():
            data_collector.stop_collection()
        
        # Final report
        if 'trading_bot' in locals():
            try:
                summary = trading_bot.get_performance_summary()
                logger.info("📊 FINAL SUMMARY:")
                logger.info(f"Balance: ${summary['current_balance']:.2f}")
                logger.info(f"Return: {summary['total_return_pct']:+.2f}%")
                logger.info(f"Trades: {summary['total_trades']}")
                logger.info(f"Win Rate: {summary['win_rate']:.1%}")
            except Exception as e:
                logger.error(f"Error getting final summary: {e}")
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required files are present")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to restart after error
        logger.info("Attempting to restart in 30 seconds...")
        await asyncio.sleep(30)
        
        # Recursive restart (limited to prevent infinite loops)
        if not hasattr(main, 'restart_count'):
            main.restart_count = 0
        
        if main.restart_count < 3:
            main.restart_count += 1
            logger.info(f"Restart attempt {main.restart_count}/3")
            await main()
        else:
            logger.error("Maximum restart attempts reached. Exiting.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem shutdown requested by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)