#!/usr/bin/env python3
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
        testnet=False,  # Using mainnet with your API key
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
