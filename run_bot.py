#!/usr/bin/env python3
"""
Simple script to run the trading bot with proper error handling
"""

import asyncio
import sys
import signal
from trading_bot import ScalpingBot

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal. Stopping bot...")
    sys.exit(0)

async def main():
    """Main function to run the bot"""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 50)
    print("  Ollama-Powered Scalping Trading Bot")
    print("=" * 50)
    print()
    print("⚠️  WARNING: This bot trades REAL MONEY on LIVE markets!")
    print("   Ensure you have proper API keys and sufficient balance.")
    print("   Start with small amounts until you're confident.")
    print()
    
    # Ask for confirmation
    response = input("Are you sure you want to start trading? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Bot startup cancelled.")
        return
    
    # Initialize and start bot
    bot = ScalpingBot()
    
    try:
        print("\n🚀 Starting trading bot...")
        print("   Press Ctrl+C to stop")
        print("-" * 30)
        
        await bot.start()
        
    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot crashed with error: {e}")
        print("   Check trading_bot.log for details")
    finally:
        bot.stop()
        print("   Bot shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())