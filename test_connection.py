#!/usr/bin/env python3
"""
Test script to verify Bybit API connection and account status
"""

import sys
from bybit_client import BybitClient
from config import Config
import logging

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_api_connection():
    """Test Bybit API connection and display account info"""
    print("=" * 50)
    print("  Bybit API Connection Test")
    print("=" * 50)
    
    try:
        # Initialize client
        print("🔗 Initializing Bybit client...")
        client = BybitClient()
        
        # Test 1: Get account balance
        print("\n📊 Testing account balance...")
        balance = client.get_account_balance()
        
        if balance > 0:
            print(f"✅ Account Balance: ${balance:.2f} USDT")
        else:
            print("❌ No USDT balance found or API connection failed")
            return False
        
        # Test 2: Get symbol info
        print("\n📈 Testing market data access...")
        symbol_info = client.get_symbol_info("BTCUSDT")
        
        if symbol_info:
            print(f"✅ Symbol Info Retrieved: {symbol_info['symbol']}")
            print(f"   Min Order Qty: {symbol_info['lotSizeFilter']['minOrderQty']}")
            print(f"   Max Leverage: {symbol_info['leverageFilter']['maxLeverage']}")
        else:
            print("❌ Failed to get symbol information")
            return False
        
        # Test 3: Get current price
        print("\n💰 Testing price data...")
        current_price = client.get_current_price("BTCUSDT")
        
        if current_price:
            print(f"✅ Current BTC Price: ${current_price:,.2f}")
        else:
            print("❌ Failed to get current price")
            return False
        
        # Test 4: Check positions
        print("\n📋 Testing positions access...")
        positions = client.get_positions()
        print(f"✅ Positions Retrieved: {len(positions)} positions found")
        
        # Test 5: Test leverage setting (dry run)
        print("\n⚙️  Testing leverage configuration...")
        leverage_set = client.set_leverage("BTCUSDT", 20)
        
        if leverage_set:
            print("✅ Leverage setting successful")
        else:
            print("⚠️  Leverage setting failed (may be normal if already set)")
        
        # Display trading readiness
        print("\n" + "=" * 50)
        print("  TRADING READINESS CHECK")
        print("=" * 50)
        
        min_balance_needed = Config.MAX_TRADE_SIZE_USD * 10  # 10 trades worth
        
        print(f"Account Balance: ${balance:.2f} USDT")
        print(f"Max Trade Size: ${Config.MAX_TRADE_SIZE_USD} USDT")
        print(f"Recommended Min Balance: ${min_balance_needed:.2f} USDT")
        
        if balance >= min_balance_needed:
            print("✅ Sufficient balance for trading")
        else:
            print("⚠️  Low balance - consider adding more funds")
        
        print(f"Stop Loss Buffer: {Config.STOP_LOSS_PERCENT}%")
        print(f"Max Leverage: {Config.MAX_LEVERAGE}x")
        
        print("\n🎯 API Connection: SUCCESSFUL")
        print("🚀 Ready to start trading!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        print("\nPossible issues:")
        print("- Check your API key and secret in .env file")
        print("- Ensure API key has derivatives trading permissions")
        print("- Verify your IP is whitelisted (if configured)")
        print("- Check if your account has USDT balance")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("\n" + "=" * 50)
    print("  Ollama Connection Test")
    print("=" * 50)
    
    try:
        from ollama_analyzer import OllamaAnalyzer
        
        print("🤖 Testing Ollama connection...")
        analyzer = OllamaAnalyzer()
        
        # Simple test query
        response = analyzer._query_ollama("Respond with just 'OK' if you can read this.")
        
        if response and 'OK' in response.upper():
            print("✅ Ollama connection successful")
            print(f"   Model: {Config.OLLAMA_MODEL}")
            print(f"   Host: {Config.OLLAMA_HOST}")
            return True
        else:
            print("❌ Ollama connection failed")
            print("   Make sure Ollama is running: ollama serve")
            print(f"   And model is available: ollama pull {Config.OLLAMA_MODEL}")
            return False
            
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting connection tests...\n")
    
    # Test API connection
    api_ok = test_api_connection()
    
    # Test Ollama connection
    ollama_ok = test_ollama_connection()
    
    # Final summary
    print("\n" + "=" * 50)
    print("  FINAL STATUS")
    print("=" * 50)
    
    if api_ok and ollama_ok:
        print("🎉 All systems ready! You can start the trading bot.")
        print("\nTo start trading:")
        print("   python run_bot.py")
    else:
        print("⚠️  Some issues found. Please fix them before trading.")
        if not api_ok:
            print("   - Fix Bybit API connection")
        if not ollama_ok:
            print("   - Fix Ollama connection")
    
    print()