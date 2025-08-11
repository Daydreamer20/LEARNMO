#!/usr/bin/env python3
"""
Test SIRENUSDT trading setup
"""

import sys
from bybit_client import BybitClient
from symbols import get_symbol_info, is_valid_symbol
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_siren_trading():
    """Test SIRENUSDT trading capabilities"""
    print("=" * 50)
    print("  SIRENUSDT Trading Test")
    print("=" * 50)
    
    try:
        # Check if SIRENUSDT is valid
        if not is_valid_symbol("SIRENUSDT"):
            print("❌ SIRENUSDT not available")
            return False
        
        print("✅ SIRENUSDT is available for trading")
        
        # Get symbol details
        symbol_info = get_symbol_info("SIRENUSDT")
        if symbol_info:
            print(f"📊 Symbol Details:")
            print(f"   Base Coin: {symbol_info['baseCoin']}")
            print(f"   Quote Coin: {symbol_info['quoteCoin']}")
            print(f"   Min Order Qty: {symbol_info['minOrderQty']}")
            print(f"   Max Order Qty: {symbol_info['maxOrderQty']}")
            print(f"   Max Leverage: {symbol_info['maxLeverage']}x")
            print(f"   Tick Size: {symbol_info['tickSize']}")
        
        # Test API connection
        print(f"\n🔗 Testing API connection...")
        client = BybitClient()
        
        # Get current price
        current_price = client.get_current_price("SIRENUSDT")
        if current_price:
            print(f"✅ Current SIREN Price: ${current_price:.5f}")
        else:
            print("❌ Failed to get SIREN price")
            return False
        
        # Get symbol info from API
        api_symbol_info = client.get_symbol_info("SIRENUSDT")
        if api_symbol_info:
            print(f"✅ API Symbol Info Retrieved")
            print(f"   Status: {api_symbol_info['status']}")
            print(f"   Min Qty: {api_symbol_info['lotSizeFilter']['minOrderQty']}")
            print(f"   Max Leverage: {api_symbol_info['leverageFilter']['maxLeverage']}")
        else:
            print("❌ Failed to get API symbol info")
            return False
        
        # Calculate position size for $2 trade
        balance = client.get_account_balance()
        print(f"\n💰 Account Balance: ${balance:.2f} USDT")
        
        if balance >= 2.0:
            max_qty_for_2usd = 2.0 / current_price
            min_qty = float(symbol_info['minOrderQty'])
            
            print(f"📈 Trading Calculations:")
            print(f"   Max Qty for $2: {max_qty_for_2usd:.0f} SIREN")
            print(f"   Min Order Qty: {min_qty:.0f} SIREN")
            
            if max_qty_for_2usd >= min_qty:
                print(f"✅ Can trade SIRENUSDT with $2 (≈{max_qty_for_2usd:.0f} tokens)")
            else:
                print(f"⚠️  Need at least ${min_qty * current_price:.2f} to trade SIRENUSDT")
        else:
            print("⚠️  Insufficient balance for testing")
        
        print(f"\n🎯 SIRENUSDT is ready for trading!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing SIRENUSDT: {e}")
        return False

if __name__ == "__main__":
    success = test_siren_trading()
    
    if success:
        print(f"\n🚀 Ready to trade SIRENUSDT!")
        print(f"   Run: python run_bot.py")
    else:
        print(f"\n❌ SIRENUSDT trading setup failed")
    
    print("=" * 50)