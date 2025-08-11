#!/usr/bin/env python3
"""
Simple test to check Bybit connection with manual time sync
"""

import time
import requests
from pybit.unified_trading import HTTP
from config import Config

def sync_time_with_bybit():
    """Get Bybit server time and calculate offset"""
    try:
        response = requests.get("https://api.bybit.com/v5/market/time", timeout=10)
        if response.status_code == 200:
            server_time_ms = int(response.json()['result']['timeSecond']) * 1000
            local_time_ms = int(time.time() * 1000)
            offset = server_time_ms - local_time_ms
            print(f"Time offset: {offset}ms")
            return offset
        return 0
    except Exception as e:
        print(f"Failed to sync time: {e}")
        return 0

def test_simple_connection():
    """Test connection with time sync"""
    print("Testing Bybit connection...")
    
    # Sync time first
    time_offset = sync_time_with_bybit()
    
    try:
        # Create session with larger recv_window
        session = HTTP(
            testnet=Config.BYBIT_TESTNET,
            api_key=Config.BYBIT_API_KEY,
            api_secret=Config.BYBIT_API_SECRET,
            recv_window=60000,  # Very large window
        )
        
        # Test 1: Get account balance
        print("Getting account balance...")
        response = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        
        if response['retCode'] == 0:
            balance_info = response['result']['list'][0]['coin'][0]
            balance = float(balance_info['walletBalance'])
            print(f"✅ Balance: ${balance:.2f} USDT")
            
            # Test 2: Get BTC price
            print("Getting BTC price...")
            price_response = session.get_tickers(category="linear", symbol="BTCUSDT")
            if price_response['retCode'] == 0:
                price = float(price_response['result']['list'][0]['lastPrice'])
                print(f"✅ BTC Price: ${price:,.2f}")
                
                print("\n🎉 All tests passed! API is working!")
                return True
            else:
                print(f"❌ Price test failed: {price_response}")
        else:
            print(f"❌ Balance test failed: {response}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    
    return False

if __name__ == "__main__":
    print("=" * 50)
    print("  Simple Bybit Connection Test")
    print("=" * 50)
    
    if test_simple_connection():
        print("\n✅ Ready to start trading!")
    else:
        print("\n❌ Connection issues remain")
        print("\nTry these solutions:")
        print("1. Wait a few minutes and try again")
        print("2. Check if your system clock is accurate")
        print("3. Try a different VPN server location")
        print("4. Regenerate your API key on Bybit")