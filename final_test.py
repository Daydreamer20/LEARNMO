#!/usr/bin/env python3
"""
Final test with maximum compatibility settings
"""

import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import time

load_dotenv()

def test_with_max_settings():
    """Test with maximum compatibility settings"""
    print("Testing with maximum compatibility settings...")
    
    try:
        # Create session with maximum recv_window
        session = HTTP(
            testnet=False,
            api_key=os.getenv('BYBIT_API_KEY'),
            api_secret=os.getenv('BYBIT_API_SECRET'),
            recv_window=120000,  # 2 minutes window
        )
        
        print("1. Testing server connectivity...")
        # Try public endpoint first (no auth needed)
        public_response = session.get_tickers(category="linear", symbol="BTCUSDT")
        if public_response['retCode'] == 0:
            print("✅ Public API working")
            btc_price = public_response['result']['list'][0]['lastPrice']
            print(f"   BTC Price: ${float(btc_price):,.2f}")
        else:
            print("❌ Public API failed")
            return False
        
        print("\n2. Testing API key permissions...")
        # Try to get API key info
        try:
            api_info = session.get_api_key_information()
            if api_info['retCode'] == 0:
                print("✅ API key is valid")
                permissions = api_info['result'].get('permissions', {})
                print(f"   Permissions: {permissions}")
            else:
                print(f"❌ API key issue: {api_info}")
        except Exception as e:
            print(f"⚠️  API key check failed: {e}")
        
        print("\n3. Testing account access...")
        # Try different account endpoints
        endpoints_to_try = [
            ("Wallet Balance", lambda: session.get_wallet_balance(accountType="UNIFIED")),
            ("Account Info", lambda: session.get_account_info()),
            ("Coin Balance", lambda: session.get_coin_balance(accountType="UNIFIED", coin="USDT")),
        ]
        
        for name, func in endpoints_to_try:
            try:
                print(f"   Trying {name}...")
                response = func()
                if response['retCode'] == 0:
                    print(f"   ✅ {name} successful")
                    if name == "Wallet Balance" and response['result']['list']:
                        # Extract USDT balance
                        for coin_info in response['result']['list'][0]['coin']:
                            if coin_info['coin'] == 'USDT':
                                balance = float(coin_info['walletBalance'])
                                print(f"      USDT Balance: ${balance:.2f}")
                                if balance > 0:
                                    print("\n🎉 SUCCESS! API is working and you have USDT balance!")
                                    return True
                    return True
                else:
                    print(f"   ❌ {name} failed: {response['retMsg']}")
            except Exception as e:
                print(f"   ❌ {name} error: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def check_system_time():
    """Check if system time might be causing issues"""
    print("\n4. Checking system time...")
    import requests
    try:
        response = requests.get("https://api.bybit.com/v5/market/time", timeout=10)
        if response.status_code == 200:
            server_time = int(response.json()['result']['timeSecond'])
            local_time = int(time.time())
            diff = abs(server_time - local_time)
            print(f"   Time difference: {diff} seconds")
            if diff > 30:
                print("   ⚠️  Large time difference detected!")
                print("   Try synchronizing your system clock")
            else:
                print("   ✅ Time difference is acceptable")
    except Exception as e:
        print(f"   ❌ Time check failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("  FINAL BYBIT API TEST")
    print("=" * 60)
    
    success = test_with_max_settings()
    check_system_time()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 API CONNECTION SUCCESSFUL!")
        print("You can now start the trading bot!")
    else:
        print("❌ API connection still failing")
        print("\nNext steps:")
        print("1. Check if your Bybit account is fully verified")
        print("2. Ensure derivatives trading is enabled")
        print("3. Try regenerating your API key")
        print("4. Check if there are any IP restrictions")
        print("5. Contact Bybit support if issues persist")
    print("=" * 60)