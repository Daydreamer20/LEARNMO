#!/usr/bin/env python3
"""
Simple test for Bybit connection using requests (synchronous)
"""

import requests
import json
from datetime import datetime

def test_bybit_connection():
    """Test Bybit mainnet connection"""
    print("🔗 Testing Bybit Mainnet Connection")
    print("=" * 40)
    
    # Test 1: Server time
    print("1. Testing server connection...")
    try:
        response = requests.get('https://api.bybit.com/v5/market/time', timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0:
                server_time = int(data['result']['timeSecond'])
                print(f"   ✓ Server time: {datetime.fromtimestamp(server_time)}")
            else:
                print(f"   ✗ API error: {data}")
                return False
        else:
            print(f"   ✗ HTTP error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Connection error: {e}")
        return False
    
    # Test 2: SOONUSDT data
    print("2. Testing SOONUSDT market data...")
    try:
        url = 'https://api.bybit.com/v5/market/kline'
        params = {
            'category': 'linear',
            'symbol': 'SOONUSDT',
            'interval': '3',
            'limit': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0 and data['result']['list']:
                klines = data['result']['list']
                latest = klines[0]
                
                price = float(latest[4])  # Close price
                volume = float(latest[5])
                timestamp = int(latest[0])
                
                print(f"   ✓ SOONUSDT Price: ${price:.4f}")
                print(f"   ✓ Volume: {volume:,.0f}")
                print(f"   ✓ Last update: {datetime.fromtimestamp(timestamp/1000)}")
                print(f"   ✓ Got {len(klines)} recent candles")
                
                # Show recent price movement
                prices = [float(k[4]) for k in klines]
                print(f"   ✓ Recent prices: {[f'${p:.4f}' for p in prices[:3]]}")
                
            else:
                print(f"   ✗ No data for SOONUSDT: {data}")
                return False
        else:
            print(f"   ✗ HTTP error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Market data error: {e}")
        return False
    
    # Test 3: Check if symbol is active
    print("3. Testing symbol info...")
    try:
        url = 'https://api.bybit.com/v5/market/instruments-info'
        params = {
            'category': 'linear',
            'symbol': 'SOONUSDT'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0 and data['result']['list']:
                symbol_info = data['result']['list'][0]
                status = symbol_info.get('status', 'Unknown')
                print(f"   ✓ Symbol status: {status}")
                print(f"   ✓ Min order qty: {symbol_info.get('lotSizeFilter', {}).get('minOrderQty', 'N/A')}")
            else:
                print(f"   ✗ Symbol info error: {data}")
                return False
        else:
            print(f"   ✗ HTTP error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Symbol info error: {e}")
        return False
    
    print("\n✅ All tests passed! Mainnet API is working.")
    print("✅ SOONUSDT is available for trading.")
    return True

def main():
    success = test_bybit_connection()
    
    if success:
        print("\n🚀 Connection verified! You can now:")
        print("   1. Run: python start_robust_trading_system.py")
        print("   2. Or run: python demo_realtime_system.py (for simulation)")
        print("\n⚠️  Note: Make sure to add your API keys to .env file for live trading")
    else:
        print("\n❌ Connection issues detected.")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()