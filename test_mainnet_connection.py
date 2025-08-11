#!/usr/bin/env python3
"""
Test Bybit mainnet API connection
"""

import asyncio
import aiohttp
import os
from datetime import datetime

async def test_mainnet_connection():
    """Test connection to Bybit mainnet"""
    print("🔗 Testing Bybit Mainnet Connection")
    print("=" * 40)
    
    # Test public API (no auth required)
    print("1. Testing public API...")
    try:
        async with aiohttp.ClientSession() as session:
            # Test server time
            async with session.get('https://api.bybit.com/v5/market/time') as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('retCode') == 0:
                        server_time = int(data['result']['timeSecond'])
                        print(f"   ✓ Server time: {datetime.fromtimestamp(server_time)}")
                    else:
                        print(f"   ✗ API error: {data}")
                        return False
                else:
                    print(f"   ✗ HTTP error: {response.status}")
                    return False
    except Exception as e:
        print(f"   ✗ Connection error: {e}")
        return False
    
    # Test SOONUSDT market data
    print("2. Testing SOONUSDT market data...")
    try:
        async with aiohttp.ClientSession() as session:
            url = 'https://api.bybit.com/v5/market/kline'
            params = {
                'category': 'linear',
                'symbol': 'SOONUSDT',
                'interval': '3',
                'limit': 1
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('retCode') == 0 and data['result']['list']:
                        kline = data['result']['list'][0]
                        price = float(kline[4])  # Close price
                        volume = float(kline[5])
                        print(f"   ✓ SOONUSDT Price: ${price:.4f}")
                        print(f"   ✓ Volume: {volume:,.0f}")
                    else:
                        print(f"   ✗ No data for SOONUSDT: {data}")
                        return False
                else:
                    print(f"   ✗ HTTP error: {response.status}")
                    return False
    except Exception as e:
        print(f"   ✗ Market data error: {e}")
        return False
    
    # Test WebSocket URLs
    print("3. Testing WebSocket connectivity...")
    ws_urls = [
        "wss://stream.bybit.com/v5/public/linear",
        "wss://stream.bybit.com/realtime_public"
    ]
    
    import websockets
    
    for ws_url in ws_urls:
        try:
            print(f"   Testing: {ws_url}")
            ws = await asyncio.wait_for(
                websockets.connect(ws_url, ping_interval=20, ping_timeout=10),
                timeout=10.0
            )
            print(f"   ✓ Connected to {ws_url}")
            await ws.close()
            break
        except asyncio.TimeoutError:
            print(f"   ✗ Timeout: {ws_url}")
        except Exception as e:
            print(f"   ✗ Error: {ws_url} - {e}")
    else:
        print("   ✗ All WebSocket URLs failed")
        return False
    
    print("\n✅ All tests passed! Mainnet connection is working.")
    return True

async def main():
    success = await test_mainnet_connection()
    
    if success:
        print("\n🚀 Ready to start real-time trading system!")
        print("Run: python start_robust_trading_system.py")
    else:
        print("\n❌ Connection issues detected.")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    asyncio.run(main())