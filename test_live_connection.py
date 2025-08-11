#!/usr/bin/env python3
"""
Test live market connection to Bybit
"""

import asyncio
import websockets
import json
import requests
import time
from datetime import datetime
import ssl
import urllib3

# Disable SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LiveConnectionTest:
    """Test live connection to Bybit"""
    
    def __init__(self):
        # Live WebSocket URLs
        self.ws_urls = [
            "wss://stream.bybit.com/v5/public/linear",
            "wss://stream.bybit.com/realtime_public"
        ]
        self.rest_url = "https://api.bybit.com"
        self.symbol = "SOONUSDT"  # Test our target symbol
    
    async def test_websocket_connection(self):
        """Test WebSocket connection"""
        print("🔌 Testing WebSocket connections...")
        
        for i, ws_url in enumerate(self.ws_urls, 1):
            try:
                print(f"\n📡 Testing WebSocket URL {i}: {ws_url}")
                
                # Try to connect with timeout
                ws = await asyncio.wait_for(
                    websockets.connect(ws_url, ping_interval=20, ping_timeout=10),
                    timeout=10.0
                )
                
                print(f"✅ Connected successfully!")
                
                # Subscribe to SOONUSDT kline data
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"kline.1.{self.symbol}"]  # 1-minute klines
                }
                
                await ws.send(json.dumps(subscribe_msg))
                print(f"📊 Subscribed to {self.symbol} 1-minute klines")
                
                # Listen for a few messages
                message_count = 0
                start_time = time.time()
                
                async for message in ws:
                    data = json.loads(message)
                    message_count += 1
                    
                    if 'topic' in data and data['topic'].startswith('kline'):
                        kline_data = data['data'][0]
                        price = float(kline_data['close'])
                        volume = float(kline_data['volume'])
                        timestamp = datetime.fromtimestamp(int(kline_data['timestamp']) / 1000)
                        
                        print(f"🔥 LIVE DATA: {self.symbol} @ ${price:.4f} | Volume: {volume:.2f} | Time: {timestamp}")
                    
                    # Test for 30 seconds or 5 messages
                    if message_count >= 5 or (time.time() - start_time) > 30:
                        break
                
                await ws.close()
                print(f"✅ WebSocket test successful! Received {message_count} messages")
                return True
                
            except asyncio.TimeoutError:
                print(f"⏰ Connection timeout for {ws_url}")
                continue
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                continue
        
        print("❌ All WebSocket connections failed")
        return False
    
    def test_rest_api(self):
        """Test REST API connection"""
        print("\n🌐 Testing REST API connection...")
        
        try:
            # Test basic connectivity with SSL verification disabled for VPN
            url = f"{self.rest_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': self.symbol,
                'interval': '1',
                'limit': 5
            }
            
            print(f"📡 Fetching data from: {url}")
            # Try with SSL verification disabled first
            response = requests.get(url, params=params, timeout=15, verify=False)
            
            # Suppress SSL warnings
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['retCode'] == 0:
                    klines = data['result']['list']
                    print(f"✅ REST API connection successful!")
                    print(f"📊 Retrieved {len(klines)} klines for {self.symbol}")
                    
                    # Show latest price
                    if klines:
                        latest = klines[0]  # Most recent kline
                        price = float(latest[4])  # Close price
                        volume = float(latest[5])
                        timestamp = datetime.fromtimestamp(int(latest[0]) / 1000)
                        
                        print(f"🔥 LATEST: {self.symbol} @ ${price:.4f} | Volume: {volume:.2f} | Time: {timestamp}")
                    
                    return True
                else:
                    print(f"❌ API Error: {data['retMsg']}")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print("⏰ REST API request timeout")
            return False
        except Exception as e:
            print(f"❌ REST API error: {e}")
            return False
    
    def test_basic_connectivity(self):
        """Test basic internet connectivity"""
        print("\n🌐 Testing basic connectivity...")
        
        try:
            # Test with a simple HTTP request first
            response = requests.get("https://httpbin.org/ip", timeout=10, verify=False)
            if response.status_code == 200:
                print("✅ Basic internet connectivity working")
                return True
            else:
                print("❌ Basic connectivity failed")
                return False
        except Exception as e:
            print(f"❌ Connectivity test failed: {e}")
            return False
    
    def test_symbol_info(self):
        """Test if SOONUSDT is available for trading"""
        print(f"\n🔍 Testing {self.symbol} availability...")
        
        try:
            url = f"{self.rest_url}/v5/market/instruments-info"
            params = {
                'category': 'linear',
                'symbol': self.symbol
            }
            
            # Disable SSL verification for VPN compatibility
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(url, params=params, timeout=15, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['retCode'] == 0 and data['result']['list']:
                    instrument = data['result']['list'][0]
                    
                    print(f"✅ {self.symbol} is available!")
                    print(f"📊 Status: {instrument['status']}")
                    print(f"💰 Min Order Size: {instrument['lotSizeFilter']['minOrderQty']}")
                    print(f"📈 Price Tick: {instrument['priceFilter']['tickSize']}")
                    
                    return True
                else:
                    print(f"❌ {self.symbol} not found or not available")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking symbol info: {e}")
            return False

async def main():
    """Run all connection tests"""
    print("🚀 BYBIT LIVE CONNECTION TEST")
    print("=" * 50)
    
    tester = LiveConnectionTest()
    
    # Test 0: Basic connectivity
    basic_ok = tester.test_basic_connectivity()
    
    # Test 1: Symbol availability
    symbol_ok = tester.test_symbol_info() if basic_ok else False
    
    # Test 2: REST API
    rest_ok = tester.test_rest_api()
    
    # Test 3: WebSocket (only if REST works)
    ws_ok = False
    if rest_ok:
        ws_ok = await tester.test_websocket_connection()
    else:
        print("\n⚠️ Skipping WebSocket test due to REST API failure")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 CONNECTION TEST SUMMARY")
    print("=" * 50)
    print(f"🌐 Basic Connectivity: {'✅ WORKING' if basic_ok else '❌ FAILED'}")
    print(f"🔍 Symbol Available: {'✅ YES' if symbol_ok else '❌ NO'}")
    print(f"🌐 REST API: {'✅ WORKING' if rest_ok else '❌ FAILED'}")
    print(f"🔌 WebSocket: {'✅ WORKING' if ws_ok else '❌ FAILED'}")
    
    if symbol_ok and rest_ok:
        print("\n🎉 READY FOR LIVE TRADING!")
        print("✅ You can now run the real-time trading bot")
        
        if ws_ok:
            print("🚀 WebSocket connection available - optimal performance")
        else:
            print("📡 WebSocket failed - will use REST API fallback")
    else:
        print("\n⚠️ CONNECTION ISSUES DETECTED")
        if not symbol_ok:
            print("❌ SOONUSDT may not be available for trading")
        if not rest_ok:
            print("❌ Cannot connect to Bybit API - check network/firewall")

if __name__ == "__main__":
    asyncio.run(main())