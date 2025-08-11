#!/usr/bin/env python3
"""
Test with manual timestamp adjustment
"""

import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

class ManualBybitClient:
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.base_url = "https://api.bybit.com"
        self.time_offset = self._get_time_offset()
        
    def _get_time_offset(self):
        """Get time offset from Bybit server"""
        try:
            response = requests.get(f"{self.base_url}/v5/market/time", timeout=10)
            if response.status_code == 200:
                server_time = int(response.json()['result']['timeSecond'])
                local_time = int(time.time())
                offset = server_time - local_time
                print(f"Time offset calculated: {offset} seconds")
                return offset
        except Exception as e:
            print(f"Failed to get time offset: {e}")
        return 0
    
    def _generate_signature(self, params, timestamp):
        """Generate signature for API request"""
        param_str = urlencode(sorted(params.items()))
        sign_str = f"{timestamp}{self.api_key}5000{param_str}"
        return hmac.new(
            self.api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint, params=None):
        """Make authenticated request with manual timestamp"""
        if params is None:
            params = {}
        
        # Use server-adjusted timestamp
        timestamp = str(int((time.time() + self.time_offset) * 1000))
        
        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '5000',
            'X-BAPI-SIGN': self._generate_signature(params, timestamp)
        }
        
        url = f"{self.base_url}{endpoint}"
        if params:
            url += "?" + urlencode(params)
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            return response.json()
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def get_wallet_balance(self):
        """Get wallet balance with manual timestamp"""
        return self._make_request("/v5/account/wallet-balance", {"accountType": "UNIFIED"})

def test_manual_sync():
    """Test with manual timestamp synchronization"""
    print("Testing manual timestamp synchronization...")
    
    client = ManualBybitClient()
    
    print("\nTesting wallet balance...")
    response = client.get_wallet_balance()
    
    if response and response.get('retCode') == 0:
        print("✅ Manual sync successful!")
        
        # Extract USDT balance
        try:
            coins = response['result']['list'][0]['coin']
            for coin in coins:
                if coin['coin'] == 'USDT':
                    balance = float(coin['walletBalance'])
                    print(f"💰 USDT Balance: ${balance:.2f}")
                    
                    if balance > 0:
                        print("\n🎉 SUCCESS! You have USDT and API is working!")
                        return True
        except Exception as e:
            print(f"Error parsing balance: {e}")
            
    else:
        print(f"❌ Manual sync failed: {response}")
    
    return False

if __name__ == "__main__":
    print("=" * 50)
    print("  Manual Timestamp Sync Test")
    print("=" * 50)
    
    if test_manual_sync():
        print("\n✅ Manual sync works! We can implement this in the bot.")
    else:
        print("\n❌ Even manual sync failed.")
        print("Try:")
        print("1. Different VPN server location")
        print("2. Regenerate API key")
        print("3. Check account verification status")