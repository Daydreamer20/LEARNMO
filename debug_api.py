#!/usr/bin/env python3
"""
Debug script to help troubleshoot Bybit API connection issues
"""

import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import requests

load_dotenv()

def check_api_credentials():
    """Check if API credentials are properly loaded"""
    print("🔍 Checking API credentials...")
    
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    testnet = os.getenv('BYBIT_TESTNET', 'False').lower() == 'true'
    
    print(f"API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"API Secret: {'✅ Set' if api_secret else '❌ Missing'}")
    print(f"Testnet Mode: {testnet}")
    
    if api_key:
        print(f"API Key (first 8 chars): {api_key[:8]}...")
    
    return api_key and api_secret

def test_basic_connection():
    """Test basic internet connection to Bybit"""
    print("\n🌐 Testing basic connection to Bybit...")
    
    try:
        # Test connection to Bybit's public endpoint
        response = requests.get("https://api.bybit.com/v5/market/time", timeout=10)
        if response.status_code == 200:
            print("✅ Basic connection to Bybit successful")
            server_time = response.json()
            print(f"   Server time: {server_time}")
            return True
        else:
            print(f"❌ Connection failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_api_with_different_methods():
    """Try different approaches to connect to the API"""
    print("\n🔧 Testing API with different configurations...")
    
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Missing API credentials")
        return False
    
    # Method 1: Standard connection
    try:
        print("Method 1: Standard connection...")
        session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret,
        )
        
        # Try a simple API call
        response = session.get_server_time()
        print(f"✅ Method 1 successful: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: With explicit timeout
    try:
        print("Method 2: With timeout settings...")
        session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret,
            timeout=30
        )
        
        response = session.get_server_time()
        print(f"✅ Method 2 successful: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Test with public endpoint first
    try:
        print("Method 3: Testing public endpoint...")
        session = HTTP(testnet=False)
        
        # Public endpoint (no auth needed)
        response = session.get_tickers(category="linear", symbol="BTCUSDT")
        print(f"✅ Method 3 (public) successful")
        
        # Now try with auth
        session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret,
        )
        
        response = session.get_wallet_balance(accountType="UNIFIED")
        print(f"✅ Method 3 (auth) successful: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    return False

def check_api_permissions():
    """Check what permissions the API key has"""
    print("\n🔑 Checking API permissions...")
    
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    try:
        session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret,
        )
        
        # Try to get API key info
        response = session.get_api_key_information()
        if response.get('retCode') == 0:
            permissions = response['result']
            print("✅ API Key Information:")
            print(f"   Permissions: {permissions.get('permissions', {})}")
            print(f"   IP Restrictions: {permissions.get('ips', 'None')}")
            return True
        else:
            print(f"❌ Failed to get API info: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Permission check failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  Bybit API Debug Tool")
    print("=" * 60)
    
    # Step 1: Check credentials
    if not check_api_credentials():
        print("\n❌ Please check your .env file and API credentials")
        exit(1)
    
    # Step 2: Test basic connection
    if not test_basic_connection():
        print("\n❌ Basic connection failed. Check your internet connection.")
        exit(1)
    
    # Step 3: Test API with different methods
    if test_api_with_different_methods():
        print("\n✅ API connection successful!")
        
        # Step 4: Check permissions
        check_api_permissions()
        
        print("\n🎉 All tests passed! Your API should work with the trading bot.")
    else:
        print("\n❌ All API connection methods failed.")
        print("\nPossible solutions:")
        print("1. Check if your API key is correct")
        print("2. Ensure API key has 'Derivatives' permissions")
        print("3. Check IP whitelist settings on Bybit")
        print("4. Try regenerating your API key")
        print("5. Check if your account is verified for derivatives trading")