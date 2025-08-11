#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import time

load_dotenv()

print("Quick API Test")
print("=" * 30)

# Check credentials
api_key = os.getenv('BYBIT_API_KEY')
api_secret = os.getenv('BYBIT_API_SECRET')

print(f"API Key: {api_key[:8]}..." if api_key else "Missing")
print(f"API Secret: {'Set' if api_secret else 'Missing'}")

try:
    # Create session with very large recv_window
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
        recv_window=60000,
    )
    
    print("\nTesting balance...")
    response = session.get_wallet_balance(accountType="UNIFIED")
    print(f"Response: {response}")
    
except Exception as e:
    print(f"Error: {e}")

print("\nTesting Ollama...")
import requests
try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gpt-oss", "prompt": "Say OK", "stream": False},
        timeout=10
    )
    print(f"Ollama response: {response.json()}")
except Exception as e:
    print(f"Ollama error: {e}")