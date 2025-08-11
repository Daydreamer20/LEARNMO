#!/usr/bin/env python3
"""
Test market data retrieval for SIRENUSDT
"""

from bybit_client import BybitClient
from market_data import MarketDataProvider
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_market_data():
    """Test market data retrieval"""
    print("=" * 50)
    print("  Market Data Test")
    print("=" * 50)
    
    try:
        # Initialize client and market data provider
        client = BybitClient()
        market_data = MarketDataProvider(client)
        
        symbol = "SIRENUSDT"
        
        print(f"📊 Testing market data for {symbol}...")
        
        # Test 1: Get kline data
        print("1. Getting kline data...")
        kline_data = market_data.get_kline_data(symbol, interval="1", limit=50)
        
        if not kline_data.empty:
            print(f"✅ Kline data retrieved: {len(kline_data)} candles")
            print(f"   Latest price: ${kline_data['close'].iloc[-1]:.5f}")
            print(f"   Time range: {kline_data['timestamp'].iloc[0]} to {kline_data['timestamp'].iloc[-1]}")
        else:
            print("❌ No kline data retrieved")
            return False
        
        # Test 2: Calculate indicators
        print("\n2. Calculating technical indicators...")
        indicators = market_data.calculate_technical_indicators(kline_data)
        
        if indicators:
            print("✅ Technical indicators calculated:")
            print(f"   RSI: {indicators.get('rsi', 'N/A'):.2f}")
            print(f"   EMA 9: ${indicators.get('ema_9', 'N/A'):.5f}")
            print(f"   EMA 21: ${indicators.get('ema_21', 'N/A'):.5f}")
            print(f"   ATR: {indicators.get('atr', 'N/A'):.6f}")
        else:
            print("❌ No indicators calculated")
            return False
        
        # Test 3: Get market sentiment
        print("\n3. Getting market sentiment...")
        sentiment = market_data.get_market_sentiment(symbol)
        
        if sentiment:
            print("✅ Market sentiment retrieved:")
            print(f"   24h Change: {sentiment.get('price_change_24h', 'N/A'):.2f}%")
            print(f"   24h Volume: {sentiment.get('volume_24h', 'N/A'):,.0f}")
            print(f"   Funding Rate: {sentiment.get('funding_rate', 'N/A'):.4f}%")
        else:
            print("❌ No sentiment data retrieved")
            return False
        
        # Test 4: Check market suitability
        print("\n4. Checking market suitability...")
        is_suitable, reason = market_data.is_market_suitable_for_scalping(indicators, sentiment)
        
        print(f"Market suitable for scalping: {'✅ Yes' if is_suitable else '❌ No'}")
        print(f"Reason: {reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in market data test: {e}")
        return False

if __name__ == "__main__":
    success = test_market_data()
    
    if success:
        print(f"\n🎉 Market data is working! Bot should run properly now.")
    else:
        print(f"\n❌ Market data issues remain.")
    
    print("=" * 50)