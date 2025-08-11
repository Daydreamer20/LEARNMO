#!/usr/bin/env python3
"""
Analyze SIRENUSDT data and provide trading insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtester import TradingBacktester

def analyze_sirenusdt_data():
    """Analyze the SIRENUSDT data in detail"""
    print("🔍 Analyzing SIRENUSDT Data...")
    
    # Load and analyze data
    backtester = TradingBacktester()
    data = backtester.load_data("BYBIT_SIRENUSDT.P, 5_e106e.csv", "SIRENUSDT")
    
    if data.empty:
        print("❌ Failed to load data")
        return
    
    print(f"📊 Data Overview:")
    print(f"   Period: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"   Duration: {data['timestamp'].max() - data['timestamp'].min()}")
    print(f"   Data points: {len(data)} (5-minute candles)")
    
    # Price analysis
    print(f"\n💰 Price Analysis:")
    print(f"   Starting price: ${data['close'].iloc[0]:.5f}")
    print(f"   Ending price: ${data['close'].iloc[-1]:.5f}")
    print(f"   Price change: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"   Highest price: ${data['high'].max():.5f}")
    print(f"   Lowest price: ${data['low'].min():.5f}")
    print(f"   Volatility: {data['close'].pct_change().std() * 100:.3f}% per 5min")
    
    # Calculate indicators
    data = backtester.calculate_indicators(data)
    
    # Technical analysis
    print(f"\n📈 Technical Indicators (Final Values):")
    print(f"   RSI: {data['rsi'].iloc[-1]:.1f}")
    print(f"   EMA 9: ${data['ema_9'].iloc[-1]:.5f}")
    print(f"   EMA 21: ${data['ema_21'].iloc[-1]:.5f}")
    print(f"   MACD: {data['macd'].iloc[-1]:.6f}")
    print(f"   BB Upper: ${data['bb_upper'].iloc[-1]:.5f}")
    print(f"   BB Lower: ${data['bb_lower'].iloc[-1]:.5f}")
    print(f"   ATR: {data['atr'].iloc[-1]:.6f}")
    
    # Market conditions analysis
    print(f"\n🎯 Market Conditions Analysis:")
    
    # Trend analysis
    ema_bullish = (data['ema_9'] > data['ema_21']).sum()
    ema_bearish = (data['ema_9'] < data['ema_21']).sum()
    print(f"   EMA Trend: {ema_bullish} bullish vs {ema_bearish} bearish periods")
    
    # RSI distribution
    rsi_oversold = (data['rsi'] < 30).sum()
    rsi_overbought = (data['rsi'] > 70).sum()
    rsi_neutral = ((data['rsi'] >= 30) & (data['rsi'] <= 70)).sum()
    print(f"   RSI Distribution: {rsi_oversold} oversold, {rsi_neutral} neutral, {rsi_overbought} overbought")
    
    # Volatility periods
    high_vol = (data['atr'] > data['atr'].quantile(0.75)).sum()
    low_vol = (data['atr'] < data['atr'].quantile(0.25)).sum()
    print(f"   Volatility: {high_vol} high-vol periods, {low_vol} low-vol periods")
    
    # Test different confidence thresholds
    print(f"\n🧪 Testing Different Confidence Thresholds:")
    
    for confidence in [60, 70, 75, 80, 85, 90]:
        # Generate signals with this confidence
        data_copy = data.copy()
        data_copy = backtester.generate_signals(data_copy, 'technical')
        
        # Count signals above confidence threshold
        strong_signals = (data_copy['confidence'] >= confidence).sum()
        buy_signals = ((data_copy['signal'] == 'BUY') & (data_copy['confidence'] >= confidence)).sum()
        sell_signals = ((data_copy['signal'] == 'SELL') & (data_copy['confidence'] >= confidence)).sum()
        
        print(f"   Confidence {confidence}%: {strong_signals} total ({buy_signals} BUY, {sell_signals} SELL)")
    
    # Optimal parameters suggestion
    print(f"\n💡 Optimization Suggestions:")
    
    # Based on the data characteristics
    avg_rsi = data['rsi'].mean()
    rsi_std = data['rsi'].std()
    
    print(f"   RSI Average: {avg_rsi:.1f} ± {rsi_std:.1f}")
    print(f"   Suggested RSI oversold: {max(20, avg_rsi - rsi_std):.0f}")
    print(f"   Suggested RSI overbought: {min(80, avg_rsi + rsi_std):.0f}")
    
    # EMA analysis
    ema_diff = (data['ema_9'] - data['ema_21']).abs().mean()
    print(f"   Average EMA separation: ${ema_diff:.6f}")
    
    # Volatility-based suggestions
    avg_atr = data['atr'].mean()
    atr_pct = (avg_atr / data['close'].mean()) * 100
    print(f"   Average ATR: {atr_pct:.3f}% of price")
    
    if atr_pct > 1.0:
        print("   📊 High volatility detected - good for scalping")
        print("   💡 Recommendation: Use lower confidence thresholds (70-75%)")
        print("   💡 Recommendation: Shorter EMA periods (5/15 or 7/18)")
    else:
        print("   📊 Moderate volatility - standard parameters OK")
        print("   💡 Recommendation: Standard confidence thresholds (80-85%)")
    
    # Create a simple visualization
    try:
        create_analysis_chart(data)
    except Exception as e:
        print(f"⚠️  Could not create chart: {e}")
    
    return data

def create_analysis_chart(data):
    """Create analysis charts"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Price and EMAs
    axes[0].plot(data.index, data['close'], label='Close Price', linewidth=1)
    axes[0].plot(data.index, data['ema_9'], label='EMA 9', alpha=0.7)
    axes[0].plot(data.index, data['ema_21'], label='EMA 21', alpha=0.7)
    axes[0].fill_between(data.index, data['bb_lower'], data['bb_upper'], alpha=0.2, label='Bollinger Bands')
    axes[0].set_title('SIRENUSDT Price Action with EMAs and Bollinger Bands')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RSI
    axes[1].plot(data.index, data['rsi'], label='RSI', color='purple')
    axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
    axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
    axes[1].axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    axes[1].set_title('RSI Indicator')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MACD
    axes[2].plot(data.index, data['macd'], label='MACD', color='blue')
    axes[2].plot(data.index, data['macd_signal'], label='Signal', color='red')
    axes[2].bar(data.index, data['macd_histogram'], label='Histogram', alpha=0.6, color='gray')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2].set_title('MACD Indicator')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sirenusdt_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Analysis chart saved as sirenusdt_analysis.png")

if __name__ == "__main__":
    analyze_sirenusdt_data()