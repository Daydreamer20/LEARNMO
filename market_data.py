import pandas as pd
import numpy as np
import ta
from bybit_client import BybitClient
import logging
from datetime import datetime, timedelta

class MarketDataProvider:
    def __init__(self, bybit_client):
        self.client = bybit_client
        self.logger = logging.getLogger(__name__)
        
    def get_kline_data(self, symbol, interval="1", limit=200):
        """Get kline/candlestick data using manual API"""
        try:
            # Use manual API request to avoid timestamp conversion issues
            response = self.client._make_manual_request(
                "/v5/market/kline",
                {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "limit": str(limit)
                }
            )
            
            if response and response['retCode'] == 0:
                klines = response['result']['list']
                
                if not klines:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Convert data types safely
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Sort by timestamp (newest first from Bybit, so reverse)
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting kline data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for scalping"""
        if df.empty or len(df) < 20:
            return {}
        
        try:
            indicators = {}
            
            # RSI (14 period)
            indicators['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            indicators['macd'] = macd.macd().iloc[-1]
            indicators['macd_signal'] = macd.macd_signal().iloc[-1]
            indicators['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
            indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
            indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
            
            # EMA (fast and slow for scalping)
            indicators['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator().iloc[-1]
            indicators['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator().iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma']
            
            # Price action indicators
            indicators['price_change_1m'] = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
            indicators['price_change_5m'] = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100 if len(df) >= 6 else 0
            
            # Volatility
            indicators['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[-1]
            
            # Support and Resistance levels (simple)
            recent_highs = df['high'].tail(20)
            recent_lows = df['low'].tail(20)
            indicators['resistance'] = recent_highs.max()
            indicators['support'] = recent_lows.min()
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def get_market_sentiment(self, symbol):
        """Get basic market sentiment indicators using manual API"""
        try:
            # Get 24h ticker data using manual API
            response = self.client._make_manual_request(
                "/v5/market/tickers",
                {
                    "category": "linear",
                    "symbol": symbol
                }
            )
            
            if response and response['retCode'] == 0 and response['result']['list']:
                ticker = response['result']['list'][0]
                
                sentiment = {
                    'price_change_24h': float(ticker['price24hPcnt']) * 100,
                    'volume_24h': float(ticker['volume24h']),
                    'turnover_24h': float(ticker['turnover24h']),
                    'open_interest': float(ticker.get('openInterest', 0)),
                    'funding_rate': float(ticker.get('fundingRate', 0)) * 100
                }
                
                return sentiment
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting market sentiment: {e}")
            return {}
    
    def is_market_suitable_for_scalping(self, indicators, sentiment):
        """Check if market conditions are suitable for scalping"""
        try:
            # Check volatility (ATR should be reasonable)
            if 'atr' not in indicators or indicators['atr'] == 0:
                return False, "No volatility data"
            
            # Check if market is too volatile (dangerous for scalping)
            current_price = indicators.get('bb_middle', 0)
            if current_price > 0:
                atr_percentage = (indicators['atr'] / current_price) * 100
                if atr_percentage > 2.0:  # More than 2% ATR is too volatile
                    return False, f"Too volatile: {atr_percentage:.2f}% ATR"
            
            # Check volume (need sufficient volume for scalping)
            volume_ratio = indicators.get('volume_ratio', 0)
            if volume_ratio < 0.1:  # Volume too low (reduced threshold)
                return False, f"Low volume: {volume_ratio:.2f}x average"
            
            # Check if price is trending (good for scalping)
            ema_9 = indicators.get('ema_9', 0)
            ema_21 = indicators.get('ema_21', 0)
            if ema_9 == 0 or ema_21 == 0:
                return False, "No trend data"
            
            # Market is suitable if we have clear trend and good volume
            trend_strength = abs(ema_9 - ema_21) / ema_21 * 100 if ema_21 > 0 else 0
            
            if trend_strength < 0.1:  # Very weak trend
                return False, f"Weak trend: {trend_strength:.3f}%"
            
            return True, "Market suitable for scalping"
            
        except Exception as e:
            self.logger.error(f"Error checking market suitability: {e}")
            return False, f"Error: {e}"