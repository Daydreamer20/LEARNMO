import requests
import json
import logging
from config import Config
import pandas as pd

class OllamaAnalyzer:
    def __init__(self):
        self.host = Config.OLLAMA_HOST
        self.model = Config.OLLAMA_MODEL
        self.logger = logging.getLogger(__name__)
        
    def analyze_market_data(self, symbol, price_data, indicators):
        """Analyze market data using Ollama and return trading signal"""
        try:
            # Prepare market context
            context = self._prepare_market_context(symbol, price_data, indicators)
            
            prompt = f"""
            You are a professional scalping trader analyzing {symbol} for quick entry/exit opportunities.
            
            Current Market Data:
            {context}
            
            Trading Rules:
            - Maximum trade size: $2 USD
            - Stop loss: 0.1% before liquidation
            - Strategy: Scalping (quick in/out trades)
            - Risk management is CRITICAL
            
            Based on this data, provide a JSON response with:
            {{
                "signal": "BUY" | "SELL" | "HOLD",
                "confidence": 0-100,
                "entry_price": float,
                "stop_loss": float,
                "take_profit": float,
                "reasoning": "brief explanation",
                "risk_level": "LOW" | "MEDIUM" | "HIGH"
            }}
            
            Only recommend trades with HIGH confidence (>80) and LOW-MEDIUM risk.
            """
            
            response = self._query_ollama(prompt)
            return self._parse_trading_signal(response)
            
        except Exception as e:
            self.logger.error(f"Error analyzing market data: {e}")
            return None
    
    def _prepare_market_context(self, symbol, price_data, indicators):
        """Prepare market context string for Ollama"""
        latest = price_data.iloc[-1] if not price_data.empty else {}
        
        context = f"""
        Symbol: {symbol}
        Current Price: {latest.get('close', 'N/A')}
        24h Change: {((latest.get('close', 0) - price_data.iloc[-24]['close']) / price_data.iloc[-24]['close'] * 100):.2f}% if len(price_data) >= 24 else 'N/A'
        
        Technical Indicators:
        RSI: {indicators.get('rsi', 'N/A')}
        MACD: {indicators.get('macd', 'N/A')}
        BB Upper: {indicators.get('bb_upper', 'N/A')}
        BB Lower: {indicators.get('bb_lower', 'N/A')}
        Volume: {latest.get('volume', 'N/A')}
        
        Recent Price Action:
        High: {latest.get('high', 'N/A')}
        Low: {latest.get('low', 'N/A')}
        """
        
        return context
    
    def _query_ollama(self, prompt):
        """Send query to Ollama"""
        try:
            url = f"{self.host}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            
            # Increase timeout and add retry logic
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except requests.exceptions.Timeout:
            self.logger.error("Ollama request timed out")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.error("Cannot connect to Ollama - is it running?")
            return None
        except Exception as e:
            self.logger.error(f"Error querying Ollama: {e}")
            return None
    
    def _parse_trading_signal(self, response):
        """Parse Ollama response into trading signal"""
        try:
            if not response:
                return None
                
            # Try to parse JSON response
            signal_data = json.loads(response)
            
            # Validate required fields
            required_fields = ['signal', 'confidence', 'reasoning']
            if not all(field in signal_data for field in required_fields):
                self.logger.error("Missing required fields in Ollama response")
                return None
            
            # Validate signal values
            if signal_data['signal'] not in ['BUY', 'SELL', 'HOLD']:
                self.logger.error(f"Invalid signal: {signal_data['signal']}")
                return None
            
            if not (0 <= signal_data['confidence'] <= 100):
                self.logger.error(f"Invalid confidence: {signal_data['confidence']}")
                return None
            
            return signal_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing Ollama JSON response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing trading signal: {e}")
            return None
    
    def validate_trade_signal(self, signal, current_price, balance):
        """Validate if trade signal is safe to execute"""
        if not signal or signal['signal'] == 'HOLD':
            return False
        
        # Check confidence threshold
        if signal['confidence'] < 80:
            self.logger.info(f"Signal confidence too low: {signal['confidence']}")
            return False
        
        # Check risk level
        if signal.get('risk_level') == 'HIGH':
            self.logger.info("Risk level too high, skipping trade")
            return False
        
        # Check if we have enough balance
        if balance < Config.MAX_TRADE_SIZE_USD:
            self.logger.info(f"Insufficient balance: {balance}")
            return False
        
        return True