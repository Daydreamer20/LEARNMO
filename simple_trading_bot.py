#!/usr/bin/env python3
"""
Simplified trading bot that uses technical indicators instead of AI
This allows immediate trading while Ollama issues are resolved
"""

import logging
import time
import asyncio
from datetime import datetime
from bybit_client import BybitClient
from market_data import MarketDataProvider
from risk_manager import RiskManager
from config import Config

class SimpleTradingBot:
    def __init__(self):
        self.setup_logging()
        
        # Initialize components (no Ollama needed)
        self.bybit = BybitClient()
        self.market_data = MarketDataProvider(self.bybit)
        self.risk_manager = RiskManager()
        
        # Trading state
        self.is_running = False
        self.positions = {}
        self.last_trade_time = {}
        
        # Dynamic symbol selection - will be updated with top movers
        self.symbols = []
        self.last_symbol_update = 0
        self.symbol_update_interval = 300  # Update top movers every 5 minutes
        
        self.logger.info("Simple trading bot initialized (no AI dependency)")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('simple_trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the trading bot"""
        self.logger.info("Starting simple trading bot...")
        self.is_running = True
        
        # Check initial setup
        if not self.validate_setup():
            return
        
        # Main trading loop
        while self.is_running:
            try:
                await self.trading_cycle()
                await asyncio.sleep(15)  # Wait 15 seconds between cycles
                
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def validate_setup(self):
        """Validate bot setup"""
        try:
            # Check API connection
            balance = self.bybit.get_account_balance()
            if balance <= 0:
                self.logger.error("No USDT balance or API connection failed")
                return False
            
            self.logger.info(f"Account balance: ${balance:.2f} USDT")
            
            # Test market data
            test_data = self.market_data.get_kline_data("BTCUSDT", limit=10)
            if test_data.empty:
                self.logger.error("Cannot get market data")
                return False
            
            self.logger.info("All systems validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup validation failed: {e}")
            return False
    
    async def trading_cycle(self):
        """Main trading cycle"""
        for symbol in self.symbols:
            try:
                await self.process_symbol(symbol)
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
    
    async def process_symbol(self, symbol):
        """Process trading logic for a single symbol"""
        try:
            # Get current market data
            kline_data = self.market_data.get_kline_data(symbol, interval="1", limit=100)
            if kline_data.empty:
                self.logger.debug(f"No market data for {symbol}")
                return
            
            # Calculate technical indicators
            indicators = self.market_data.calculate_technical_indicators(kline_data)
            if not indicators:
                self.logger.debug(f"No indicators calculated for {symbol}")
                return
            
            # Get market sentiment
            sentiment = self.market_data.get_market_sentiment(symbol)
            
            # Check if market is suitable for scalping
            is_suitable, reason = self.market_data.is_market_suitable_for_scalping(indicators, sentiment)
            if not is_suitable:
                self.logger.debug(f"{symbol}: {reason}")
                return
            
            # Check existing positions first
            await self.manage_existing_positions(symbol, indicators)
            
            # Look for new trading opportunities using technical analysis
            await self.look_for_entry_technical(symbol, kline_data, indicators, sentiment)
            
        except Exception as e:
            self.logger.error(f"Error in process_symbol for {symbol}: {e}")
    
    async def manage_existing_positions(self, symbol, indicators):
        """Manage existing positions"""
        try:
            positions = self.bybit.get_positions(symbol)
            
            for position in positions:
                if float(position['size']) == 0:
                    continue
                
                # Check if we should close the position
                should_close, reason = self.risk_manager.should_close_position(
                    position, 
                    indicators.get('bb_middle', 0), 
                    indicators
                )
                
                if should_close:
                    self.logger.info(f"Closing {symbol} position: {reason}")
                    result = self.bybit.close_position(symbol, position['side'])
                    if result:
                        self.logger.info(f"Position closed successfully: {result['orderId']}")
                
        except Exception as e:
            self.logger.error(f"Error managing positions for {symbol}: {e}")
    
    async def look_for_entry_technical(self, symbol, kline_data, indicators, sentiment):
        """Look for entry opportunities using technical analysis"""
        try:
            # Check cooldown period
            if self.is_in_cooldown(symbol):
                return
            
            # Get current price
            current_price = self.bybit.get_current_price(symbol)
            if not current_price:
                return
            
            # Generate signal using technical indicators
            signal = self.generate_technical_signal(symbol, indicators, sentiment, current_price)
            if not signal:
                return
            
            # Validate signal
            balance = self.bybit.get_account_balance()
            if not self.validate_trade_signal(signal, current_price, balance):
                return
            
            # Execute trade
            await self.execute_trade(symbol, signal, current_price, balance)
            
        except Exception as e:
            self.logger.error(f"Error looking for entry in {symbol}: {e}")
    
    def generate_technical_signal(self, symbol, indicators, sentiment, current_price):
        """Generate trading signal using technical indicators"""
        try:
            rsi = indicators.get('rsi', 50)
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            bb_upper = indicators.get('bb_upper', current_price * 1.02)
            bb_lower = indicators.get('bb_lower', current_price * 0.98)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            price_change_24h = sentiment.get('price_change_24h', 0)
            
            # Scoring system for buy/sell signals
            buy_score = 0
            sell_score = 0
            
            # RSI signals
            if rsi < 30:  # Oversold
                buy_score += 2
            elif rsi > 70:  # Overbought
                sell_score += 2
            elif 40 < rsi < 60:  # Neutral zone
                buy_score += 1
                sell_score += 1
            
            # EMA crossover signals
            if ema_9 > ema_21:  # Bullish trend
                buy_score += 2
            elif ema_9 < ema_21:  # Bearish trend
                sell_score += 2
            
            # Price vs Bollinger Bands
            if current_price < bb_lower:  # Below lower band
                buy_score += 2
            elif current_price > bb_upper:  # Above upper band
                sell_score += 2
            
            # MACD signals
            if macd > macd_signal:  # MACD above signal
                buy_score += 1
            elif macd < macd_signal:  # MACD below signal
                sell_score += 1
            
            # Volume confirmation
            if volume_ratio > 1.5:  # High volume
                buy_score += 1
                sell_score += 1
            
            # 24h momentum
            if price_change_24h > 5:  # Strong positive momentum
                buy_score += 1
            elif price_change_24h < -5:  # Strong negative momentum
                sell_score += 1
            
            # Determine signal
            confidence = 0
            signal_type = "HOLD"
            
            if buy_score >= 4 and buy_score > sell_score:
                signal_type = "BUY"
                confidence = min(85 + (buy_score - 4) * 3, 95)
            elif sell_score >= 4 and sell_score > buy_score:
                signal_type = "SELL"
                confidence = min(85 + (sell_score - 4) * 3, 95)
            
            if signal_type == "HOLD":
                return None
            
            # Create signal object
            signal = {
                "signal": signal_type,
                "confidence": confidence,
                "entry_price": current_price,
                "reasoning": f"Technical analysis: RSI={rsi:.1f}, EMA trend={'bullish' if ema_9 > ema_21 else 'bearish'}, Volume={volume_ratio:.1f}x",
                "risk_level": "LOW" if confidence > 90 else "MEDIUM"
            }
            
            self.logger.info(f"{symbol} Technical Signal: {signal_type} (confidence: {confidence}%)")
            self.logger.info(f"Reasoning: {signal['reasoning']}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating technical signal: {e}")
            return None
    
    def validate_trade_signal(self, signal, current_price, balance):
        """Validate if trade signal is safe to execute"""
        if not signal or signal['signal'] == 'HOLD':
            return False
        
        # Check confidence threshold
        if signal['confidence'] < 80:
            self.logger.debug(f"Signal confidence too low: {signal['confidence']}")
            return False
        
        # Check if we have enough balance
        if balance < Config.MAX_TRADE_SIZE_USD:
            self.logger.debug(f"Insufficient balance: {balance}")
            return False
        
        return True
    
    def is_in_cooldown(self, symbol, cooldown_minutes=5):
        """Check if symbol is in cooldown period"""
        if symbol not in self.last_trade_time:
            return False
        
        time_since_last = time.time() - self.last_trade_time[symbol]
        return time_since_last < (cooldown_minutes * 60)
    
    async def execute_trade(self, symbol, signal, current_price, balance):
        """Execute a trade based on the signal"""
        try:
            # Get symbol information
            symbol_info = self.bybit.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Could not get symbol info for {symbol}")
                return
            
            # Calculate safe leverage
            leverage = self.risk_manager.calculate_safe_leverage(symbol_info, Config.STOP_LOSS_PERCENT)
            
            # Calculate stop loss and take profit
            side = signal['signal']
            stop_loss_price = self.risk_manager.calculate_stop_loss_price(current_price, side, leverage)
            take_profit_price = self.risk_manager.calculate_take_profit_price(
                current_price, stop_loss_price, side
            )
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                balance, current_price, stop_loss_price, leverage
            )
            
            if position_size <= 0:
                self.logger.warning(f"Invalid position size calculated: {position_size}")
                return
            
            # Validate trade parameters
            is_valid, validation_msg = self.risk_manager.validate_trade_parameters(
                symbol_info, position_size, current_price, leverage
            )
            
            if not is_valid:
                self.logger.warning(f"Trade validation failed: {validation_msg}")
                return
            
            # Place the order
            self.logger.info(f"Placing {side} order for {symbol}: {position_size:.6f} @ {current_price}")
            self.logger.info(f"SL: {stop_loss_price:.6f}, TP: {take_profit_price:.6f}, Leverage: {leverage}x")
            
            result = self.bybit.place_order(
                symbol=symbol,
                side=side,
                qty=position_size,
                order_type="Market",
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                leverage=leverage
            )
            
            if result:
                self.logger.info(f"Order placed successfully: {result['orderId']}")
                self.last_trade_time[symbol] = time.time()
                
                # Log trade details
                self.log_trade_details(symbol, signal, result, current_price, stop_loss_price, take_profit_price)
            else:
                self.logger.error(f"Failed to place order for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
    
    def log_trade_details(self, symbol, signal, order_result, entry_price, stop_loss, take_profit):
        """Log detailed trade information"""
        trade_log = f"""
        === TRADE EXECUTED ===
        Symbol: {symbol}
        Signal: {signal['signal']}
        Confidence: {signal['confidence']}%
        Reasoning: {signal['reasoning']}
        Entry Price: {entry_price}
        Stop Loss: {stop_loss}
        Take Profit: {take_profit}
        Order ID: {order_result['orderId']}
        Time: {datetime.now()}
        =====================
        """
        self.logger.info(trade_log)
    
    def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping trading bot...")
        self.is_running = False

# Main execution
async def main():
    print("=" * 60)
    print("  SIMPLE TECHNICAL TRADING BOT")
    print("  (No AI dependency - Pure Technical Analysis)")
    print("=" * 60)
    print()
    print("⚠️  WARNING: This bot trades REAL MONEY!")
    print("   Max $2 per trade with technical indicators")
    print()
    
    response = input("Start trading with technical analysis? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Bot startup cancelled.")
        return
    
    bot = SimpleTradingBot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        bot.stop()
        print("\nBot stopped by user")

if __name__ == "__main__":
    asyncio.run(main())