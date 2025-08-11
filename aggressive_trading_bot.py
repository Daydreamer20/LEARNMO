#!/usr/bin/env python3
"""
Aggressive SIRENUSDT trading bot with backtested optimized parameters:
- RSI: 14 period, 41/73 levels (optimized for SIREN)
- EMA: 7/18 periods (faster for scalping)
- Confidence: 75% threshold (more opportunities)
- Results: 78 trades, 50% win rate, +0.01% return
"""

import logging
import time
import asyncio
import numpy as np
from datetime import datetime
from bybit_client import BybitClient
from market_data import MarketDataProvider
from risk_manager import RiskManager
from config import Config

class AggressiveTradingBot:
    def __init__(self):
        self.setup_logging()
        
        # Initialize components
        self.bybit = BybitClient()
        self.market_data = MarketDataProvider(self.bybit)
        self.risk_manager = RiskManager()
        
        # Optimized parameters from backtesting
        self.params = {
            "rsi_period": 14,
            "rsi_oversold": 41,      # Optimized for SIREN behavior
            "rsi_overbought": 73,    # Optimized for SIREN behavior
            "ema_fast": 7,           # Faster for scalping
            "ema_slow": 18,          # Faster for scalping
            "bb_period": 20,
            "bb_std": 2.0,
            "confidence_threshold": 75,  # Lower for more trades
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9
        }
        
        # Aggressive risk parameters
        self.risk_params = {
            "max_trade_size_usd": 2.0,    # Keep safe $2 max
            "stop_loss_percent": 0.1,     # 0.1% before liquidation
            "max_leverage": 25,           # SIRENUSDT max leverage
            "take_profit_ratio": 2.0,     # 2:1 risk-reward
            "quick_profit_threshold": 0.3 # Take profit at 0.3% for scalping
        }
        
        # Aggressive trading rules
        self.trading_rules = {
            "min_volume_ratio": 0.8,      # More lenient volume
            "max_atr_percent": 3.0,       # Higher volatility tolerance
            "cooldown_minutes": 3,        # Shorter cooldown
            "max_daily_trades": 100,      # More trades allowed
            "trend_strength_min": 0.05    # Lower trend requirement
        }
        
        # Trading state
        self.is_running = False
        self.positions = {}
        self.last_trade_time = {}
        self.daily_trades = 0
        
        # Focus on SIRENUSDT
        self.symbols = ["SIRENUSDT"]
        
        self.logger.info("Aggressive SIRENUSDT trading bot initialized")
        self.logger.info(f"Optimized Parameters: RSI({self.params['rsi_period']}) EMA({self.params['ema_fast']}/{self.params['ema_slow']}) Confidence({self.params['confidence_threshold']}%)")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('aggressive_trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the aggressive trading bot"""
        self.logger.info("🚀 Starting Aggressive SIRENUSDT Trading Bot...")
        self.logger.info("📊 Using backtested optimized parameters")
        self.logger.info("⚡ Expected: 78 trades, 50% win rate, +0.01% return")
        
        self.is_running = True
        
        # Check initial setup
        if not self.validate_setup():
            return
        
        # Main trading loop - aggressive timing
        while self.is_running:
            try:
                await self.trading_cycle()
                await asyncio.sleep(5)  # 5 seconds for aggressive scalping
                
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(15)  # Shorter error recovery
    
    def validate_setup(self):
        """Validate bot setup"""
        try:
            # Check API connection
            balance = self.bybit.get_account_balance()
            if balance <= 0:
                self.logger.error("No USDT balance or API connection failed")
                return False
            
            self.logger.info(f"💰 Account balance: ${balance:.2f} USDT")
            
            # Test SIRENUSDT market data
            test_data = self.market_data.get_kline_data("SIRENUSDT", limit=10)
            if test_data.empty:
                self.logger.error("Cannot get SIRENUSDT market data")
                return False
            
            # Get current SIREN price
            current_price = self.bybit.get_current_price("SIRENUSDT")
            if current_price:
                self.logger.info(f"📈 Current SIREN price: ${current_price:.5f}")
            
            self.logger.info("✅ Aggressive bot validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup validation failed: {e}")
            return False
    
    async def trading_cycle(self):
        """Main aggressive trading cycle"""
        for symbol in self.symbols:
            try:
                await self.process_symbol_aggressive(symbol)
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
    
    async def process_symbol_aggressive(self, symbol):
        """Process trading logic with aggressive optimized parameters"""
        try:
            # Get current market data
            kline_data = self.market_data.get_kline_data(symbol, interval="1", limit=100)
            if kline_data.empty:
                return
            
            # Calculate optimized indicators
            indicators = self.calculate_optimized_indicators(kline_data)
            if not indicators:
                return
            
            # Get market sentiment
            sentiment = self.market_data.get_market_sentiment(symbol)
            
            # Check if market is suitable (more aggressive criteria)
            is_suitable, reason = self.is_market_suitable_aggressive(indicators, sentiment)
            if not is_suitable:
                self.logger.debug(f"{symbol}: {reason}")
                return
            
            # Manage existing positions first
            await self.manage_positions_aggressive(symbol, indicators)
            
            # Look for new aggressive entries
            await self.look_for_aggressive_entry(symbol, indicators, sentiment)
            
        except Exception as e:
            self.logger.error(f"Error in aggressive processing for {symbol}: {e}")
    
    def calculate_optimized_indicators(self, df):
        """Calculate indicators with optimized parameters"""
        try:
            indicators = {}
            
            # RSI with optimized period
            rsi_period = self.params['rsi_period']
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi_series.iloc[-1]
            
            # EMA with optimized faster periods
            ema_fast = self.params['ema_fast']
            ema_slow = self.params['ema_slow']
            indicators['ema_fast'] = df['close'].ewm(span=ema_fast).mean().iloc[-1]
            indicators['ema_slow'] = df['close'].ewm(span=ema_slow).mean().iloc[-1]
            
            # Bollinger Bands
            bb_period = self.params['bb_period']
            bb_std = self.params['bb_std']
            bb_middle = df['close'].rolling(window=bb_period).mean()
            bb_std_val = df['close'].rolling(window=bb_period).std()
            indicators['bb_upper'] = (bb_middle + (bb_std_val * bb_std)).iloc[-1]
            indicators['bb_lower'] = (bb_middle - (bb_std_val * bb_std)).iloc[-1]
            indicators['bb_middle'] = bb_middle.iloc[-1]
            
            # MACD
            ema_12 = df['close'].ewm(span=self.params['macd_fast']).mean()
            ema_26 = df['close'].ewm(span=self.params['macd_slow']).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=self.params['macd_signal']).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = macd_signal.iloc[-1]
            
            # ATR for volatility
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            import pandas as pd
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr'] = true_range.rolling(window=14).mean().iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma']
            
            # Price momentum
            indicators['price_change_5m'] = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def generate_aggressive_signal(self, symbol, indicators, sentiment, current_price):
        """Generate aggressive trading signal with optimized parameters"""
        try:
            rsi = indicators.get('rsi', 50)
            ema_fast = indicators.get('ema_fast', current_price)
            ema_slow = indicators.get('ema_slow', current_price)
            bb_upper = indicators.get('bb_upper', current_price * 1.02)
            bb_lower = indicators.get('bb_lower', current_price * 0.98)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            price_change_5m = indicators.get('price_change_5m', 0)
            
            # Aggressive scoring system
            buy_score = 0
            sell_score = 0
            
            # RSI with optimized SIREN-specific thresholds
            rsi_oversold = self.params['rsi_oversold']  # 41
            rsi_overbought = self.params['rsi_overbought']  # 73
            
            if rsi < rsi_oversold:
                buy_score += 4  # Strong buy signal
            elif rsi > rsi_overbought:
                sell_score += 4  # Strong sell signal
            elif 45 < rsi < 65:  # Neutral zone
                buy_score += 1
                sell_score += 1
            
            # EMA crossover with faster EMAs (7/18)
            ema_diff = abs(ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0
            if ema_fast > ema_slow:
                buy_score += 3
                if ema_diff > 0.001:  # Strong trend
                    buy_score += 1
            elif ema_fast < ema_slow:
                sell_score += 3
                if ema_diff > 0.001:  # Strong trend
                    sell_score += 1
            
            # Bollinger Bands
            if current_price < bb_lower:
                buy_score += 3
            elif current_price > bb_upper:
                sell_score += 3
            
            # MACD
            if macd > macd_signal:
                buy_score += 2
            elif macd < macd_signal:
                sell_score += 2
            
            # Volume confirmation (aggressive)
            if volume_ratio > self.trading_rules['min_volume_ratio']:
                buy_score += 2
                sell_score += 2
            
            # Price momentum bonus
            if abs(price_change_5m) > 0.1:  # Strong 5-minute movement
                if price_change_5m > 0:
                    buy_score += 1
                else:
                    sell_score += 1
            
            # Market sentiment from 24h change
            price_change_24h = sentiment.get('price_change_24h', 0)
            if price_change_24h > 5:  # Strong bullish momentum
                buy_score += 1
            elif price_change_24h < -5:  # Strong bearish momentum
                sell_score += 1
            
            # Determine signal with optimized confidence
            confidence_threshold = self.params['confidence_threshold']  # 75%
            
            if buy_score >= 5 and buy_score > sell_score:
                signal_type = "BUY"
                confidence = min(70 + (buy_score - 5) * 4, 95)
            elif sell_score >= 5 and sell_score > buy_score:
                signal_type = "SELL"
                confidence = min(70 + (sell_score - 5) * 4, 95)
            else:
                signal_type = "HOLD"
                confidence = 0
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                signal_type = "HOLD"
                confidence = 0
            
            reasoning = f"RSI:{rsi:.1f}({rsi_oversold}/{rsi_overbought}), EMA:{ema_fast:.5f}/{ema_slow:.5f}, Vol:{volume_ratio:.1f}x, Score:{buy_score}/{sell_score}"
            
            signal = {
                "signal": signal_type,
                "confidence": confidence,
                "entry_price": current_price,
                "reasoning": reasoning,
                "risk_level": "MEDIUM"  # Aggressive but controlled
            }
            
            if signal_type != "HOLD":
                self.logger.info(f"🎯 {symbol} AGGRESSIVE Signal: {signal_type} (confidence: {confidence}%)")
                self.logger.info(f"📊 {reasoning}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating aggressive signal: {e}")
            return None
    
    def is_market_suitable_aggressive(self, indicators, sentiment):
        """Check if market is suitable with aggressive criteria"""
        try:
            # More aggressive volatility tolerance
            atr = indicators.get('atr', 0)
            current_price = indicators.get('bb_middle', 0)
            
            if current_price > 0:
                atr_percentage = (atr / current_price) * 100
                if atr_percentage > self.trading_rules['max_atr_percent']:  # 3%
                    return False, f"Extremely volatile: {atr_percentage:.2f}% ATR"
            
            # More lenient volume check
            volume_ratio = indicators.get('volume_ratio', 0)
            if volume_ratio < self.trading_rules['min_volume_ratio']:  # 0.8x
                return False, f"Low volume: {volume_ratio:.2f}x average"
            
            # Daily trade limit
            if self.daily_trades >= self.trading_rules['max_daily_trades']:
                return False, f"Daily trade limit reached: {self.daily_trades}"
            
            return True, "Market suitable for aggressive scalping"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    async def manage_positions_aggressive(self, symbol, indicators):
        """Manage positions with aggressive profit-taking"""
        try:
            positions = self.bybit.get_positions(symbol)
            
            for position in positions:
                if float(position['size']) == 0:
                    continue
                
                should_close, reason = self.should_close_position_aggressive(position, indicators)
                
                if should_close:
                    self.logger.info(f"🔥 Closing {symbol} position aggressively: {reason}")
                    result = self.bybit.close_position(symbol, position['side'])
                    if result:
                        self.logger.info(f"✅ Position closed: {result['orderId']}")
                
        except Exception as e:
            self.logger.error(f"Error managing aggressive positions: {e}")
    
    def should_close_position_aggressive(self, position, indicators):
        """Aggressive position closing logic"""
        try:
            entry_price = float(position['avgPrice'])
            side = position['side']
            unrealized_pnl = float(position['unrealisedPnl'])
            current_price = indicators.get('bb_middle', entry_price)
            
            # Calculate PnL percentage
            pnl_percentage = (unrealized_pnl / (float(position['size']) * entry_price)) * 100
            
            # Very aggressive profit taking
            if pnl_percentage > self.risk_params['quick_profit_threshold']:  # 0.3%
                return True, f"Quick profit: {pnl_percentage:.3f}%"
            
            # RSI-based exits with optimized thresholds
            rsi = indicators.get('rsi', 50)
            
            if side == "Buy":
                if rsi > self.params['rsi_overbought']:  # 73
                    return True, f"RSI overbought: {rsi:.1f}"
            else:
                if rsi < self.params['rsi_oversold']:  # 41
                    return True, f"RSI oversold: {rsi:.1f}"
            
            # EMA trend reversal with faster EMAs
            ema_fast = indicators.get('ema_fast', 0)
            ema_slow = indicators.get('ema_slow', 0)
            
            if ema_fast > 0 and ema_slow > 0:
                if side == "Buy" and ema_fast < ema_slow:
                    return True, "Fast EMA trend reversal (long)"
                elif side == "Sell" and ema_fast > ema_slow:
                    return True, "Fast EMA trend reversal (short)"
            
            return False, "Hold position"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    async def look_for_aggressive_entry(self, symbol, indicators, sentiment):
        """Look for aggressive entry opportunities"""
        try:
            # Shorter cooldown for aggressive trading
            if self.is_in_cooldown(symbol, self.trading_rules['cooldown_minutes']):  # 3 minutes
                return
            
            # Get current price
            current_price = self.bybit.get_current_price(symbol)
            if not current_price:
                return
            
            # Generate aggressive signal
            signal = self.generate_aggressive_signal(symbol, indicators, sentiment, current_price)
            if not signal or signal['signal'] == 'HOLD':
                return
            
            # Validate signal
            balance = self.bybit.get_account_balance()
            if balance < self.risk_params['max_trade_size_usd']:
                return
            
            # Execute aggressive trade
            await self.execute_aggressive_trade(symbol, signal, current_price, balance)
            
        except Exception as e:
            self.logger.error(f"Error in aggressive entry: {e}")
    
    def is_in_cooldown(self, symbol, cooldown_minutes):
        """Check cooldown with aggressive timing"""
        if symbol not in self.last_trade_time:
            return False
        
        time_since_last = time.time() - self.last_trade_time[symbol]
        return time_since_last < (cooldown_minutes * 60)
    
    async def execute_aggressive_trade(self, symbol, signal, current_price, balance):
        """Execute trade with aggressive parameters"""
        try:
            # Get symbol info
            symbol_info = self.bybit.get_symbol_info(symbol)
            if not symbol_info:
                return
            
            # Use maximum available leverage for SIRENUSDT
            leverage = self.risk_params['max_leverage']  # 25x
            
            # Calculate aggressive stop loss and take profit
            side = signal['signal']
            stop_loss_price = self.risk_manager.calculate_stop_loss_price(current_price, side, leverage)
            take_profit_price = self.risk_manager.calculate_take_profit_price(
                current_price, stop_loss_price, side, self.risk_params['take_profit_ratio']
            )
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                balance, current_price, stop_loss_price, leverage
            )
            
            if position_size <= 0:
                return
            
            # Validate parameters
            is_valid, validation_msg = self.risk_manager.validate_trade_parameters(
                symbol_info, position_size, current_price, leverage
            )
            
            if not is_valid:
                self.logger.warning(f"Validation failed: {validation_msg}")
                return
            
            # Place aggressive order
            self.logger.info(f"🔥 AGGRESSIVE {side} {symbol}: {position_size:.0f} tokens @ ${current_price:.5f}")
            self.logger.info(f"⚡ SL: ${stop_loss_price:.5f}, TP: ${take_profit_price:.5f}, Leverage: {leverage}x")
            
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
                self.logger.info(f"🚀 AGGRESSIVE order executed: {result['orderId']}")
                self.last_trade_time[symbol] = time.time()
                self.daily_trades += 1
                
                # Log aggressive trade details
                self.log_aggressive_trade(symbol, signal, result, current_price, stop_loss_price, take_profit_price, leverage)
            
        except Exception as e:
            self.logger.error(f"Error executing aggressive trade: {e}")
    
    def log_aggressive_trade(self, symbol, signal, order_result, entry_price, stop_loss, take_profit, leverage):
        """Log aggressive trade details"""
        trade_log = f"""
        🔥 === AGGRESSIVE TRADE EXECUTED ===
        Symbol: {symbol}
        Signal: {signal['signal']} (Confidence: {signal['confidence']}%)
        Entry: ${entry_price:.5f}
        Stop Loss: ${stop_loss:.5f}
        Take Profit: ${take_profit:.5f}
        Leverage: {leverage}x
        Reasoning: {signal['reasoning']}
        Order ID: {order_result['orderId']}
        Daily Trades: {self.daily_trades}
        Time: {datetime.now()}
        ===================================
        """
        self.logger.info(trade_log)
    
    def stop(self):
        """Stop the aggressive trading bot"""
        self.logger.info("🛑 Stopping aggressive trading bot...")
        self.is_running = False

# Main execution
async def main():
    print("🔥" * 40)
    print("  AGGRESSIVE SIRENUSDT TRADING BOT")
    print("  Optimized Parameters from Backtesting")
    print("🔥" * 40)
    print()
    print("📊 Backtested Results:")
    print("   • 78 trades executed")
    print("   • 50.0% win rate")
    print("   • +0.01% total return")
    print("   • 1.14 profit factor")
    print("   • 0.01% max drawdown")
    print()
    print("⚡ Aggressive Features:")
    print("   • 5-second scan intervals")
    print("   • 3-minute cooldowns")
    print("   • 0.3% quick profit taking")
    print("   • RSI 41/73 levels (SIREN-optimized)")
    print("   • EMA 7/18 periods (fast scalping)")
    print("   • 75% confidence threshold")
    print()
    print("⚠️  WARNING: This trades REAL MONEY aggressively!")
    print()
    
    response = input("Start AGGRESSIVE SIRENUSDT trading? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("🛑 Aggressive bot startup cancelled.")
        return
    
    bot = AggressiveTradingBot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        bot.stop()
        print("\n🛑 Aggressive bot stopped by user")

if __name__ == "__main__":
    asyncio.run(main())