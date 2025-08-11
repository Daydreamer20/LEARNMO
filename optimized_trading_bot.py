#!/usr/bin/env python3
"""
Optimized SIRENUSDT trading bot with backtested parameters
"""

import logging
import time
import asyncio
from datetime import datetime
from bybit_client import BybitClient
from market_data import MarketDataProvider
from risk_manager import RiskManager
from config import Config
from optimized_bot_config import OPTIMIZED_PARAMS, RISK_PARAMS, TRADING_RULES

class OptimizedTradingBot:
    def __init__(self):
        self.setup_logging()
        
        # Initialize components
        self.bybit = BybitClient()
        self.market_data = MarketDataProvider(self.bybit)
        self.risk_manager = RiskManager()
        
        # Load optimized parameters
        self.params = OPTIMIZED_PARAMS
        self.risk_params = RISK_PARAMS
        self.trading_rules = TRADING_RULES
        
        # Trading state
        self.is_running = False
        self.positions = {}
        self.last_trade_time = {}
        
        # Focus on SIRENUSDT with optimized parameters
        self.symbols = ["SIRENUSDT"]
        
        self.logger.info("Optimized SIRENUSDT trading bot initialized")
        self.logger.info(f"Parameters: RSI({self.params['rsi_period']}) EMA({self.params['ema_fast']}/{self.params['ema_slow']}) Confidence({self.params['confidence_threshold']}%)")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimized_trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the optimized trading bot"""
        self.logger.info("Starting optimized SIRENUSDT trading bot...")
        self.is_running = True
        
        # Check initial setup
        if not self.validate_setup():
            return
        
        # Main trading loop with optimized timing
        while self.is_running:
            try:
                await self.trading_cycle()
                await asyncio.sleep(10)  # 10 seconds for scalping
                
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(30)
    
    def validate_setup(self):
        """Validate bot setup"""
        try:
            # Check API connection
            balance = self.bybit.get_account_balance()
            if balance <= 0:
                self.logger.error("No USDT balance or API connection failed")
                return False
            
            self.logger.info(f"Account balance: ${balance:.2f} USDT")
            
            # Test market data for SIRENUSDT
            test_data = self.market_data.get_kline_data("SIRENUSDT", limit=10)
            if test_data.empty:
                self.logger.error("Cannot get SIRENUSDT market data")
                return False
            
            self.logger.info("Optimized bot validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup validation failed: {e}")
            return False
    
    async def trading_cycle(self):
        """Main trading cycle with optimized logic"""
        for symbol in self.symbols:
            try:
                await self.process_symbol_optimized(symbol)
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
    
    async def process_symbol_optimized(self, symbol):
        """Process trading logic with optimized parameters"""
        try:
            # Get current market data
            kline_data = self.market_data.get_kline_data(symbol, interval="1", limit=100)
            if kline_data.empty:
                self.logger.debug(f"No market data for {symbol}")
                return
            
            # Calculate optimized technical indicators
            indicators = self.calculate_optimized_indicators(kline_data)
            if not indicators:
                self.logger.debug(f"No indicators calculated for {symbol}")
                return
            
            # Get market sentiment
            sentiment = self.market_data.get_market_sentiment(symbol)
            
            # Check if market is suitable with optimized criteria
            is_suitable, reason = self.is_market_suitable_optimized(indicators, sentiment)
            if not is_suitable:
                self.logger.debug(f"{symbol}: {reason}")
                return
            
            # Check existing positions first
            await self.manage_existing_positions_optimized(symbol, indicators)
            
            # Look for new trading opportunities with optimized logic
            await self.look_for_entry_optimized(symbol, kline_data, indicators, sentiment)
            
        except Exception as e:
            self.logger.error(f"Error in process_symbol_optimized for {symbol}: {e}")
    
    def calculate_optimized_indicators(self, df):
        """Calculate technical indicators with optimized parameters"""
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
            
            # EMA with optimized periods
            ema_fast = self.params['ema_fast']
            ema_slow = self.params['ema_slow']
            indicators['ema_fast'] = df['close'].ewm(span=ema_fast).mean().iloc[-1]
            indicators['ema_slow'] = df['close'].ewm(span=ema_slow).mean().iloc[-1]
            
            # Bollinger Bands with optimized parameters
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
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating optimized indicators: {e}")
            return {}
    
    def generate_optimized_signal(self, symbol, indicators, sentiment, current_price):
        """Generate trading signal with optimized parameters"""
        try:
            rsi = indicators.get('rsi', 50)
            ema_fast = indicators.get('ema_fast', current_price)
            ema_slow = indicators.get('ema_slow', current_price)
            bb_upper = indicators.get('bb_upper', current_price * 1.02)
            bb_lower = indicators.get('bb_lower', current_price * 0.98)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            
            # Optimized scoring system
            buy_score = 0
            sell_score = 0
            
            # RSI with optimized thresholds
            rsi_oversold = self.params['rsi_oversold']
            rsi_overbought = self.params['rsi_overbought']
            
            if rsi < rsi_oversold:
                buy_score += 3  # Strong buy signal
            elif rsi > rsi_overbought:
                sell_score += 3  # Strong sell signal
            elif 45 < rsi < 65:  # Neutral zone
                buy_score += 1
                sell_score += 1
            
            # EMA crossover (faster EMAs)
            if ema_fast > ema_slow:
                buy_score += 2
            elif ema_fast < ema_slow:
                sell_score += 2
            
            # Bollinger Bands
            if current_price < bb_lower:
                buy_score += 2
            elif current_price > bb_upper:
                sell_score += 2
            
            # MACD
            if macd > macd_signal:
                buy_score += 1
            elif macd < macd_signal:
                sell_score += 1
            
            # Volume confirmation (more lenient)
            if volume_ratio > self.trading_rules['min_volume_ratio']:
                buy_score += 1
                sell_score += 1
            
            # Trend strength bonus
            trend_strength = abs(ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0
            if trend_strength > self.trading_rules['trend_strength_min']:
                if ema_fast > ema_slow:
                    buy_score += 1
                else:
                    sell_score += 1
            
            # Determine signal with optimized confidence threshold
            confidence_threshold = self.params['confidence_threshold']
            
            if buy_score >= 4 and buy_score > sell_score:
                signal_type = "BUY"
                confidence = min(70 + (buy_score - 4) * 5, 95)
            elif sell_score >= 4 and sell_score > buy_score:
                signal_type = "SELL"
                confidence = min(70 + (sell_score - 4) * 5, 95)
            else:
                signal_type = "HOLD"
                confidence = 0
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                signal_type = "HOLD"
                confidence = 0
            
            reasoning = f"RSI:{rsi:.1f}, EMA:{ema_fast:.5f}/{ema_slow:.5f}, Vol:{volume_ratio:.1f}x, Score:{buy_score}/{sell_score}"
            
            signal = {
                "signal": signal_type,
                "confidence": confidence,
                "entry_price": current_price,
                "reasoning": reasoning,
                "risk_level": "LOW" if confidence > 85 else "MEDIUM"
            }
            
            if signal_type != "HOLD":
                self.logger.info(f"{symbol} Optimized Signal: {signal_type} (confidence: {confidence}%)")
                self.logger.info(f"Reasoning: {reasoning}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating optimized signal: {e}")
            return None
    
    def is_market_suitable_optimized(self, indicators, sentiment):
        """Check if market conditions are suitable with optimized criteria"""
        try:
            # More lenient volatility check for SIRENUSDT
            atr = indicators.get('atr', 0)
            current_price = indicators.get('bb_middle', 0)
            
            if current_price > 0:
                atr_percentage = (atr / current_price) * 100
                if atr_percentage > self.trading_rules['max_atr_percent']:
                    return False, f"Too volatile: {atr_percentage:.2f}% ATR"
            
            # Volume check (more lenient)
            volume_ratio = indicators.get('volume_ratio', 0)
            if volume_ratio < self.trading_rules['min_volume_ratio']:
                return False, f"Low volume: {volume_ratio:.2f}x average"
            
            return True, "Market suitable for optimized scalping"
            
        except Exception as e:
            self.logger.error(f"Error checking market suitability: {e}")
            return False, f"Error: {e}"
    
    async def manage_existing_positions_optimized(self, symbol, indicators):
        """Manage existing positions with optimized logic"""
        try:
            positions = self.bybit.get_positions(symbol)
            
            for position in positions:
                if float(position['size']) == 0:
                    continue
                
                # Check if we should close the position with optimized criteria
                should_close, reason = self.should_close_position_optimized(
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
    
    def should_close_position_optimized(self, position, current_price, indicators):
        """Determine if position should be closed with optimized logic"""
        try:
            if not position or float(position.get('size', 0)) == 0:
                return False, "No position"
            
            entry_price = float(position['avgPrice'])
            side = position['side']
            unrealized_pnl = float(position['unrealisedPnl'])
            
            # Calculate PnL percentage
            pnl_percentage = (unrealized_pnl / (float(position['size']) * entry_price)) * 100
            
            # Quick profit taking for scalping (optimized threshold)
            if pnl_percentage > 0.3:  # 0.3% profit - take it quickly
                return True, f"Quick profit: {pnl_percentage:.3f}%"
            
            # RSI-based exits with optimized thresholds
            rsi = indicators.get('rsi', 50)
            rsi_oversold = self.params['rsi_oversold']
            rsi_overbought = self.params['rsi_overbought']
            
            if side == "Buy":
                # Close long if RSI is overbought
                if rsi > rsi_overbought:
                    return True, f"RSI overbought: {rsi:.1f}"
            else:
                # Close short if RSI is oversold
                if rsi < rsi_oversold:
                    return True, f"RSI oversold: {rsi:.1f}"
            
            # EMA trend reversal with optimized EMAs
            ema_fast = indicators.get('ema_fast', 0)
            ema_slow = indicators.get('ema_slow', 0)
            
            if ema_fast > 0 and ema_slow > 0:
                if side == "Buy" and ema_fast < ema_slow:
                    return True, "Trend reversal detected (long)"
                elif side == "Sell" and ema_fast > ema_slow:
                    return True, "Trend reversal detected (short)"
            
            return False, "Hold position"
            
        except Exception as e:
            self.logger.error(f"Error checking position close: {e}")
            return False, f"Error: {e}"
    
    async def look_for_entry_optimized(self, symbol, kline_data, indicators, sentiment):
        """Look for entry opportunities with optimized logic"""
        try:
            # Check cooldown period (optimized)
            if self.is_in_cooldown(symbol, self.trading_rules['cooldown_minutes']):
                return
            
            # Get current price
            current_price = self.bybit.get_current_price(symbol)
            if not current_price:
                return
            
            # Generate optimized signal
            signal = self.generate_optimized_signal(symbol, indicators, sentiment, current_price)
            if not signal:
                return
            
            # Validate signal
            balance = self.bybit.get_account_balance()
            if not self.validate_trade_signal_optimized(signal, current_price, balance):
                return
            
            # Execute trade with optimized parameters
            await self.execute_trade_optimized(symbol, signal, current_price, balance)
            
        except Exception as e:
            self.logger.error(f"Error looking for entry in {symbol}: {e}")
    
    def validate_trade_signal_optimized(self, signal, current_price, balance):
        """Validate if trade signal is safe to execute with optimized criteria"""
        if not signal or signal['signal'] == 'HOLD':
            return False
        
        # Check confidence threshold (optimized)
        if signal['confidence'] < self.params['confidence_threshold']:
            self.logger.debug(f"Signal confidence too low: {signal['confidence']}")
            return False
        
        # Check if we have enough balance
        if balance < self.risk_params['max_trade_size_usd']:
            self.logger.debug(f"Insufficient balance: {balance}")
            return False
        
        return True
    
    def is_in_cooldown(self, symbol, cooldown_minutes):
        """Check if symbol is in cooldown period"""
        if symbol not in self.last_trade_time:
            return False
        
        time_since_last = time.time() - self.last_trade_time[symbol]
        return time_since_last < (cooldown_minutes * 60)
    
    async def execute_trade_optimized(self, symbol, signal, current_price, balance):
        """Execute a trade with optimized parameters"""
        try:
            # Get symbol information
            symbol_info = self.bybit.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Could not get symbol info for {symbol}")
                return
            
            # Use SIRENUSDT max leverage (25x)
            leverage = min(self.risk_params['max_leverage'], 
                          int(float(symbol_info['leverageFilter']['maxLeverage'])))
            
            # Calculate stop loss and take profit with optimized ratios
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
            self.logger.info(f"Placing OPTIMIZED {side} order for {symbol}: {position_size:.6f} @ {current_price}")
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
                self.logger.info(f"OPTIMIZED order placed successfully: {result['orderId']}")
                self.last_trade_time[symbol] = time.time()
                
                # Log trade details
                self.log_trade_details_optimized(symbol, signal, result, current_price, stop_loss_price, take_profit_price)
            else:
                self.logger.error(f"Failed to place optimized order for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error executing optimized trade for {symbol}: {e}")
    
    def log_trade_details_optimized(self, symbol, signal, order_result, entry_price, stop_loss, take_profit):
        """Log detailed trade information for optimized trades"""
        trade_log = f"""
        === OPTIMIZED TRADE EXECUTED ===
        Symbol: {symbol}
        Signal: {signal['signal']}
        Confidence: {signal['confidence']}%
        Reasoning: {signal['reasoning']}
        Entry Price: {entry_price}
        Stop Loss: {stop_loss}
        Take Profit: {take_profit}
        Order ID: {order_result['orderId']}
        Parameters: RSI({self.params['rsi_period']}) EMA({self.params['ema_fast']}/{self.params['ema_slow']})
        Time: {datetime.now()}
        ================================
        """
        self.logger.info(trade_log)
    
    def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping optimized trading bot...")
        self.is_running = False

# Main execution
async def main():
    print("=" * 80)
    print("  OPTIMIZED SIRENUSDT TRADING BOT")
    print("  (Backtested & Optimized Parameters)")
    print("=" * 80)
    print()
    print("🎯 Optimized for SIRENUSDT scalping")
    print("📊 Based on backtesting analysis")
    print("⚡ 78 trades, 50% win rate, +0.01% return")
    print()
    
    response = input("Start optimized SIRENUSDT trading? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Bot startup cancelled.")
        return
    
    bot = OptimizedTradingBot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        bot.stop()
        print("\nOptimized bot stopped by user")

if __name__ == "__main__":
    asyncio.run(main())