import logging
import time
import asyncio
from datetime import datetime
from bybit_client import BybitClient
from ollama_analyzer import OllamaAnalyzer
from market_data import MarketDataProvider
from risk_manager import RiskManager
from config import Config

class ScalpingBot:
    def __init__(self):
        self.setup_logging()
        
        # Initialize components
        self.bybit = BybitClient()
        self.analyzer = OllamaAnalyzer()
        self.market_data = MarketDataProvider(self.bybit)
        self.risk_manager = RiskManager()
        
        # Trading state
        self.is_running = False
        self.positions = {}
        self.last_trade_time = {}
        
        # Import available symbols
        try:
            from symbols import POPULAR_SCALPING_PAIRS, is_valid_symbol
            # Add SIRENUSDT to the trading list
            self.symbols = ["SIRENUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"] + POPULAR_SCALPING_PAIRS[:7]
            # Remove duplicates while preserving order
            seen = set()
            self.symbols = [x for x in self.symbols if not (x in seen or seen.add(x))]
        except ImportError:
            # Fallback if symbols module not available
            self.symbols = ["SIRENUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        self.logger.info("Scalping bot initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the trading bot"""
        self.logger.info("Starting scalping bot...")
        self.is_running = True
        
        # Check initial setup
        if not self.validate_setup():
            return
        
        # Main trading loop
        while self.is_running:
            try:
                await self.trading_cycle()
                await asyncio.sleep(10)  # Wait 10 seconds between cycles
                
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def validate_setup(self):
        """Validate bot setup and configuration"""
        try:
            # Check API connection
            balance = self.bybit.get_account_balance()
            if balance <= 0:
                self.logger.error("No USDT balance or API connection failed")
                return False
            
            self.logger.info(f"Account balance: ${balance:.2f} USDT")
            
            # Check Ollama connection (non-blocking)
            test_response = self.analyzer._query_ollama("Test connection")
            if not test_response:
                self.logger.warning("Ollama connection issue - will retry during trading")
            else:
                self.logger.info("Ollama connection verified")
            
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
                self.logger.warning(f"No market data for {symbol}")
                return
            
            # Calculate technical indicators
            indicators = self.market_data.calculate_technical_indicators(kline_data)
            if not indicators:
                self.logger.warning(f"No indicators calculated for {symbol}")
                return
            
            # Get market sentiment
            sentiment = self.market_data.get_market_sentiment(symbol)
            
            # Check if market is suitable for scalping
            is_suitable, reason = self.market_data.is_market_suitable_for_scalping(indicators, sentiment)
            if not is_suitable:
                self.logger.info(f"{symbol}: {reason}")
                return
            
            # Check existing positions first
            await self.manage_existing_positions(symbol, indicators)
            
            # Look for new trading opportunities
            await self.look_for_entry(symbol, kline_data, indicators, sentiment)
            
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
    
    async def look_for_entry(self, symbol, kline_data, indicators, sentiment):
        """Look for entry opportunities"""
        try:
            # Check cooldown period (avoid overtrading)
            if self.is_in_cooldown(symbol):
                return
            
            # Get current price
            current_price = self.bybit.get_current_price(symbol)
            if not current_price:
                return
            
            # Get AI analysis
            signal = self.analyzer.analyze_market_data(symbol, kline_data, indicators)
            if not signal:
                self.logger.debug(f"No AI signal for {symbol} - continuing to next symbol")
                return
            
            # Validate signal
            balance = self.bybit.get_account_balance()
            if not self.analyzer.validate_trade_signal(signal, current_price, balance):
                return
            
            # Execute trade if signal is valid
            await self.execute_trade(symbol, signal, current_price, balance)
            
        except Exception as e:
            self.logger.error(f"Error looking for entry in {symbol}: {e}")
    
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
    bot = ScalpingBot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        bot.stop()
        print("\nBot stopped by user")

if __name__ == "__main__":
    asyncio.run(main())