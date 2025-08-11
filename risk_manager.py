import logging
from config import Config
import math

class RiskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, balance, entry_price, stop_loss_price, leverage):
        """Calculate safe position size based on risk management"""
        try:
            # Maximum USD we can risk per trade
            max_trade_usd = min(Config.MAX_TRADE_SIZE_USD, balance * Config.RISK_PER_TRADE)
            
            # Calculate stop loss distance in percentage
            stop_distance = abs(entry_price - stop_loss_price) / entry_price
            
            # Calculate position size in base currency
            # Position size = Risk Amount / (Stop Distance * Entry Price)
            position_size_usd = max_trade_usd / stop_distance
            
            # Convert to quantity (contracts)
            quantity = position_size_usd / entry_price
            
            # Apply leverage constraint
            max_quantity_with_leverage = (max_trade_usd * leverage) / entry_price
            quantity = min(quantity, max_quantity_with_leverage)
            
            self.logger.info(f"Calculated position size: {quantity:.6f} contracts (${position_size_usd:.2f} notional)")
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def calculate_safe_leverage(self, symbol_info, stop_loss_percent):
        """Calculate safe leverage based on stop loss and liquidation margins"""
        try:
            # Get available leverages and their liquidation margins
            available_leverages = [20, 50, 75, 100]
            
            for leverage in sorted(available_leverages, reverse=True):
                liquidation_margin = Config.LIQUIDATION_MARGINS.get(leverage, 0.5)
                
                # Ensure our stop loss is well before liquidation
                safety_buffer = 0.05  # 0.05% additional buffer
                required_margin = stop_loss_percent + safety_buffer
                
                if required_margin < liquidation_margin:
                    self.logger.info(f"Selected leverage {leverage}x (liquidation at {liquidation_margin}%, stop at {stop_loss_percent}%)")
                    return leverage
            
            # If no leverage is safe, use minimum
            self.logger.warning(f"No safe leverage found for {stop_loss_percent}% stop loss, using 20x")
            return 20
            
        except Exception as e:
            self.logger.error(f"Error calculating safe leverage: {e}")
            return 20
    
    def calculate_stop_loss_price(self, entry_price, side, leverage):
        """Calculate stop loss price based on liquidation safety"""
        try:
            # Get liquidation margin for this leverage
            liquidation_margin = Config.LIQUIDATION_MARGINS.get(leverage, 0.5) / 100
            
            # Our stop loss should be Config.STOP_LOSS_PERCENT before liquidation
            stop_loss_margin = (liquidation_margin - (Config.STOP_LOSS_PERCENT / 100))
            
            if side.upper() == "BUY":
                # For long positions, stop loss is below entry
                stop_loss_price = entry_price * (1 - stop_loss_margin)
            else:
                # For short positions, stop loss is above entry
                stop_loss_price = entry_price * (1 + stop_loss_margin)
            
            self.logger.info(f"Stop loss price: {stop_loss_price:.6f} ({stop_loss_margin*100:.3f}% from entry)")
            return stop_loss_price
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return entry_price * 0.999 if side.upper() == "BUY" else entry_price * 1.001
    
    def calculate_take_profit_price(self, entry_price, stop_loss_price, side, risk_reward_ratio=2.0):
        """Calculate take profit based on risk-reward ratio"""
        try:
            # Calculate risk (distance to stop loss)
            risk_distance = abs(entry_price - stop_loss_price)
            
            # Calculate reward (risk * ratio)
            reward_distance = risk_distance * risk_reward_ratio
            
            if side.upper() == "BUY":
                take_profit_price = entry_price + reward_distance
            else:
                take_profit_price = entry_price - reward_distance
            
            self.logger.info(f"Take profit price: {take_profit_price:.6f} (R:R = 1:{risk_reward_ratio})")
            return take_profit_price
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return entry_price * 1.002 if side.upper() == "BUY" else entry_price * 0.998
    
    def validate_trade_parameters(self, symbol_info, quantity, price, leverage):
        """Validate trade parameters against exchange limits"""
        try:
            if not symbol_info:
                return False, "No symbol information"
            
            # Check minimum quantity
            min_qty = float(symbol_info.get('lotSizeFilter', {}).get('minOrderQty', 0))
            if quantity < min_qty:
                return False, f"Quantity {quantity} below minimum {min_qty}"
            
            # Check maximum quantity
            max_qty = float(symbol_info.get('lotSizeFilter', {}).get('maxOrderQty', float('inf')))
            if quantity > max_qty:
                return False, f"Quantity {quantity} above maximum {max_qty}"
            
            # Check price precision
            tick_size = float(symbol_info.get('priceFilter', {}).get('tickSize', 0.01))
            if price % tick_size != 0:
                return False, f"Price {price} not aligned with tick size {tick_size}"
            
            # Check leverage limits
            max_leverage = int(symbol_info.get('leverageFilter', {}).get('maxLeverage', 100))
            if leverage > max_leverage:
                return False, f"Leverage {leverage} above maximum {max_leverage}"
            
            return True, "Parameters valid"
            
        except Exception as e:
            self.logger.error(f"Error validating trade parameters: {e}")
            return False, f"Validation error: {e}"
    
    def should_close_position(self, position, current_price, indicators):
        """Determine if position should be closed based on market conditions"""
        try:
            if not position or float(position.get('size', 0)) == 0:
                return False, "No position"
            
            entry_price = float(position['avgPrice'])
            side = position['side']
            unrealized_pnl = float(position['unrealisedPnl'])
            
            # Check if we're in profit and should take it (scalping - quick profits)
            pnl_percentage = (unrealized_pnl / (float(position['size']) * entry_price)) * 100
            
            # For scalping, take profits quickly
            if pnl_percentage > 0.2:  # 0.2% profit - take it
                return True, f"Taking quick profit: {pnl_percentage:.3f}%"
            
            # Check technical indicators for exit signals
            rsi = indicators.get('rsi', 50)
            
            if side == "Buy":
                # Close long if RSI is overbought
                if rsi > 75:
                    return True, f"RSI overbought: {rsi:.1f}"
            else:
                # Close short if RSI is oversold
                if rsi < 25:
                    return True, f"RSI oversold: {rsi:.1f}"
            
            # Check if trend is reversing
            ema_9 = indicators.get('ema_9', 0)
            ema_21 = indicators.get('ema_21', 0)
            
            if ema_9 > 0 and ema_21 > 0:
                if side == "Buy" and ema_9 < ema_21:
                    return True, "Trend reversal detected (long)"
                elif side == "Sell" and ema_9 > ema_21:
                    return True, "Trend reversal detected (short)"
            
            return False, "Hold position"
            
        except Exception as e:
            self.logger.error(f"Error checking position close: {e}")
            return False, f"Error: {e}"