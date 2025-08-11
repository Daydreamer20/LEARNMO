#!/usr/bin/env python3
"""
Live trading bot that uses real-time optimized parameters
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import json

from bybit_realtime_data import BybitRealtimeDataCollector, BybitConfig
from realtime_training_manager import RealtimeTrainingManager
from wandb_integration import WandbTradingTracker

logger = logging.getLogger(__name__)

class LiveTradingBot:
    """
    Live trading bot that uses continuously optimized parameters
    """
    
    def __init__(self, 
                 data_collector: BybitRealtimeDataCollector,
                 training_manager: RealtimeTrainingManager,
                 wandb_tracker: WandbTradingTracker = None):
        
        self.data_collector = data_collector
        self.training_manager = training_manager
        self.wandb_tracker = wandb_tracker
        
        # Trading state
        self.active_positions = {}
        self.trade_history = []
        self.current_balance = 1000.0  # Starting balance
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Risk management
        self.max_daily_loss = -50.0  # Stop trading if daily loss exceeds this
        self.max_positions = 3  # Maximum concurrent positions
        
        # Add callback for new data
        self.data_collector.add_data_callback(self._on_new_data)
        
        # Trading signals buffer
        self.signal_buffer = []
        self.last_signal_check = 0
    
    async def _on_new_data(self, data_type: str, data: Dict):
        """Process new market data for trading signals"""
        if data_type == 'kline':
            await self._check_trading_signals(data)
            await self._manage_positions(data)
    
    async def _check_trading_signals(self, kline_data: Dict):
        """Check for trading signals based on latest data"""
        try:
            # Get recent data for signal analysis
            symbol = kline_data['symbol']
            recent_data = self.data_collector.get_recent_klines(symbol, limit=50)
            
            if len(recent_data) < 20:
                return
            
            # Prepare data for signal analysis
            df = self._prepare_signal_data(recent_data)
            
            # Check for entry signals
            latest_row = df.iloc[-1]
            
            # Get current optimized parameters
            params = self.training_manager.get_current_parameters()
            
            # Check for bullish signal
            if latest_row.get('Bullish BOS', False) and len(self.active_positions) < self.max_positions:
                await self._execute_trade(symbol, 'long', latest_row['close'], params)
            
            # Check for bearish signal
            elif latest_row.get('Bearish BOS', False) and len(self.active_positions) < self.max_positions:
                await self._execute_trade(symbol, 'short', latest_row['close'], params)
            
        except Exception as e:
            logger.error(f"Error checking trading signals: {e}")
    
    def _prepare_signal_data(self, kline_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for signal generation"""
        df = pd.DataFrame()
        df['timestamp'] = pd.to_datetime(kline_data['timestamp'], unit='ms')
        df['open'] = kline_data['open_price']
        df['high'] = kline_data['high_price']
        df['low'] = kline_data['low_price']
        df['close'] = kline_data['close_price']
        df['volume'] = kline_data['volume']
        
        # Add technical indicators
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Simple signal logic (replace with your actual signals)
        df['price_momentum'] = df['close'].pct_change(3)
        df['volume_spike'] = df['volume'] > df['volume'].rolling(10).mean() * 1.5
        
        # Generate signals
        df['Bullish BOS'] = (
            (df['close'] > df['sma_10']) & 
            (df['sma_10'] > df['sma_20']) & 
            (df['rsi'] > 40) & (df['rsi'] < 70) &
            (df['price_momentum'] > 0.002) &
            df['volume_spike']
        )
        
        df['Bearish BOS'] = (
            (df['close'] < df['sma_10']) & 
            (df['sma_10'] < df['sma_20']) & 
            (df['rsi'] > 30) & (df['rsi'] < 60) &
            (df['price_momentum'] < -0.002) &
            df['volume_spike']
        )
        
        return df.fillna(False)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def _execute_trade(self, symbol: str, side: str, price: float, params: Dict):
        """Execute a trade with current optimized parameters"""
        try:
            # Check daily loss limit
            if self.daily_pnl <= self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
                return
            
            # Calculate position size
            position_size = self.current_balance * params['position_size_pct']
            leverage = params['leverage']
            
            # Calculate stop loss and take profit levels
            if side == 'long':
                stop_loss = price * (1 - params['stop_loss_pct'])
                take_profit = price * (1 + params['take_profit_pct'])
            else:
                stop_loss = price * (1 + params['stop_loss_pct'])
                take_profit = price * (1 - params['take_profit_pct'])
            
            # Create position
            position = {
                'id': f"{symbol}_{side}_{int(time.time())}",
                'symbol': symbol,
                'side': side,
                'entry_price': price,
                'position_size': position_size,
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'status': 'open'
            }
            
            # Add to active positions
            self.active_positions[position['id']] = position
            
            logger.info(f"🔥 TRADE EXECUTED: {side.upper()} {symbol} @ ${price:.4f}")
            logger.info(f"   Size: ${position_size:.2f} | Leverage: {leverage}x")
            logger.info(f"   SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
            
            # Log to wandb
            if self.wandb_tracker:
                self._log_trade_execution(position)
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def _manage_positions(self, kline_data: Dict):
        """Manage active positions (check stop loss, take profit)"""
        current_price = kline_data['close_price']
        symbol = kline_data['symbol']
        
        positions_to_close = []
        
        for position_id, position in self.active_positions.items():
            if position['symbol'] != symbol:
                continue
            
            # Check stop loss and take profit
            should_close = False
            exit_reason = ""
            
            if position['side'] == 'long':
                if current_price <= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif current_price >= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            else:  # short
                if current_price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif current_price <= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position_id, current_price, exit_reason))
        
        # Close positions
        for position_id, exit_price, exit_reason in positions_to_close:
            await self._close_position(position_id, exit_price, exit_reason)
    
    async def _close_position(self, position_id: str, exit_price: float, exit_reason: str):
        """Close a position and calculate PnL"""
        try:
            position = self.active_positions[position_id]
            
            # Calculate PnL
            if position['side'] == 'long':
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
            
            # Apply leverage
            pnl_pct *= position['leverage']
            pnl_amount = position['position_size'] * pnl_pct
            
            # Update balance and stats
            self.current_balance += pnl_amount
            self.daily_pnl += pnl_amount
            self.total_trades += 1
            
            if pnl_amount > 0:
                self.winning_trades += 1
            
            # Create trade record
            trade_record = {
                'position_id': position_id,
                'symbol': position['symbol'],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'position_size': position['position_size'],
                'leverage': position['leverage'],
                'pnl_pct': pnl_pct,
                'pnl_amount': pnl_amount,
                'exit_reason': exit_reason
            }
            
            self.trade_history.append(trade_record)
            
            # Remove from active positions
            del self.active_positions[position_id]
            
            # Log trade closure
            logger.info(f"💰 POSITION CLOSED: {position['side'].upper()} {position['symbol']}")
            logger.info(f"   Entry: ${position['entry_price']:.4f} | Exit: ${exit_price:.4f}")
            logger.info(f"   PnL: ${pnl_amount:.2f} ({pnl_pct*100:.2f}%) | Reason: {exit_reason}")
            logger.info(f"   Balance: ${self.current_balance:.2f}")
            
            # Log to wandb
            if self.wandb_tracker:
                self._log_trade_closure(trade_record)
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _log_trade_execution(self, position: Dict):
        """Log trade execution to wandb"""
        try:
            wandb.log({
                'trade_executed': True,
                'symbol': position['symbol'],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'position_size': position['position_size'],
                'leverage': position['leverage'],
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'current_balance': self.current_balance,
                'active_positions': len(self.active_positions)
            })
        except Exception as e:
            logger.error(f"Error logging trade execution: {e}")
    
    def _log_trade_closure(self, trade: Dict):
        """Log trade closure to wandb"""
        try:
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            wandb.log({
                'trade_closed': True,
                'symbol': trade['symbol'],
                'side': trade['side'],
                'pnl_amount': trade['pnl_amount'],
                'pnl_pct': trade['pnl_pct'],
                'exit_reason': trade['exit_reason'],
                'current_balance': self.current_balance,
                'daily_pnl': self.daily_pnl,
                'total_trades': self.total_trades,
                'win_rate': win_rate,
                'active_positions': len(self.active_positions)
            })
        except Exception as e:
            logger.error(f"Error logging trade closure: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'current_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'total_return_pct': (self.current_balance - 1000) / 1000 * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'active_positions': len(self.active_positions),
            'current_parameters': self.training_manager.get_current_parameters()
        }
    
    async def start_trading(self):
        """Start live trading"""
        logger.info("🚀 Starting live trading bot...")
        logger.info(f"Initial balance: ${self.current_balance:.2f}")
        
        # Start performance reporting
        asyncio.create_task(self._performance_reporter())
        
        # Trading is handled by data callbacks
        logger.info("✅ Live trading bot is active and monitoring markets...")
    
    async def _performance_reporter(self):
        """Periodic performance reporting"""
        while True:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                summary = self.get_performance_summary()
                
                logger.info("📊 PERFORMANCE SUMMARY:")
                logger.info(f"   Balance: ${summary['current_balance']:.2f} ({summary['total_return_pct']:+.2f}%)")
                logger.info(f"   Daily PnL: ${summary['daily_pnl']:+.2f}")
                logger.info(f"   Trades: {summary['total_trades']} | Win Rate: {summary['win_rate']:.1%}")
                logger.info(f"   Active Positions: {summary['active_positions']}")
                
                # Log to wandb
                if self.wandb_tracker:
                    wandb.log({
                        'performance_summary': summary,
                        'timestamp': datetime.now().isoformat()
                    })
                
            except Exception as e:
                logger.error(f"Error in performance reporter: {e}")

async def main():
    """Main function to run the complete live trading system"""
    # Configuration
    config = BybitConfig(
        testnet=True,
        symbols=["SOONUSDT"]
    )
    
    # Initialize components
    data_collector = BybitRealtimeDataCollector(config)
    wandb_tracker = WandbTradingTracker("live-trading-bot")
    
    # Initialize training manager
    training_manager = RealtimeTrainingManager(
        data_collector=data_collector,
        wandb_tracker=wandb_tracker,
        optimization_interval=1800,  # 30 minutes
        min_data_points=200
    )
    
    # Initialize trading bot
    trading_bot = LiveTradingBot(
        data_collector=data_collector,
        training_manager=training_manager,
        wandb_tracker=wandb_tracker
    )
    
    logger.info("🤖 LIVE TRADING SYSTEM STARTING...")
    logger.info("=" * 50)
    
    try:
        # Start trading bot
        await trading_bot.start_trading()
        
        # Start data collection (this will trigger everything)
        await data_collector.start_collection()
        
    except KeyboardInterrupt:
        logger.info("Shutting down live trading system...")
        data_collector.stop_collection()
        
        # Print final performance
        final_summary = trading_bot.get_performance_summary()
        logger.info("📈 FINAL PERFORMANCE:")
        logger.info(f"   Final Balance: ${final_summary['current_balance']:.2f}")
        logger.info(f"   Total Return: {final_summary['total_return_pct']:+.2f}%")
        logger.info(f"   Total Trades: {final_summary['total_trades']}")
        logger.info(f"   Win Rate: {final_summary['win_rate']:.1%}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())