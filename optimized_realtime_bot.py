#!/usr/bin/env python3
"""
Optimized real-time bot using best parameters from WANDB sweep
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class OptimizedTradingBot:
    """Trading bot with WANDB-optimized parameters"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.balance = 1000.0
        self.positions = []
        self.trades = []
        
        # WANDB-OPTIMIZED PARAMETERS (from best performing run)
        self.params = {
            'leverage': 20,                    # High leverage for amplified returns
            'position_size_pct': 0.243,       # 24.3% position size
            'stop_loss_pct': 0.027,           # 2.7% stop loss
            'take_profit_pct': 0.025,         # 2.5% take profit
            'chunk_size': 150                 # 150-candle chunks
        }
        
        # Price and signal history
        self.price_history = []
        self.volume_history = []
        self.signal_history = []
        
        logger.info("🚀 Initialized Optimized Trading Bot")
        logger.info(f"📊 Parameters: {self.params}")
    
    def load_and_process_data(self):
        """Load and prepare data for optimized trading"""
        try:
            df = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(df)} candles from {self.data_file}")
            
            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = np.random.uniform(15000, 45000, len(df))
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    async def run_optimized_backtest(self):
        """Run backtest with optimized parameters"""
        df = self.load_and_process_data()
        if df is None:
            return
        
        logger.info("🎯 Starting optimized backtest...")
        
        # Process data in chunks of 150 candles (optimized chunk size)
        chunk_size = self.params['chunk_size']
        total_chunks = len(df) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(df))
            chunk_data = df.iloc[start_idx:end_idx]
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks}")
            
            # Process each candle in the chunk
            for idx, row in chunk_data.iterrows():
                await self._process_candle(row)
                
                # Small delay to simulate real-time processing
                await asyncio.sleep(0.01)
            
            # Report chunk performance
            chunk_pnl = sum([t['pnl_amount'] for t in self.trades[-10:] if self.trades])
            logger.info(f"Chunk {chunk_idx + 1} PnL: ${chunk_pnl:.2f}")
        
        # Final results
        await self._report_final_results()
    
    async def _process_candle(self, row):
        """Process individual candle with optimized logic"""
        price = float(row['close'])
        volume = float(row.get('volume', 25000))
        
        # Update history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Keep only last 50 candles for analysis
        if len(self.price_history) > 50:
            self.price_history = self.price_history[-50:]
            self.volume_history = self.volume_history[-50:]
        
        # Check for optimized signals
        await self._check_optimized_signals(row, price, volume)
        
        # Manage positions with optimized parameters
        await self._manage_optimized_positions(price)
    
    async def _check_optimized_signals(self, row, price, volume):
        """Check for signals using optimized logic"""
        if len(self.price_history) < 20 or len(self.positions) > 0:
            return
        
        # Use actual BOS signals if available
        bullish_bos = bool(row.get('Bullish BOS', 0))
        bearish_bos = bool(row.get('Bearish BOS', 0))
        
        # Enhanced technical analysis
        sma_10 = np.mean(self.price_history[-10:])
        sma_20 = np.mean(self.price_history[-20:])
        
        # Volume confirmation (optimized threshold)
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_spike = volume > avg_volume * 1.8  # Stricter volume requirement
        
        # Momentum confirmation
        momentum = (price - self.price_history[-5]) / self.price_history[-5] if len(self.price_history) >= 5 else 0
        strong_momentum = abs(momentum) > 0.004  # 0.4% momentum threshold
        
        # Optimized signal logic
        bullish_signal = (
            (bullish_bos or (price > sma_10 and sma_10 > sma_20 and momentum > 0.004)) and
            volume_spike and strong_momentum
        )
        
        bearish_signal = (
            (bearish_bos or (price < sma_10 and sma_10 < sma_20 and momentum < -0.004)) and
            volume_spike and strong_momentum
        )
        
        if bullish_signal:
            await self._open_optimized_position('long', price)
        elif bearish_signal:
            await self._open_optimized_position('short', price)
    
    async def _open_optimized_position(self, side, price):
        """Open position with optimized parameters"""
        # Calculate position size using optimized percentage
        position_size = self.balance * self.params['position_size_pct']
        
        # Calculate optimized stop loss and take profit
        if side == 'long':
            stop_loss = price * (1 - self.params['stop_loss_pct'])
            take_profit = price * (1 + self.params['take_profit_pct'])
        else:
            stop_loss = price * (1 + self.params['stop_loss_pct'])
            take_profit = price * (1 - self.params['take_profit_pct'])
        
        position = {
            'id': f"{side}_{len(self.trades)}",
            'side': side,
            'entry_price': price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now()
        }
        
        self.positions.append(position)
        
        logger.info(f"🔥 OPTIMIZED {side.upper()} @ ${price:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f} | Size: ${position_size:.2f}")
    
    async def _manage_optimized_positions(self, price):
        """Manage positions with optimized exit logic"""
        positions_to_close = []
        
        for position in self.positions:
            should_close = False
            exit_reason = ""
            
            if position['side'] == 'long':
                if price <= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif price >= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            else:  # short
                if price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif price <= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position, price, exit_reason))
        
        # Close positions
        for position, exit_price, exit_reason in positions_to_close:
            await self._close_optimized_position(position, exit_price, exit_reason)
    
    async def _close_optimized_position(self, position, exit_price, exit_reason):
        """Close position with optimized leverage calculation"""
        # Calculate PnL with optimized leverage
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        # Apply optimized leverage
        pnl_amount = position['size'] * pnl_pct * self.params['leverage']
        self.balance += pnl_amount
        
        # Record trade
        trade = {
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'leverage_used': self.params['leverage']
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        logger.info(f"💰 CLOSED {position['side'].upper()} @ ${exit_price:.4f} | PnL: ${pnl_amount:.2f} ({pnl_pct*100:.2f}%) | Balance: ${self.balance:.2f}")
    
    async def _report_final_results(self):
        """Report final optimized results"""
        if not self.trades:
            logger.info("No trades executed")
            return
        
        # Calculate comprehensive metrics
        winning_trades = [t for t in self.trades if t['pnl_amount'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_amount'] < 0]
        
        total_return = (self.balance - 1000) / 1000 * 100
        win_rate = len(winning_trades) / len(self.trades) * 100
        avg_win = np.mean([t['pnl_amount'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum([t['pnl_amount'] for t in winning_trades]) / sum([t['pnl_amount'] for t in losing_trades])) if losing_trades else float('inf')
        
        # Calculate Sharpe ratio
        returns = [t['pnl_amount'] for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        print("\n" + "=" * 60)
        print("🏆 OPTIMIZED BOT FINAL RESULTS")
        print("=" * 60)
        print(f"💰 Final Balance: ${self.balance:.2f}")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"🎯 Total Trades: {len(self.trades)}")
        print(f"✅ Win Rate: {win_rate:.1f}%")
        print(f"🏅 Winning Trades: {len(winning_trades)}")
        print(f"❌ Losing Trades: {len(losing_trades)}")
        print(f"💵 Average Win: ${avg_win:.2f}")
        print(f"💸 Average Loss: ${avg_loss:.2f}")
        print(f"⚖️ Profit Factor: {profit_factor:.2f}")
        print(f"📊 Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"🎛️ Leverage Used: {self.params['leverage']}x")
        print(f"📏 Position Size: {self.params['position_size_pct']*100:.1f}%")
        print(f"🛑 Stop Loss: {self.params['stop_loss_pct']*100:.1f}%")
        print(f"🎯 Take Profit: {self.params['take_profit_pct']*100:.1f}%")
        print("=" * 60)
        
        # Performance comparison
        if total_return > 50:
            print("🚀 EXCELLENT PERFORMANCE! Bot significantly outperformed!")
        elif total_return > 20:
            print("✅ GOOD PERFORMANCE! Bot performed well!")
        elif total_return > 0:
            print("📈 POSITIVE PERFORMANCE! Bot was profitable!")
        else:
            print("📉 Needs optimization - consider adjusting parameters")

async def main():
    """Main function to run optimized bot"""
    print("🤖 WANDB-OPTIMIZED TRADING BOT")
    print("=" * 50)
    print("Using best parameters discovered by WANDB hyperparameter sweep")
    print()
    
    # Find data file
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'SOONUSDT' in f]
    
    if not csv_files:
        print("❌ No SOONUSDT CSV files found!")
        return
    
    csv_file = csv_files[0]  # Use first available file
    print(f"📊 Using data file: {csv_file}")
    
    # Initialize and run optimized bot
    bot = OptimizedTradingBot(csv_file)
    await bot.run_optimized_backtest()

if __name__ == "__main__":
    asyncio.run(main())