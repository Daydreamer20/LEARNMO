#!/usr/bin/env python3
"""
Balanced optimized bot combining WANDB insights with practical signal frequency
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging

# Configure logging (less verbose)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class BalancedOptimizedBot:
    """Trading bot with balanced WANDB-optimized parameters"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.balance = 1000.0
        self.positions = []
        self.trades = []
        
        # BALANCED PARAMETERS (combining WANDB insights with practical trading)
        self.params = {
            'leverage': 15,                    # Reduced from 20x for safety
            'position_size_pct': 0.15,        # Reduced from 24.3% for risk management
            'stop_loss_pct': 0.020,           # Slightly wider than 2.7%
            'take_profit_pct': 0.035,         # Wider than 2.5% for better capture
            'chunk_size': 150,                # Keep optimal chunk size
            'momentum_threshold': 0.002,      # Reduced from 0.004 for more signals
            'volume_multiplier': 1.4          # Reduced from 1.8 for more signals
        }
        
        # Price and signal history
        self.price_history = []
        self.volume_history = []
        self.rsi_history = []
        
        print("🚀 Initialized Balanced Optimized Trading Bot")
        print(f"📊 Parameters: {self.params}")
    
    def load_and_process_data(self):
        """Load and prepare data for balanced trading"""
        try:
            df = pd.read_csv(self.data_file)
            print(f"📊 Loaded {len(df)} candles from {self.data_file}")
            
            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = np.random.uniform(15000, 45000, len(df))
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI for additional signal confirmation"""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def run_balanced_backtest(self):
        """Run backtest with balanced parameters"""
        df = self.load_and_process_data()
        if df is None:
            return
        
        print("🎯 Starting balanced optimized backtest...")
        
        # Process data in chunks of 150 candles (optimal from WANDB)
        chunk_size = self.params['chunk_size']
        total_chunks = len(df) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(df))
            chunk_data = df.iloc[start_idx:end_idx]
            
            if chunk_idx % 50 == 0:  # Report every 50 chunks
                print(f"⚡ Processing chunk {chunk_idx + 1}/{total_chunks}")
            
            # Process each candle in the chunk
            for idx, row in chunk_data.iterrows():
                await self._process_candle(row)
                
                # Small delay to simulate real-time processing
                await asyncio.sleep(0.001)  # Faster processing
            
            # Report chunk performance every 50 chunks
            if chunk_idx % 50 == 0 and self.trades:
                recent_pnl = sum([t['pnl_amount'] for t in self.trades[-10:]])
                print(f"💰 Recent 10 trades PnL: ${recent_pnl:.2f} | Total trades: {len(self.trades)}")
        
        # Final results
        await self._report_final_results()
    
    async def _process_candle(self, row):
        """Process individual candle with balanced logic"""
        price = float(row['close'])
        volume = float(row.get('volume', 25000))
        
        # Update history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Keep only last 50 candles for analysis
        if len(self.price_history) > 50:
            self.price_history = self.price_history[-50:]
            self.volume_history = self.volume_history[-50:]
        
        # Calculate RSI
        if len(self.price_history) >= 15:
            rsi = self._calculate_rsi(self.price_history)
            self.rsi_history.append(rsi)
            if len(self.rsi_history) > 50:
                self.rsi_history = self.rsi_history[-50:]
        
        # Check for balanced signals
        await self._check_balanced_signals(row, price, volume)
        
        # Manage positions with balanced parameters
        await self._manage_balanced_positions(price)
    
    async def _check_balanced_signals(self, row, price, volume):
        """Check for signals using balanced logic"""
        if len(self.price_history) < 20 or len(self.positions) > 0:
            return
        
        # Use actual BOS signals if available
        bullish_bos = bool(row.get('Bullish BOS', 0))
        bearish_bos = bool(row.get('Bearish BOS', 0))
        
        # Technical analysis
        sma_10 = np.mean(self.price_history[-10:])
        sma_20 = np.mean(self.price_history[-20:])
        
        # Volume confirmation (balanced threshold)
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_spike = volume > avg_volume * self.params['volume_multiplier']
        
        # Momentum confirmation (balanced threshold)
        momentum = (price - self.price_history[-5]) / self.price_history[-5] if len(self.price_history) >= 5 else 0
        strong_momentum = abs(momentum) > self.params['momentum_threshold']
        
        # RSI confirmation
        rsi = self.rsi_history[-1] if self.rsi_history else 50
        rsi_bullish = 30 < rsi < 70  # Not overbought/oversold
        
        # Balanced signal logic (less strict than optimized version)
        bullish_signal = (
            (bullish_bos or (price > sma_10 and sma_10 > sma_20 and momentum > self.params['momentum_threshold'])) and
            (volume_spike or strong_momentum) and  # Either volume OR momentum (not both required)
            rsi_bullish
        )
        
        bearish_signal = (
            (bearish_bos or (price < sma_10 and sma_10 < sma_20 and momentum < -self.params['momentum_threshold'])) and
            (volume_spike or strong_momentum) and  # Either volume OR momentum (not both required)
            rsi_bullish
        )
        
        if bullish_signal:
            await self._open_balanced_position('long', price)
        elif bearish_signal:
            await self._open_balanced_position('short', price)
    
    async def _open_balanced_position(self, side, price):
        """Open position with balanced parameters"""
        # Calculate position size using balanced percentage
        position_size = self.balance * self.params['position_size_pct']
        
        # Calculate balanced stop loss and take profit
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
        
        if len(self.trades) % 100 == 0:  # Log every 100th trade
            print(f"🔥 BALANCED {side.upper()} @ ${price:.4f} | Trade #{len(self.trades)+1}")
    
    async def _manage_balanced_positions(self, price):
        """Manage positions with balanced exit logic"""
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
            await self._close_balanced_position(position, exit_price, exit_reason)
    
    async def _close_balanced_position(self, position, exit_price, exit_reason):
        """Close position with balanced leverage calculation"""
        # Calculate PnL with balanced leverage
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        # Apply balanced leverage
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
        
        if len(self.trades) % 50 == 0:  # Log every 50th trade
            print(f"💰 CLOSED #{len(self.trades)} {position['side'].upper()} | PnL: ${pnl_amount:.2f} | Balance: ${self.balance:.2f}")
    
    async def _report_final_results(self):
        """Report final balanced results"""
        if not self.trades:
            print("❌ No trades executed - signals too strict")
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
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum([t['pnl_amount'] for t in self.trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        print("\n" + "=" * 60)
        print("🏆 BALANCED OPTIMIZED BOT FINAL RESULTS")
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
        print(f"📉 Max Drawdown: ${max_drawdown:.2f}")
        print(f"🎛️ Leverage Used: {self.params['leverage']}x")
        print(f"📏 Position Size: {self.params['position_size_pct']*100:.1f}%")
        print(f"🛑 Stop Loss: {self.params['stop_loss_pct']*100:.1f}%")
        print(f"🎯 Take Profit: {self.params['take_profit_pct']*100:.1f}%")
        print("=" * 60)
        
        # Performance comparison
        if total_return > 50:
            print("🚀 EXCELLENT PERFORMANCE! Balanced approach worked great!")
        elif total_return > 20:
            print("✅ GOOD PERFORMANCE! Balanced optimization successful!")
        elif total_return > 0:
            print("📈 POSITIVE PERFORMANCE! Profitable with balanced approach!")
        else:
            print("📉 Mixed results - WANDB insights need further refinement")
        
        # Trading frequency analysis
        trade_frequency = len(self.trades) / 26874 * 100
        print(f"📊 Trade Frequency: {trade_frequency:.3f}% ({len(self.trades)} trades in 26,874 candles)")

async def main():
    """Main function to run balanced optimized bot"""
    print("🤖 BALANCED WANDB-OPTIMIZED TRADING BOT")
    print("=" * 60)
    print("Combining WANDB insights with practical signal frequency")
    print()
    
    # Find data file
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'SOONUSDT' in f]
    
    if not csv_files:
        print("❌ No SOONUSDT CSV files found!")
        return
    
    csv_file = csv_files[0]  # Use first available file
    print(f"📊 Using data file: {csv_file}")
    
    # Initialize and run balanced bot
    bot = BalancedOptimizedBot(csv_file)
    await bot.run_balanced_backtest()

if __name__ == "__main__":
    asyncio.run(main())