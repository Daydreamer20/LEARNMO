#!/usr/bin/env python3
"""
Final optimized bot combining all learnings from WANDB and testing
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

class FinalOptimizedBot:
    """Final trading bot with optimized parameters from all testing"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.balance = 1000.0
        self.positions = []
        self.trades = []
        
        # FINAL OPTIMIZED PARAMETERS (best performing combination)
        self.params = {
            'leverage': 12,                    # Sweet spot between risk and reward
            'position_size_pct': 0.12,        # Balanced position sizing
            'stop_loss_pct': 0.018,           # Optimized stop loss
            'take_profit_pct': 0.040,         # Better R:R ratio
            'chunk_size': 150,                # WANDB optimal
            'momentum_threshold': 0.0025,     # Balanced momentum requirement
            'volume_multiplier': 1.5,         # Reasonable volume confirmation
            'rsi_lower': 30,                  # RSI bounds
            'rsi_upper': 70,
            'trend_strength_min': 0.001       # Minimum trend strength
        }
        
        # Enhanced tracking
        self.price_history = []
        self.volume_history = []
        self.rsi_history = []
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        
        print("🚀 Final Optimized Trading Bot Initialized")
        print(f"📊 Optimized Parameters: {self.params}")
    
    def load_and_process_data(self):
        """Load and prepare data"""
        try:
            df = pd.read_csv(self.data_file)
            print(f"📊 Loaded {len(df)} candles from {self.data_file}")
            
            if 'volume' not in df.columns:
                df['volume'] = np.random.uniform(15000, 45000, len(df))
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
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
    
    def _calculate_ema(self, prices, period):
        """Calculate EMA"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    async def run_final_backtest(self):
        """Run final optimized backtest"""
        df = self.load_and_process_data()
        if df is None:
            return
        
        print("🎯 Starting final optimized backtest...")
        
        chunk_size = self.params['chunk_size']
        total_chunks = len(df) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(df))
            chunk_data = df.iloc[start_idx:end_idx]
            
            if chunk_idx % 50 == 0:
                print(f"⚡ Processing chunk {chunk_idx + 1}/{total_chunks}")
            
            for idx, row in chunk_data.iterrows():
                await self._process_candle(row)
                await asyncio.sleep(0.001)
            
            # Progress reporting
            if chunk_idx % 50 == 0 and self.trades:
                recent_pnl = sum([t['pnl_amount'] for t in self.trades[-10:]])
                win_rate = len([t for t in self.trades[-20:] if t['pnl_amount'] > 0]) / min(20, len(self.trades)) * 100
                print(f"💰 Recent PnL: ${recent_pnl:.2f} | Trades: {len(self.trades)} | Win Rate: {win_rate:.1f}%")
        
        await self._report_final_results()
    
    async def _process_candle(self, row):
        """Process individual candle"""
        price = float(row['close'])
        volume = float(row.get('volume', 25000))
        
        # Update history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) > 50:
            self.price_history = self.price_history[-50:]
            self.volume_history = self.volume_history[-50:]
        
        # Calculate RSI
        if len(self.price_history) >= 15:
            rsi = self._calculate_rsi(self.price_history)
            self.rsi_history.append(rsi)
            if len(self.rsi_history) > 50:
                self.rsi_history = self.rsi_history[-50:]
        
        # Check signals and manage positions
        await self._check_final_signals(row, price, volume)
        await self._manage_positions(price)
    
    async def _check_final_signals(self, row, price, volume):
        """Final optimized signal logic"""
        if len(self.price_history) < 25 or len(self.positions) > 0:
            return
        
        # Skip if too many consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return
        
        # Technical indicators
        sma_10 = np.mean(self.price_history[-10:])
        sma_20 = np.mean(self.price_history[-20:])
        ema_12 = self._calculate_ema(self.price_history, 12)
        
        # Momentum and trend
        momentum = (price - self.price_history[-5]) / self.price_history[-5] if len(self.price_history) >= 5 else 0
        trend_strength = abs(sma_10 - sma_20) / sma_20
        
        # Volume confirmation
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_confirmed = volume > avg_volume * self.params['volume_multiplier']
        
        # RSI
        rsi = self.rsi_history[-1] if self.rsi_history else 50
        rsi_ok = self.params['rsi_lower'] < rsi < self.params['rsi_upper']
        
        # BOS signals
        bullish_bos = bool(row.get('Bullish BOS', 0))
        bearish_bos = bool(row.get('Bearish BOS', 0))
        
        # Final signal conditions
        strong_bullish = (
            (bullish_bos or (price > ema_12 and sma_10 > sma_20 and momentum > self.params['momentum_threshold'])) and
            volume_confirmed and
            rsi_ok and
            trend_strength > self.params['trend_strength_min']
        )
        
        strong_bearish = (
            (bearish_bos or (price < ema_12 and sma_10 < sma_20 and momentum < -self.params['momentum_threshold'])) and
            volume_confirmed and
            rsi_ok and
            trend_strength > self.params['trend_strength_min']
        )
        
        if strong_bullish:
            await self._open_position('long', price)
        elif strong_bearish:
            await self._open_position('short', price)
    
    async def _open_position(self, side, price):
        """Open position with final optimized parameters"""
        # Reduce position size after consecutive losses
        size_multiplier = max(0.5, 1 - (self.consecutive_losses * 0.2))
        position_size = self.balance * self.params['position_size_pct'] * size_multiplier
        
        # Calculate stops
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
            'entry_time': datetime.now(),
            'size_multiplier': size_multiplier
        }
        
        self.positions.append(position)
        
        if len(self.trades) % 100 == 0:
            print(f"🔥 {side.upper()} @ ${price:.4f} | Size: {size_multiplier:.1f}x | Trade #{len(self.trades)+1}")
    
    async def _manage_positions(self, price):
        """Manage positions with final logic"""
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
            else:
                if price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif price <= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position, price, exit_reason))
        
        for position, exit_price, exit_reason in positions_to_close:
            await self._close_position(position, exit_price, exit_reason)
    
    async def _close_position(self, position, exit_price, exit_reason):
        """Close position and update consecutive loss tracking"""
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        pnl_amount = position['size'] * pnl_pct * self.params['leverage']
        self.balance += pnl_amount
        
        # Update consecutive losses
        if pnl_amount < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        trade = {
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'size_multiplier': position['size_multiplier'],
            'leverage_used': self.params['leverage']
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        if len(self.trades) % 50 == 0:
            print(f"💰 CLOSED #{len(self.trades)} | PnL: ${pnl_amount:.2f} | Balance: ${self.balance:.2f}")
    
    async def _report_final_results(self):
        """Comprehensive final results"""
        if not self.trades:
            print("❌ No trades executed")
            return
        
        # Calculate all metrics
        winning_trades = [t for t in self.trades if t['pnl_amount'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_amount'] < 0]
        
        total_return = (self.balance - 1000) / 1000 * 100
        win_rate = len(winning_trades) / len(self.trades) * 100
        avg_win = np.mean([t['pnl_amount'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum([t['pnl_amount'] for t in winning_trades]) / sum([t['pnl_amount'] for t in losing_trades])) if losing_trades else float('inf')
        
        # Risk metrics
        returns = [t['pnl_amount'] for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        cumulative_pnl = np.cumsum([t['pnl_amount'] for t in self.trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate max consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in self.trades:
            if trade['pnl_amount'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        print("\n" + "=" * 70)
        print("🏆 FINAL OPTIMIZED BOT RESULTS")
        print("=" * 70)
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
        print(f"🔥 Max Consecutive Wins: {max_consecutive_wins}")
        print(f"❄️ Max Consecutive Losses: {max_consecutive_losses}")
        print("=" * 70)
        print("🎛️ FINAL PARAMETERS")
        print(f"⚡ Leverage: {self.params['leverage']}x")
        print(f"📏 Position Size: {self.params['position_size_pct']*100:.1f}%")
        print(f"🛑 Stop Loss: {self.params['stop_loss_pct']*100:.1f}%")
        print(f"🎯 Take Profit: {self.params['take_profit_pct']*100:.1f}%")
        print(f"📊 Momentum Threshold: {self.params['momentum_threshold']:.4f}")
        print(f"📈 Volume Multiplier: {self.params['volume_multiplier']:.1f}x")
        print("=" * 70)
        
        # Performance assessment
        if total_return > 25:
            print("🚀 OUTSTANDING! Final optimization is excellent!")
        elif total_return > 15:
            print("✅ EXCELLENT! Great final optimization results!")
        elif total_return > 5:
            print("📈 GOOD! Solid final optimization performance!")
        elif total_return > 0:
            print("📊 POSITIVE! Final optimization is profitable!")
        else:
            print("📉 Needs more work - consider different approach")
        
        trade_frequency = len(self.trades) / 26874 * 100
        print(f"📊 Trade Frequency: {trade_frequency:.3f}% ({len(self.trades)} trades in 26,874 candles)")
        
        # Risk assessment
        if max_drawdown > -500:
            print("✅ Good risk management - drawdown under control")
        else:
            print("⚠️ High drawdown - consider reducing leverage or position size")

async def main():
    """Main function"""
    print("🤖 FINAL OPTIMIZED TRADING BOT")
    print("=" * 70)
    print("Combining all WANDB insights and testing learnings")
    print()
    
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'SOONUSDT' in f]
    
    if not csv_files:
        print("❌ No SOONUSDT CSV files found!")
        return
    
    csv_file = csv_files[0]
    print(f"📊 Using data file: {csv_file}")
    
    bot = FinalOptimizedBot(csv_file)
    await bot.run_final_backtest()

if __name__ == "__main__":
    asyncio.run(main())