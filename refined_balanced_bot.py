#!/usr/bin/env python3
"""
Refined balanced bot with improved risk management and signal filtering
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import time

class RefinedBalancedBot:
    """Trading bot with refined balanced parameters based on results analysis"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.balance = 1000.0
        self.positions = []
        self.trades = []
        
        # REFINED PARAMETERS (addressing the issues from balanced version)
        self.params = {
            'leverage': 10,                    # Reduced leverage for better risk management
            'position_size_pct': 0.10,        # Smaller position size for safety
            'stop_loss_pct': 0.015,           # Tighter stop loss
            'take_profit_pct': 0.045,         # Wider take profit for better R:R
            'chunk_size': 150,                # Keep optimal chunk size
            'momentum_threshold': 0.003,      # Slightly higher for quality signals
            'volume_multiplier': 1.6,         # Balanced volume requirement
            'rsi_oversold': 25,               # More aggressive RSI levels
            'rsi_overbought': 75,
            'trend_confirmation_period': 15   # Trend confirmation period
        }
        
        # Enhanced tracking
        self.price_history = []
        self.volume_history = []
        self.rsi_history = []
        self.signal_quality_scores = []
        
        print("🚀 Initialized Refined Balanced Trading Bot")
        print(f"📊 Parameters: {self.params}")
    
    def load_and_process_data(self):
        """Load and prepare data"""
        try:
            df = pd.read_csv(self.data_file)
            print(f"📊 Loaded {len(df)} candles from {self.data_file}")
            
            # Add volume if missing
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
    
    def _calculate_signal_quality(self, price, volume, momentum, rsi, trend_strength):
        """Calculate signal quality score (0-100)"""
        score = 0
        
        # Momentum quality (0-30 points)
        if abs(momentum) > self.params['momentum_threshold'] * 2:
            score += 30
        elif abs(momentum) > self.params['momentum_threshold']:
            score += 20
        elif abs(momentum) > self.params['momentum_threshold'] * 0.5:
            score += 10
        
        # Volume quality (0-25 points)
        avg_volume = np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else volume
        volume_ratio = volume / avg_volume
        if volume_ratio > 2.0:
            score += 25
        elif volume_ratio > self.params['volume_multiplier']:
            score += 15
        elif volume_ratio > 1.2:
            score += 10
        
        # RSI quality (0-25 points)
        if self.params['rsi_oversold'] < rsi < self.params['rsi_overbought']:
            score += 25
        elif 20 < rsi < 80:
            score += 15
        elif 15 < rsi < 85:
            score += 10
        
        # Trend strength (0-20 points)
        if trend_strength > 0.8:
            score += 20
        elif trend_strength > 0.6:
            score += 15
        elif trend_strength > 0.4:
            score += 10
        
        return min(score, 100)
    
    async def run_refined_backtest(self):
        """Run backtest with refined parameters"""
        df = self.load_and_process_data()
        if df is None:
            return
        
        print("🎯 Starting refined balanced backtest...")
        
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
            
            # Report performance every 50 chunks
            if chunk_idx % 50 == 0 and self.trades:
                recent_pnl = sum([t['pnl_amount'] for t in self.trades[-10:]])
                avg_quality = np.mean(self.signal_quality_scores[-10:]) if self.signal_quality_scores else 0
                print(f"💰 Recent PnL: ${recent_pnl:.2f} | Trades: {len(self.trades)} | Avg Signal Quality: {avg_quality:.1f}")
        
        await self._report_final_results()
    
    async def _process_candle(self, row):
        """Process individual candle with refined logic"""
        price = float(row['close'])
        volume = float(row.get('volume', 25000))
        
        # Update history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Keep history manageable
        if len(self.price_history) > 50:
            self.price_history = self.price_history[-50:]
            self.volume_history = self.volume_history[-50:]
        
        # Calculate RSI
        if len(self.price_history) >= 15:
            rsi = self._calculate_rsi(self.price_history)
            self.rsi_history.append(rsi)
            if len(self.rsi_history) > 50:
                self.rsi_history = self.rsi_history[-50:]
        
        # Check for refined signals
        await self._check_refined_signals(row, price, volume)
        
        # Manage positions
        await self._manage_positions(price)
    
    async def _check_refined_signals(self, row, price, volume):
        """Check for signals using refined quality-based logic"""
        if len(self.price_history) < 25 or len(self.positions) > 0:
            return
        
        # Technical indicators
        sma_10 = np.mean(self.price_history[-10:])
        sma_20 = np.mean(self.price_history[-20:])
        ema_15 = self._calculate_ema(self.price_history, 15)
        
        # Momentum and trend
        momentum = (price - self.price_history[-5]) / self.price_history[-5] if len(self.price_history) >= 5 else 0
        trend_strength = abs(sma_10 - sma_20) / sma_20
        
        # RSI
        rsi = self.rsi_history[-1] if self.rsi_history else 50
        
        # BOS signals
        bullish_bos = bool(row.get('Bullish BOS', 0))
        bearish_bos = bool(row.get('Bearish BOS', 0))
        
        # Calculate signal quality
        signal_quality = self._calculate_signal_quality(price, volume, momentum, rsi, trend_strength)
        
        # Only take high-quality signals (score > 60)
        if signal_quality < 60:
            return
        
        # Refined signal logic
        bullish_signal = (
            (bullish_bos or (price > ema_15 and sma_10 > sma_20 and momentum > self.params['momentum_threshold'])) and
            rsi < self.params['rsi_overbought'] and
            trend_strength > 0.002
        )
        
        bearish_signal = (
            (bearish_bos or (price < ema_15 and sma_10 < sma_20 and momentum < -self.params['momentum_threshold'])) and
            rsi > self.params['rsi_oversold'] and
            trend_strength > 0.002
        )
        
        if bullish_signal:
            await self._open_position('long', price, signal_quality)
        elif bearish_signal:
            await self._open_position('short', price, signal_quality)
    
    def _calculate_ema(self, prices, period):
        """Calculate EMA"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    async def _open_position(self, side, price, signal_quality):
        """Open position with dynamic sizing based on signal quality"""
        # Adjust position size based on signal quality
        quality_multiplier = signal_quality / 100
        position_size = self.balance * self.params['position_size_pct'] * quality_multiplier
        
        # Calculate stops with improved R:R
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
            'signal_quality': signal_quality
        }
        
        self.positions.append(position)
        self.signal_quality_scores.append(signal_quality)
        
        if len(self.trades) % 100 == 0:
            print(f"🔥 {side.upper()} @ ${price:.4f} | Quality: {signal_quality:.1f} | Trade #{len(self.trades)+1}")
    
    async def _manage_positions(self, price):
        """Manage positions with refined exit logic"""
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
        """Close position with refined leverage"""
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        pnl_amount = position['size'] * pnl_pct * self.params['leverage']
        self.balance += pnl_amount
        
        trade = {
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'signal_quality': position['signal_quality'],
            'leverage_used': self.params['leverage']
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        if len(self.trades) % 50 == 0:
            print(f"💰 CLOSED #{len(self.trades)} | PnL: ${pnl_amount:.2f} | Balance: ${self.balance:.2f}")
    
    async def _report_final_results(self):
        """Report comprehensive results"""
        if not self.trades:
            print("❌ No trades executed")
            return
        
        # Calculate metrics
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
        
        # Signal quality analysis
        avg_signal_quality = np.mean(self.signal_quality_scores) if self.signal_quality_scores else 0
        winning_quality = np.mean([t['signal_quality'] for t in winning_trades]) if winning_trades else 0
        losing_quality = np.mean([t['signal_quality'] for t in losing_trades]) if losing_trades else 0
        
        print("\n" + "=" * 60)
        print("🏆 REFINED BALANCED BOT FINAL RESULTS")
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
        print("🧠 SIGNAL QUALITY ANALYSIS")
        print(f"📊 Average Signal Quality: {avg_signal_quality:.1f}/100")
        print(f"✅ Winning Trades Quality: {winning_quality:.1f}/100")
        print(f"❌ Losing Trades Quality: {losing_quality:.1f}/100")
        print("=" * 60)
        
        # Performance assessment
        if total_return > 30:
            print("🚀 EXCELLENT! Refined approach is working great!")
        elif total_return > 15:
            print("✅ GOOD! Refined parameters showing improvement!")
        elif total_return > 0:
            print("📈 POSITIVE! Moving in the right direction!")
        else:
            print("📉 Need further refinement")
        
        trade_frequency = len(self.trades) / 26874 * 100
        print(f"📊 Trade Frequency: {trade_frequency:.3f}% ({len(self.trades)} trades)")

async def main():
    """Main function"""
    print("🤖 REFINED BALANCED TRADING BOT")
    print("=" * 60)
    print("Quality-based signal filtering with improved risk management")
    print()
    
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'SOONUSDT' in f]
    
    if not csv_files:
        print("❌ No SOONUSDT CSV files found!")
        return
    
    csv_file = csv_files[0]
    print(f"📊 Using data file: {csv_file}")
    
    bot = RefinedBalancedBot(csv_file)
    await bot.run_refined_backtest()

if __name__ == "__main__":
    asyncio.run(main())