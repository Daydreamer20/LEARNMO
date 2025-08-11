#!/usr/bin/env python3
"""
DOGE & XRP Training Bot with WANDB Integration
Comprehensive backtesting and parameter optimization for both symbols
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
import os
from typing import Dict, List, Optional
import wandb
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MultiSymbolParams:
    """Parameters for multi-symbol trading"""
    leverage: float = 15.0
    position_size_pct: float = 0.15
    stop_loss_pct: float = 0.020
    take_profit_pct: float = 0.035
    momentum_threshold: float = 0.002
    volume_multiplier: float = 1.4
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0
    max_positions_per_symbol: int = 1
    max_total_positions: int = 2
    min_balance: float = 100.0
    
    def to_dict(self):
        return asdict(self)

class DogeXrpTrainer:
    """Training bot for DOGE and XRP with comprehensive analysis"""
    
    def __init__(self, initial_balance=1000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # Separate positions for each symbol
        self.trades = []
        
        # Parameters
        self.params = MultiSymbolParams()
        
        # Symbols and data files
        self.symbols = ["DOGEUSDT", "XRPUSDT"]
        self.data_files = {
            "DOGEUSDT": "BYBIT_DOGEUSDT.P, 3_56d4e.csv",
            "XRPUSDT": "BYBIT_XRPUSDT.P, 3_0564b.csv"
        }
        
        # Market data storage per symbol
        self.market_data = {}
        for symbol in self.symbols:
            self.market_data[symbol] = {
                'price_history': [],
                'volume_history': [],
                'rsi_history': [],
                'current_price': 0.0,
                'data_length': 0
            }
        
        # Initialize positions dict
        for symbol in self.symbols:
            self.positions[symbol] = []
        
        # Performance tracking
        self.symbol_performance = {symbol: {'trades': [], 'pnl': 0.0} for symbol in self.symbols}
        
        # WANDB setup
        self._init_wandb()
        
        logger.info("🚀 DOGE & XRP Training Bot Initialized")
        logger.info(f"💰 Starting Balance: ${self.balance:.2f}")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"⚡ Parameters: {self.params.to_dict()}")
    
    def _init_wandb(self):
        """Initialize WANDB for experiment tracking"""
        try:
            wandb.init(
                project="doge-xrp-training",
                name=f"multi-symbol-{datetime.now().strftime('%Y%m%d-%H%M')}",
                config=self.params.to_dict(),
                tags=["DOGE", "XRP", "multi-symbol", "training", "15x-leverage"]
            )
            logger.info("✅ WANDB initialized for DOGE & XRP training")
        except Exception as e:
            logger.warning(f"⚠️ WANDB initialization failed: {e}")
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load and prepare data for symbol"""
        try:
            file_path = self.data_files[symbol]
            df = pd.read_csv(file_path)
            
            logger.info(f"📊 Raw data for {symbol}: {len(df)} rows")
            logger.info(f"📊 Columns: {list(df.columns)}")
            
            # Handle different timestamp formats
            if 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                # Create synthetic timestamps
                df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='3min')
            
            # Ensure we have OHLCV data
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"❌ Missing required column: {col}")
                    return None
            
            # Add volume if missing
            if 'volume' not in df.columns:
                # Generate synthetic volume based on price volatility
                price_change = abs(df['close'] - df['open']) / df['open']
                df['volume'] = price_change * np.random.uniform(500000, 2000000, len(df))
                logger.info(f"📈 Generated synthetic volume for {symbol}")
            
            # Add BOS signals if missing (simplified)
            if 'Bullish BOS' not in df.columns:
                # Simple BOS detection based on price breakouts
                df['sma_20'] = df['close'].rolling(20).mean()
                df['Bullish BOS'] = (df['close'] > df['sma_20']) & (df['close'].shift(1) <= df['sma_20'].shift(1))
                df['Bearish BOS'] = (df['close'] < df['sma_20']) & (df['close'].shift(1) >= df['sma_20'].shift(1))
                logger.info(f"📊 Generated BOS signals for {symbol}")
            
            # Clean data
            df = df.dropna()
            
            logger.info(f"✅ Loaded {len(df)} clean candles for {symbol}")
            logger.info(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
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
        return 100 - (100 / (1 + rs))
    
    async def run_training(self):
        """Run comprehensive training on both symbols"""
        logger.info("🚀 Starting DOGE & XRP Training")
        logger.info("=" * 70)
        
        # Load data for both symbols
        data = {}
        for symbol in self.symbols:
            df = self.load_data(symbol)
            if df is None:
                logger.error(f"❌ Failed to load data for {symbol}")
                return
            data[symbol] = df
            self.market_data[symbol]['data_length'] = len(df)
        
        # Find common time range (use shorter dataset)
        min_length = min(len(data[symbol]) for symbol in self.symbols)
        logger.info(f"📊 Processing {min_length} candles for each symbol")
        
        # Process candles simultaneously for both symbols
        start_time = time.time()
        
        for i in range(min_length):
            # Process each symbol
            for symbol in self.symbols:
                row = data[symbol].iloc[i]
                await self._process_candle(symbol, row)
            
            # Log progress every 2000 candles
            if i % 2000 == 0 and i > 0:
                await self._report_progress(i, min_length)
            
            # Emergency stop check
            if self.balance < self.params.min_balance:
                logger.error(f"🛑 EMERGENCY STOP: Balance below ${self.params.min_balance}")
                break
            
            # Small delay to prevent overwhelming
            if i % 500 == 0:
                await asyncio.sleep(0.001)
        
        # Calculate training duration
        training_duration = time.time() - start_time
        
        # Final results
        await self._report_final_results(training_duration)
    
    async def _process_candle(self, symbol: str, row):
        """Process individual candle for training"""
        price = float(row['close'])
        volume = float(row.get('volume', 1000))
        
        # Update market data
        self.market_data[symbol]['price_history'].append(price)
        self.market_data[symbol]['volume_history'].append(volume)
        self.market_data[symbol]['current_price'] = price
        
        # Keep history manageable (last 200 candles)
        if len(self.market_data[symbol]['price_history']) > 200:
            self.market_data[symbol]['price_history'] = self.market_data[symbol]['price_history'][-200:]
            self.market_data[symbol]['volume_history'] = self.market_data[symbol]['volume_history'][-200:]
        
        # Calculate RSI
        if len(self.market_data[symbol]['price_history']) >= 15:
            rsi = self._calculate_rsi(self.market_data[symbol]['price_history'])
            self.market_data[symbol]['rsi_history'].append(rsi)
            if len(self.market_data[symbol]['rsi_history']) > 200:
                self.market_data[symbol]['rsi_history'] = self.market_data[symbol]['rsi_history'][-200:]
        
        # Check for trading signals
        if len(self.market_data[symbol]['price_history']) >= 25:
            await self._check_trading_signals(symbol, row)
        
        # Manage existing positions
        await self._manage_positions(symbol, price)
    
    async def _check_trading_signals(self, symbol: str, row):
        """Check for trading signals with proven logic"""
        # Skip if already have position for this symbol
        if len(self.positions[symbol]) >= self.params.max_positions_per_symbol:
            return
        
        # Skip if total positions at max
        total_positions = sum(len(positions) for positions in self.positions.values())
        if total_positions >= self.params.max_total_positions:
            return
        
        price = float(row['close'])
        volume = float(row.get('volume', 1000))
        
        # Technical indicators
        prices = self.market_data[symbol]['price_history']
        volumes = self.market_data[symbol]['volume_history']
        
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        
        # Momentum calculation
        momentum = 0
        if len(prices) >= 5:
            momentum = (price - prices[-5]) / prices[-5]
        
        # Volume confirmation
        avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else volume
        volume_spike = volume > avg_volume * self.params.volume_multiplier
        
        # RSI filter
        rsi = self.market_data[symbol]['rsi_history'][-1] if self.market_data[symbol]['rsi_history'] else 50
        rsi_ok = self.params.rsi_lower < rsi < self.params.rsi_upper
        
        # BOS signals from data
        bullish_bos = bool(row.get('Bullish BOS', 0))
        bearish_bos = bool(row.get('Bearish BOS', 0))
        
        # Proven signal logic (from balanced bot)
        bullish_signal = (
            (bullish_bos or (price > sma_10 and sma_10 > sma_20 and momentum > self.params.momentum_threshold)) and
            (volume_spike or abs(momentum) > self.params.momentum_threshold * 2) and
            rsi_ok
        )
        
        bearish_signal = (
            (bearish_bos or (price < sma_10 and sma_10 < sma_20 and momentum < -self.params.momentum_threshold)) and
            (volume_spike or abs(momentum) > self.params.momentum_threshold * 2) and
            rsi_ok
        )
        
        if bullish_signal:
            await self._open_position(symbol, 'long', price, momentum, rsi, volume_spike)
        elif bearish_signal:
            await self._open_position(symbol, 'short', price, momentum, rsi, volume_spike)
    
    async def _open_position(self, symbol: str, side: str, price: float, momentum: float, rsi: float, volume_spike: bool):
        """Open position with proven parameters"""
        # Calculate position size
        position_size = self.balance * self.params.position_size_pct
        
        # Calculate stop loss and take profit
        if side == 'long':
            stop_loss = price * (1 - self.params.stop_loss_pct)
            take_profit = price * (1 + self.params.take_profit_pct)
        else:
            stop_loss = price * (1 + self.params.stop_loss_pct)
            take_profit = price * (1 - self.params.take_profit_pct)
        
        position = {
            'id': f"{symbol}_{side}_{len(self.trades)}_{int(time.time())}",
            'symbol': symbol,
            'side': side,
            'entry_price': price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'leverage': self.params.leverage,
            'entry_momentum': momentum,
            'entry_rsi': rsi,
            'volume_spike': volume_spike
        }
        
        self.positions[symbol].append(position)
        
        logger.info(f"🔥 OPENED {side.upper()} {symbol} @ ${price:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
    
    async def _manage_positions(self, symbol: str, current_price: float):
        """Manage positions with proven logic"""
        positions_to_close = []
        
        for position in self.positions[symbol]:
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
                positions_to_close.append((position, current_price, exit_reason))
        
        # Close positions
        for position, exit_price, exit_reason in positions_to_close:
            await self._close_position(position, exit_price, exit_reason)
    
    async def _close_position(self, position, exit_price: float, exit_reason: str):
        """Close position and record trade"""
        symbol = position['symbol']
        
        # Calculate PnL
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        # Apply leverage
        pnl_amount = position['position_size'] * pnl_pct * position['leverage']
        self.balance += pnl_amount
        
        # Calculate trade duration
        duration = (datetime.now() - position['entry_time']).total_seconds() / 60
        
        # Record trade
        trade = {
            'symbol': symbol,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct * 100,
            'leveraged_return': pnl_pct * position['leverage'] * 100,
            'exit_reason': exit_reason,
            'duration_minutes': duration,
            'leverage': position['leverage'],
            'entry_momentum': position['entry_momentum'],
            'entry_rsi': position['entry_rsi'],
            'volume_spike': position['volume_spike'],
            'entry_time': position['entry_time'],
            'exit_time': datetime.now()
        }
        
        self.trades.append(trade)
        self.symbol_performance[symbol]['trades'].append(trade)
        self.symbol_performance[symbol]['pnl'] += pnl_amount
        self.positions[symbol].remove(position)
        
        logger.info(f"💰 CLOSED {position['side'].upper()} {symbol}: ${pnl_amount:+.2f} ({exit_reason}) | Balance: ${self.balance:.2f}")
        
        # Log to WANDB
        if wandb.run:
            wandb.log({
                f'{symbol}_trade_pnl': pnl_amount,
                f'{symbol}_leveraged_return': pnl_pct * position['leverage'] * 100,
                'total_balance': self.balance,
                'total_trades': len(self.trades),
                'exit_reason': exit_reason,
                f'{symbol}_trades_count': len(self.symbol_performance[symbol]['trades'])
            })
    
    async def _report_progress(self, current: int, total: int):
        """Report training progress"""
        progress = (current / total) * 100
        total_positions = sum(len(positions) for positions in self.positions.values())
        
        # Calculate recent performance
        if self.trades:
            recent_trades = self.trades[-20:]
            recent_pnl = sum(t['pnl_amount'] for t in recent_trades)
            win_rate = len([t for t in recent_trades if t['pnl_amount'] > 0]) / len(recent_trades) * 100
        else:
            recent_pnl = 0
            win_rate = 0
        
        # Symbol-specific stats
        doge_trades = len(self.symbol_performance['DOGEUSDT']['trades'])
        xrp_trades = len(self.symbol_performance['XRPUSDT']['trades'])
        
        logger.info(f"📊 Progress: {progress:.1f}% | Balance: ${self.balance:.2f}")
        logger.info(f"   🎯 Total Trades: {len(self.trades)} | DOGE: {doge_trades} | XRP: {xrp_trades}")
        logger.info(f"   💰 Recent 20 PnL: ${recent_pnl:+.2f} | Win Rate: {win_rate:.1f}%")
        logger.info(f"   🔄 Open Positions: {total_positions}")
    
    async def _report_final_results(self, training_duration: float):
        """Generate comprehensive final results"""
        if not self.trades:
            logger.info("❌ No trades executed during training")
            return
        
        # Separate results by symbol
        doge_trades = self.symbol_performance['DOGEUSDT']['trades']
        xrp_trades = self.symbol_performance['XRPUSDT']['trades']
        
        # Overall metrics
        winning_trades = [t for t in self.trades if t['pnl_amount'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_amount'] < 0]
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        win_rate = len(winning_trades) / len(self.trades) * 100
        avg_win = np.mean([t['pnl_amount'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum([t['pnl_amount'] for t in winning_trades]) / sum([t['pnl_amount'] for t in losing_trades])) if losing_trades else float('inf')
        
        # Symbol-specific metrics
        doge_pnl = self.symbol_performance['DOGEUSDT']['pnl']
        xrp_pnl = self.symbol_performance['XRPUSDT']['pnl']
        
        doge_win_rate = len([t for t in doge_trades if t['pnl_amount'] > 0]) / len(doge_trades) * 100 if doge_trades else 0
        xrp_win_rate = len([t for t in xrp_trades if t['pnl_amount'] > 0]) / len(xrp_trades) * 100 if xrp_trades else 0
        
        # Risk metrics
        returns = [t['pnl_amount'] for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum([t['pnl_amount'] for t in self.trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Average trade duration
        avg_duration = np.mean([t['duration_minutes'] for t in self.trades])
        
        print("\n" + "=" * 80)
        print("🏆 DOGE & XRP TRAINING RESULTS")
        print("=" * 80)
        print(f"⏱️ Training Duration: {training_duration/60:.1f} minutes")
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
        print(f"⏱️ Avg Trade Duration: {avg_duration:.1f} minutes")
        print("=" * 80)
        print("📊 SYMBOL-SPECIFIC RESULTS")
        print(f"🐕 DOGE: {len(doge_trades)} trades | ${doge_pnl:+.2f} PnL | {doge_win_rate:.1f}% win rate")
        print(f"💎 XRP:  {len(xrp_trades)} trades | ${xrp_pnl:+.2f} PnL | {xrp_win_rate:.1f}% win rate")
        print("=" * 80)
        print("🎛️ TRAINING PARAMETERS")
        print(f"⚡ Leverage: {self.params.leverage}x")
        print(f"📏 Position Size: {self.params.position_size_pct*100:.1f}%")
        print(f"🛑 Stop Loss: {self.params.stop_loss_pct*100:.1f}%")
        print(f"🎯 Take Profit: {self.params.take_profit_pct*100:.1f}%")
        print(f"📊 Momentum Threshold: {self.params.momentum_threshold:.4f}")
        print(f"📈 Volume Multiplier: {self.params.volume_multiplier:.1f}x")
        print("=" * 80)
        
        # Performance assessment
        if total_return > 30:
            print("🚀 OUTSTANDING! Multi-symbol strategy is highly profitable!")
        elif total_return > 15:
            print("✅ EXCELLENT! DOGE & XRP strategy working very well!")
        elif total_return > 5:
            print("📈 GOOD! Solid performance on both symbols!")
        elif total_return > 0:
            print("📊 POSITIVE! Strategy is profitable!")
        else:
            print("📉 Mixed results - consider parameter adjustment")
        
        # Symbol performance comparison
        if doge_pnl > xrp_pnl:
            print(f"🐕 DOGE outperformed XRP by ${doge_pnl - xrp_pnl:.2f}")
        elif xrp_pnl > doge_pnl:
            print(f"💎 XRP outperformed DOGE by ${xrp_pnl - doge_pnl:.2f}")
        else:
            print("⚖️ Both symbols performed similarly")
        
        # Log comprehensive results to WANDB
        if wandb.run:
            wandb.log({
                'final_balance': self.balance,
                'total_return': total_return,
                'total_trades': len(self.trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_trade_duration': avg_duration,
                'training_duration_minutes': training_duration / 60,
                'doge_trades': len(doge_trades),
                'xrp_trades': len(xrp_trades),
                'doge_pnl': doge_pnl,
                'xrp_pnl': xrp_pnl,
                'doge_win_rate': doge_win_rate,
                'xrp_win_rate': xrp_win_rate
            })
        
        # Save results to file
        results = {
            'summary': {
                'final_balance': self.balance,
                'total_return': total_return,
                'total_trades': len(self.trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'training_duration_minutes': training_duration / 60
            },
            'symbol_performance': {
                'DOGE': {
                    'trades': len(doge_trades),
                    'pnl': doge_pnl,
                    'win_rate': doge_win_rate
                },
                'XRP': {
                    'trades': len(xrp_trades),
                    'pnl': xrp_pnl,
                    'win_rate': xrp_win_rate
                }
            },
            'parameters': self.params.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"doge_xrp_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {filename}")

async def main():
    """Main function to run DOGE & XRP training"""
    print("🚀 DOGE & XRP TRAINING BOT")
    print("=" * 80)
    print("Multi-symbol backtesting with WANDB integration")
    print("Using proven parameters from SOONUSDT optimization")
    print()
    
    # Initialize trainer
    trainer = DogeXrpTrainer(initial_balance=1000.0)
    
    # Run training
    await trainer.run_training()

if __name__ == "__main__":
    asyncio.run(main())