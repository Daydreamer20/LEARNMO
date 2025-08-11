#!/usr/bin/env python3
"""
Aggressive backtesting with specific conditions:
- Leverage: 25x
- Margin: 5 USDT per trade
- SL: Right before liquidation
- TP: 3-4%
"""

from backtester import TradingBacktester
from optimized_bot_config import OPTIMIZED_PARAMS
import pandas as pd
import numpy as np

class AggressiveBacktester(TradingBacktester):
    def __init__(self):
        super().__init__(initial_balance=1000.0)
        self.params = OPTIMIZED_PARAMS
        
        # Aggressive trading parameters
        self.leverage = 25  # Fixed 25x leverage
        self.margin_per_trade = 5.0  # 5 USDT margin per trade
        self.tp_percent_min = 3.0  # 3% take profit
        self.tp_percent_max = 4.0  # 4% take profit
        
        # Calculate liquidation distance for 25x leverage
        # For 25x leverage: liquidation distance ≈ 1/25 = 4%
        # We'll set SL at 3.8% (0.2% before liquidation)
        self.liquidation_distance = 4.0  # 4% to liquidation
        self.sl_buffer = 0.2  # 0.2% buffer before liquidation
        self.sl_percent = self.liquidation_distance - self.sl_buffer  # 3.8%
        
        print(f"🎯 Aggressive Trading Setup:")
        print(f"   Leverage: {self.leverage}x")
        print(f"   Margin per trade: ${self.margin_per_trade}")
        print(f"   Stop Loss: {self.sl_percent}% (before liquidation)")
        print(f"   Take Profit: {self.tp_percent_min}-{self.tp_percent_max}%")
    
    def calculate_aggressive_position_size(self, current_price):
        """Calculate position size based on fixed margin"""
        # Position size = (Margin * Leverage) / Price
        position_value = self.margin_per_trade * self.leverage
        position_size = position_value / current_price
        return position_size
    
    def calculate_aggressive_sl_tp(self, entry_price, side):
        """Calculate stop loss and take profit with aggressive parameters"""
        if side == 'BUY':
            # Long position
            stop_loss = entry_price * (1 - self.sl_percent / 100)
            take_profit = entry_price * (1 + np.random.uniform(self.tp_percent_min, self.tp_percent_max) / 100)
        else:
            # Short position
            stop_loss = entry_price * (1 + self.sl_percent / 100)
            take_profit = entry_price * (1 - np.random.uniform(self.tp_percent_min, self.tp_percent_max) / 100)
        
        return stop_loss, take_profit
    
    def calculate_aggressive_pnl(self, entry_price, exit_price, side, position_size):
        """Calculate PnL for aggressive trading"""
        if side == 'BUY':
            price_change_pct = (exit_price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - exit_price) / entry_price
        
        # PnL = Margin * Leverage * Price Change %
        pnl = self.margin_per_trade * self.leverage * price_change_pct
        return pnl
    
    def run_aggressive_backtest(self, data_file):
        """Run backtest with aggressive parameters"""
        print("🚀 Running Aggressive SIRENUSDT Backtest...")
        print("=" * 60)
        
        # Load data
        data = self.load_data(data_file, "SIRENUSDT")
        if data.empty:
            print("❌ Failed to load data")
            return None
        
        print(f"📊 Loaded {len(data)} data points")
        print(f"📅 Period: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        # Calculate indicators
        print("🔧 Calculating technical indicators...")
        data = self.calculate_indicators(data)
        
        # Generate signals
        print("🎯 Generating trading signals...")
        data = self.generate_signals(data, 'technical')
        
        # Count signals
        buy_signals = (data['signal'] == 'BUY').sum()
        sell_signals = (data['signal'] == 'SELL').sum()
        total_signals = buy_signals + sell_signals
        
        print(f"📈 Generated {total_signals} signals ({buy_signals} BUY, {sell_signals} SELL)")
        
        # Run aggressive backtest simulation
        print("⚡ Running aggressive backtest simulation...")
        
        # Reset state
        self.current_balance = self.initial_balance
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        
        # Track aggressive metrics
        total_margin_used = 0
        max_margin_used = 0
        liquidations = 0
        
        # Process each data point
        for i in range(len(data)):
            row = data.iloc[i]
            self._process_aggressive_tick(row, i)
            
            # Calculate current margin usage
            current_margin = len(self.positions) * self.margin_per_trade
            total_margin_used = max(total_margin_used, current_margin)
            max_margin_used = max(max_margin_used, current_margin)
            
            # Record equity
            equity = self._calculate_aggressive_equity(row)
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'balance': self.current_balance,
                'equity': equity,
                'margin_used': current_margin
            })
        
        # Close any remaining positions
        self._close_all_positions(data.iloc[-1])
        
        # Calculate results
        results = self._calculate_results()
        
        # Add aggressive-specific metrics
        results.max_margin_used = max_margin_used
        results.liquidations = liquidations
        results.margin_efficiency = results.total_pnl / max_margin_used if max_margin_used > 0 else 0
        
        print(f"✅ Aggressive backtest completed!")
        return results, data
    
    def _process_aggressive_tick(self, row, index):
        """Process single data point with aggressive logic"""
        timestamp = row['timestamp']
        symbol = row.get('symbol', 'SIRENUSDT')
        current_price = row['close']
        
        # Check existing positions for exits
        self._check_aggressive_exits(row)
        
        # Check for new entries
        if row['signal'] in ['BUY', 'SELL'] and row['confidence'] >= 75:
            if symbol not in self.positions:  # No existing position
                # Check if we have enough balance for margin
                if self.current_balance >= self.margin_per_trade:
                    self._enter_aggressive_position(row)
    
    def _enter_aggressive_position(self, row):
        """Enter aggressive position"""
        symbol = row.get('symbol', 'SIRENUSDT')
        side = row['signal']
        current_price = row['close']
        confidence = row['confidence']
        reasoning = row['reasoning']
        
        # Calculate aggressive position parameters
        position_size = self.calculate_aggressive_position_size(current_price)
        stop_loss, take_profit = self.calculate_aggressive_sl_tp(current_price, side)
        
        # Deduct margin from balance
        self.current_balance -= self.margin_per_trade
        
        # Create position
        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': current_price,
            'entry_time': row['timestamp'],
            'position_size': position_size,
            'leverage': self.leverage,
            'margin': self.margin_per_trade,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'reasoning': reasoning
        }
        
        self.positions[symbol] = position
        
        # Calculate risk/reward ratio
        if side == 'BUY':
            risk_pct = (current_price - stop_loss) / current_price * 100
            reward_pct = (take_profit - current_price) / current_price * 100
        else:
            risk_pct = (stop_loss - current_price) / current_price * 100
            reward_pct = (current_price - take_profit) / current_price * 100
        
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
        
        self.logger.info(f"Aggressive {side} position: {symbol} @ {current_price:.5f}")
        self.logger.info(f"  Margin: ${self.margin_per_trade}, Leverage: {self.leverage}x")
        self.logger.info(f"  SL: {stop_loss:.5f} ({risk_pct:.2f}%), TP: {take_profit:.5f} ({reward_pct:.2f}%)")
        self.logger.info(f"  Risk/Reward: 1:{rr_ratio:.2f}")
    
    def _check_aggressive_exits(self, row):
        """Check if positions should be closed with aggressive logic"""
        symbol = row.get('symbol', 'SIRENUSDT')
        current_price = row['close']
        
        if symbol in self.positions:
            position = self.positions[symbol]
            exit_reason = None
            
            # Check stop loss (liquidation protection)
            if position['side'] == 'BUY' and current_price <= position['stop_loss']:
                exit_reason = 'SL'
            elif position['side'] == 'SELL' and current_price >= position['stop_loss']:
                exit_reason = 'SL'
            
            # Check take profit (3-4% target)
            elif position['side'] == 'BUY' and current_price >= position['take_profit']:
                exit_reason = 'TP'
            elif position['side'] == 'SELL' and current_price <= position['take_profit']:
                exit_reason = 'TP'
            
            # Check for strong reversal signals (early exit)
            elif row['signal'] != 'HOLD' and row['signal'] != position['side'] and row['confidence'] >= 85:
                exit_reason = 'SIGNAL'
            
            if exit_reason:
                self._exit_aggressive_position(symbol, current_price, row['timestamp'], exit_reason)
    
    def _exit_aggressive_position(self, symbol, exit_price, exit_time, exit_reason):
        """Exit aggressive position and record trade"""
        position = self.positions[symbol]
        
        # Calculate PnL using aggressive method
        pnl = self.calculate_aggressive_pnl(
            position['entry_price'], 
            exit_price, 
            position['side'], 
            position['position_size']
        )
        
        # Calculate percentage PnL
        if position['side'] == 'BUY':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
        
        # Return margin to balance and add/subtract PnL
        self.current_balance += position['margin'] + pnl
        
        # Create trade record
        from backtester import Trade
        trade = Trade(
            timestamp=position['entry_time'],
            symbol=symbol,
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['position_size'],
            leverage=position['leverage'],
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration=exit_time - position['entry_time'],
            confidence=position['confidence'],
            reasoning=position['reasoning']
        )
        
        self.trades.append(trade)
        
        # Log trade result
        self.logger.info(f"Aggressive {position['side']} exit: {symbol} @ {exit_price:.5f} ({exit_reason})")
        self.logger.info(f"  PnL: ${pnl:.2f} ({pnl_pct:.2f}%), Margin: ${position['margin']}")
        
        # Remove position
        del self.positions[symbol]
    
    def _calculate_aggressive_equity(self, row):
        """Calculate current equity including unrealized PnL"""
        equity = self.current_balance
        
        for symbol, position in self.positions.items():
            if symbol == row.get('symbol', 'SIRENUSDT'):
                current_price = row['close']
                unrealized_pnl = self.calculate_aggressive_pnl(
                    position['entry_price'],
                    current_price,
                    position['side'],
                    position['position_size']
                )
                equity += unrealized_pnl
        
        return equity
    
    def generate_aggressive_report(self, results, data):
        """Generate comprehensive aggressive trading report"""
        print("\n" + "=" * 80)
        print("AGGRESSIVE SIRENUSDT BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"📊 PERFORMANCE SUMMARY")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Winning Trades: {results.winning_trades} ({results.win_rate:.1f}%)")
        print(f"   Losing Trades: {results.losing_trades}")
        
        print(f"\n💰 PROFIT & LOSS")
        print(f"   Total PnL: ${results.total_pnl:.2f}")
        print(f"   Total Return: {results.total_pnl_pct:.2f}%")
        print(f"   Best Trade: ${results.best_trade:.2f}")
        print(f"   Worst Trade: ${results.worst_trade:.2f}")
        
        print(f"\n📈 RISK METRICS")
        print(f"   Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   Profit Factor: {results.profit_factor:.2f}")
        
        print(f"\n⚡ AGGRESSIVE METRICS")
        print(f"   Leverage Used: {self.leverage}x")
        print(f"   Margin per Trade: ${self.margin_per_trade}")
        print(f"   Max Margin Used: ${results.max_margin_used:.2f}")
        print(f"   Margin Efficiency: {results.margin_efficiency:.2f} (PnL/Margin)")
        print(f"   Stop Loss Distance: {self.sl_percent}%")
        print(f"   Take Profit Range: {self.tp_percent_min}-{self.tp_percent_max}%")
        
        # Analyze trade outcomes
        if results.trades:
            tp_trades = [t for t in results.trades if t.exit_reason == 'TP']
            sl_trades = [t for t in results.trades if t.exit_reason == 'SL']
            signal_trades = [t for t in results.trades if t.exit_reason == 'SIGNAL']
            
            print(f"\n🎯 EXIT ANALYSIS")
            print(f"   Take Profit Exits: {len(tp_trades)} ({len(tp_trades)/len(results.trades)*100:.1f}%)")
            print(f"   Stop Loss Exits: {len(sl_trades)} ({len(sl_trades)/len(results.trades)*100:.1f}%)")
            print(f"   Signal Exits: {len(signal_trades)} ({len(signal_trades)/len(results.trades)*100:.1f}%)")
            
            if tp_trades:
                avg_tp_pnl = sum(t.pnl for t in tp_trades) / len(tp_trades)
                print(f"   Avg TP Trade: ${avg_tp_pnl:.2f}")
            
            if sl_trades:
                avg_sl_pnl = sum(t.pnl for t in sl_trades) / len(sl_trades)
                print(f"   Avg SL Trade: ${avg_sl_pnl:.2f}")
        
        # Risk/Reward analysis
        print(f"\n⚖️  RISK/REWARD ANALYSIS")
        total_risk = results.total_trades * self.margin_per_trade
        print(f"   Total Risk Exposure: ${total_risk:.2f}")
        print(f"   Risk per Trade: ${self.margin_per_trade} ({self.sl_percent}% SL)")
        print(f"   Reward per Trade: {self.tp_percent_min}-{self.tp_percent_max}% TP")
        print(f"   Expected R:R Ratio: 1:{self.tp_percent_min/self.sl_percent:.2f} to 1:{self.tp_percent_max/self.sl_percent:.2f}")
        
        # Performance vs conservative
        print(f"\n📊 vs CONSERVATIVE APPROACH:")
        print(f"   Conservative: 78 trades, 50% win rate, +0.01% return")
        print(f"   Aggressive: {results.total_trades} trades, {results.win_rate:.1f}% win rate, {results.total_pnl_pct:.2f}% return")
        
        if results.total_pnl_pct > 0.01:
            improvement = results.total_pnl_pct - 0.01
            print(f"   📈 Return Improvement: +{improvement:.2f}%")
        
        return results

def main():
    """Run aggressive backtest"""
    print("=" * 80)
    print("AGGRESSIVE SIRENUSDT BACKTEST")
    print("25x Leverage | 5 USDT Margin | 3.8% SL | 3-4% TP")
    print("=" * 80)
    
    # Initialize aggressive backtester
    backtester = AggressiveBacktester()
    
    # Run aggressive backtest
    results, data = backtester.run_aggressive_backtest("BYBIT_SIRENUSDT.P, 5_e106e.csv")
    
    if results:
        # Generate comprehensive report
        backtester.generate_aggressive_report(results, data)
        
        # Save detailed report
        backtester.generate_report(results, 'aggressive_backtest_report.txt')
        print(f"\n📄 Detailed report saved to 'aggressive_backtest_report.txt'")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()