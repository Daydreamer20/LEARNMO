#!/usr/bin/env python3
"""
Ultra-lightweight trading bot for Railway free plan
Minimal memory usage, optimized for 512MB RAM
"""

import asyncio
import websockets
import json
import time
import logging
import os
from datetime import datetime
from collections import deque
import requests

# Minimal logging for Railway free plan
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class LiteTradingBot:
    """Ultra-lightweight bot for Railway free plan"""
    
    def __init__(self):
        # Minimal memory footprint
        self.balance = 1000.0
        self.positions = []
        self.trades = []
        
        # Lightweight parameters
        self.leverage = 10  # Reduced for safety
        self.position_size = 0.1  # 10% position size
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.03  # 3%
        
        # Minimal data storage (only last 20 candles)
        self.prices = deque(maxlen=20)
        self.volumes = deque(maxlen=20)
        
        # Connection
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        self.symbol = "SOONUSDT"
        self.running = False
        self.last_kline_time = 0
        
        logger.info(f"🚀 Lite Bot Started | Balance: ${self.balance}")
    
    async def connect_websocket(self):
        """Lightweight WebSocket connection"""
        try:
            self.ws_connection = await asyncio.wait_for(
                websockets.connect(self.ws_url, ping_interval=30),
                timeout=10.0
            )
            
            # Subscribe to 5-minute klines (less frequent = less CPU)
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"kline.5.{self.symbol}"]
            }
            
            await self.ws_connection.send(json.dumps(subscribe_msg))
            logger.info(f"✅ Connected to {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def listen_websocket(self):
        """Lightweight message processing"""
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                if 'topic' in data and data['topic'].startswith('kline'):
                    await self._process_kline(data)
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
    
    async def _process_kline(self, data):
        """Minimal kline processing"""
        try:
            kline_data = data['data'][0]
            
            if not kline_data['confirm']:
                return  # Only process confirmed candles
            
            timestamp = int(kline_data['timestamp'])
            if timestamp == self.last_kline_time:
                return  # Skip duplicates
            
            self.last_kline_time = timestamp
            
            price = float(kline_data['close'])
            volume = float(kline_data['volume'])
            
            # Store minimal data
            self.prices.append(price)
            self.volumes.append(volume)
            
            logger.info(f"📊 {self.symbol}: ${price:.4f}")
            
            # Simple trading logic
            if len(self.prices) >= 10:
                await self._check_signals(price, volume)
            
            await self._manage_positions(price)
            
        except Exception as e:
            logger.error(f"❌ Process error: {e}")
    
    async def _check_signals(self, price, volume):
        """Ultra-simple signal detection"""
        if len(self.positions) > 0:
            return
        
        # Simple moving averages
        prices_list = list(self.prices)
        sma_5 = sum(prices_list[-5:]) / 5
        sma_10 = sum(prices_list[-10:]) / 10
        
        # Volume check
        volumes_list = list(self.volumes)
        avg_volume = sum(volumes_list[-5:]) / 5
        volume_spike = volume > avg_volume * 1.5
        
        # Simple signals
        bullish = price > sma_5 > sma_10 and volume_spike
        bearish = price < sma_5 < sma_10 and volume_spike
        
        if bullish:
            await self._open_position('long', price)
        elif bearish:
            await self._open_position('short', price)
    
    async def _open_position(self, side, price):
        """Open minimal position"""
        size = self.balance * self.position_size
        
        if side == 'long':
            stop_loss = price * (1 - self.stop_loss)
            take_profit = price * (1 + self.take_profit)
        else:
            stop_loss = price * (1 + self.stop_loss)
            take_profit = price * (1 - self.take_profit)
        
        position = {
            'side': side,
            'entry_price': price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': time.time()
        }
        
        self.positions.append(position)
        logger.info(f"🔥 {side.upper()}: ${price:.4f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
    
    async def _manage_positions(self, current_price):
        """Manage positions"""
        for position in self.positions[:]:
            should_close = False
            exit_reason = ""
            
            if position['side'] == 'long':
                if current_price <= position['stop_loss']:
                    should_close = True
                    exit_reason = "SL"
                elif current_price >= position['take_profit']:
                    should_close = True
                    exit_reason = "TP"
            else:
                if current_price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "SL"
                elif current_price <= position['take_profit']:
                    should_close = True
                    exit_reason = "TP"
            
            if should_close:
                await self._close_position(position, current_price, exit_reason)
    
    async def _close_position(self, position, exit_price, exit_reason):
        """Close position"""
        if position['side'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        pnl_amount = position['size'] * pnl_pct * self.leverage
        self.balance += pnl_amount
        
        trade = {
            'side': position['side'],
            'pnl_amount': pnl_amount,
            'exit_reason': exit_reason,
            'timestamp': time.time()
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        logger.info(f"💰 CLOSED {position['side'].upper()}: ${pnl_amount:.2f} ({exit_reason}) | Balance: ${self.balance:.2f}")
    
    async def health_server(self):
        """Minimal health server"""
        from aiohttp import web
        
        async def health(request):
            return web.json_response({
                "status": "healthy",
                "balance": round(self.balance, 2),
                "trades": len(self.trades),
                "positions": len(self.positions)
            })
        
        app = web.Application()
        app.router.add_get('/health', health)
        app.router.add_get('/', health)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        port = int(os.getenv('PORT', 8080))
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"🌐 Health server: port {port}")
    
    async def start(self):
        """Start lite bot"""
        logger.info("🚀 Starting Lite Trading Bot")
        
        self.running = True
        
        # Start health server
        asyncio.create_task(self.health_server())
        
        # Connect and trade
        if await self.connect_websocket():
            try:
                await self.listen_websocket()
            except KeyboardInterrupt:
                logger.info("⏹️ Stopping bot")
            except Exception as e:
                logger.error(f"❌ Error: {e}")
        
        # Cleanup
        if hasattr(self, 'ws_connection'):
            await self.ws_connection.close()
        
        logger.info(f"📊 Final: ${self.balance:.2f} | Trades: {len(self.trades)}")

async def main():
    """Main function for Railway"""
    bot = LiteTradingBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())