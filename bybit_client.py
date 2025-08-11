from pybit.unified_trading import HTTP
from pybit.unified_trading import WebSocket
import logging
from config import Config
import time
import requests
import hmac
import hashlib
from urllib.parse import urlencode

class BybitClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://api.bybit.com"
        
        # Sync time with Bybit server first
        self._sync_time()
        
        # Initialize both manual and pybit sessions
        self.session = HTTP(
            testnet=Config.BYBIT_TESTNET,
            api_key=Config.BYBIT_API_KEY,
            api_secret=Config.BYBIT_API_SECRET,
            recv_window=60000,  # Large receive window for VPN latency
        )
        self.ws = None
        
    def _sync_time(self):
        """Sync local time with Bybit server time"""
        try:
            response = requests.get("https://api.bybit.com/v5/market/time", timeout=10)
            if response.status_code == 200:
                server_time = int(response.json()['result']['timeSecond'])
                local_time = int(time.time())
                self.time_offset = server_time - local_time
                self.logger.info(f"Time synced with Bybit. Offset: {self.time_offset}s")
            else:
                self.time_offset = 0
        except Exception as e:
            self.logger.warning(f"Time sync failed: {e}")
            self.time_offset = 0
    
    def _generate_signature(self, params, timestamp):
        """Generate signature for manual API requests"""
        param_str = urlencode(sorted(params.items()))
        sign_str = f"{timestamp}{Config.BYBIT_API_KEY}5000{param_str}"
        return hmac.new(
            Config.BYBIT_API_SECRET.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_manual_request(self, endpoint, params=None, method="GET"):
        """Make authenticated request with manual timestamp sync"""
        if params is None:
            params = {}
        
        # Use server-adjusted timestamp
        timestamp = str(int((time.time() + self.time_offset) * 1000))
        
        headers = {
            'X-BAPI-API-KEY': Config.BYBIT_API_KEY,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '5000',
            'X-BAPI-SIGN': self._generate_signature(params, timestamp),
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                if params:
                    url += "?" + urlencode(params)
                response = requests.get(url, headers=headers, timeout=30)
            else:
                response = requests.post(url, headers=headers, json=params, timeout=30)
            
            return response.json()
        except Exception as e:
            self.logger.error(f"Manual request failed: {e}")
            return None
        
    def get_account_balance(self):
        """Get USDT balance using manual sync"""
        try:
            response = self._make_manual_request("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
            if response and response['retCode'] == 0:
                # Find USDT balance
                for coin_info in response['result']['list'][0]['coin']:
                    if coin_info['coin'] == 'USDT':
                        return float(coin_info['walletBalance'])
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def get_symbol_info(self, symbol):
        """Get symbol information using manual API"""
        try:
            response = self._make_manual_request(
                "/v5/market/instruments-info",
                {
                    "category": "linear",
                    "symbol": symbol
                }
            )
            if response and response['retCode'] == 0 and response['result']['list']:
                return response['result']['list'][0]
            return None
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Get current market price using manual API"""
        try:
            response = self._make_manual_request(
                "/v5/market/tickers",
                {
                    "category": "linear",
                    "symbol": symbol
                }
            )
            if response and response['retCode'] == 0 and response['result']['list']:
                return float(response['result']['list'][0]['lastPrice'])
            return None
        except Exception as e:
            self.logger.error(f"Error getting price: {e}")
            return None
    
    def place_order(self, symbol, side, qty, price=None, order_type="Market", 
                   stop_loss=None, take_profit=None, leverage=None):
        """Place an order with optional SL/TP"""
        try:
            # Set leverage first if specified
            if leverage:
                self.set_leverage(symbol, leverage)
            
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "timeInForce": "GTC"
            }
            
            if price and order_type == "Limit":
                order_params["price"] = str(price)
            
            if stop_loss:
                order_params["stopLoss"] = str(stop_loss)
            
            if take_profit:
                order_params["takeProfit"] = str(take_profit)
            
            response = self.session.place_order(**order_params)
            
            if response['retCode'] == 0:
                self.logger.info(f"Order placed successfully: {response['result']['orderId']}")
                return response['result']
            else:
                self.logger.error(f"Order failed: {response['retMsg']}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def set_leverage(self, symbol, leverage):
        """Set leverage for symbol"""
        try:
            response = self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            return response['retCode'] == 0
        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")
            return False
    
    def get_positions(self, symbol=None):
        """Get current positions using manual API"""
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
                
            response = self._make_manual_request("/v5/position/list", params)
            if response and response['retCode'] == 0:
                return response['result']['list']
            return []
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def close_position(self, symbol, side):
        """Close position by placing opposite order"""
        try:
            positions = self.get_positions(symbol)
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['size']) > 0:
                    opposite_side = "Sell" if pos['side'] == "Buy" else "Buy"
                    return self.place_order(
                        symbol=symbol,
                        side=opposite_side,
                        qty=pos['size'],
                        order_type="Market"
                    )
            return None
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return None