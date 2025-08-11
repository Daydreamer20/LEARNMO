#!/usr/bin/env python3
"""
Scrape all available perpetual trading symbols from Bybit
"""

import requests
import json
from datetime import datetime

def get_all_perpetual_symbols():
    """Get all available perpetual trading symbols from Bybit"""
    print("🔍 Scraping all Bybit perpetual trading symbols...")
    
    try:
        # Get all linear perpetual instruments
        response = requests.get(
            "https://api.bybit.com/v5/market/instruments-info",
            params={"category": "linear"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data['retCode'] == 0:
                instruments = data['result']['list']
                
                # Filter and organize symbols
                perpetual_symbols = []
                usdt_symbols = []
                usdc_symbols = []
                other_symbols = []
                
                for instrument in instruments:
                    symbol = instrument['symbol']
                    status = instrument['status']
                    
                    # Only include trading symbols
                    if status == 'Trading':
                        perpetual_symbols.append({
                            'symbol': symbol,
                            'baseCoin': instrument['baseCoin'],
                            'quoteCoin': instrument['quoteCoin'],
                            'minOrderQty': instrument['lotSizeFilter']['minOrderQty'],
                            'maxOrderQty': instrument['lotSizeFilter']['maxOrderQty'],
                            'maxLeverage': instrument['leverageFilter']['maxLeverage'],
                            'tickSize': instrument['priceFilter']['tickSize']
                        })
                        
                        # Categorize by quote currency
                        if symbol.endswith('USDT'):
                            usdt_symbols.append(symbol)
                        elif symbol.endswith('USDC'):
                            usdc_symbols.append(symbol)
                        else:
                            other_symbols.append(symbol)
                
                return {
                    'all_symbols': perpetual_symbols,
                    'usdt_symbols': sorted(usdt_symbols),
                    'usdc_symbols': sorted(usdc_symbols),
                    'other_symbols': sorted(other_symbols),
                    'total_count': len(perpetual_symbols)
                }
            else:
                print(f"❌ API Error: {data['retMsg']}")
                return None
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error scraping symbols: {e}")
        return None

def save_symbols_to_file(symbols_data):
    """Save symbols data to JSON file"""
    if not symbols_data:
        return False
    
    try:
        # Create comprehensive symbols file
        output_data = {
            'last_updated': datetime.now().isoformat(),
            'total_symbols': symbols_data['total_count'],
            'usdt_count': len(symbols_data['usdt_symbols']),
            'usdc_count': len(symbols_data['usdc_symbols']),
            'other_count': len(symbols_data['other_symbols']),
            'symbols': {
                'usdt_perpetuals': symbols_data['usdt_symbols'],
                'usdc_perpetuals': symbols_data['usdc_symbols'],
                'other_perpetuals': symbols_data['other_symbols']
            },
            'detailed_info': symbols_data['all_symbols']
        }
        
        with open('bybit_symbols.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Symbols saved to bybit_symbols.json")
        return True
        
    except Exception as e:
        print(f"❌ Error saving symbols: {e}")
        return False

def create_symbols_module():
    """Create a Python module with symbols for easy import"""
    try:
        with open('bybit_symbols.json', 'r') as f:
            data = json.load(f)
        
        # Create Python module
        module_content = f'''"""
Bybit Perpetual Trading Symbols
Auto-generated on {data['last_updated']}
Total symbols: {data['total_symbols']}
"""

# USDT Perpetuals ({data['usdt_count']} symbols)
USDT_PERPETUALS = {data['symbols']['usdt_perpetuals']}

# USDC Perpetuals ({data['usdc_count']} symbols)
USDC_PERPETUALS = {data['symbols']['usdc_perpetuals']}

# Other Perpetuals ({data['other_count']} symbols)
OTHER_PERPETUALS = {data['symbols']['other_perpetuals']}

# All perpetual symbols
ALL_PERPETUALS = USDT_PERPETUALS + USDC_PERPETUALS + OTHER_PERPETUALS

# Popular trading pairs for scalping
POPULAR_SCALPING_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT",
    "LINKUSDT", "LTCUSDT", "BCHUSDT", "XRPUSDT", "EOSUSDT",
    "TRXUSDT", "ETCUSDT", "XLMUSDT", "ATOMUSDT", "VETUSDT"
]

# High volume pairs (good for scalping)
HIGH_VOLUME_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT"
]

def get_symbol_info(symbol):
    """Get detailed info for a specific symbol"""
    detailed_info = {data['detailed_info']}
    
    for info in detailed_info:
        if info['symbol'] == symbol:
            return info
    return None

def is_valid_symbol(symbol):
    """Check if symbol is available for trading"""
    return symbol in ALL_PERPETUALS

def get_usdt_symbols():
    """Get all USDT perpetual symbols"""
    return USDT_PERPETUALS.copy()

def get_popular_symbols():
    """Get popular symbols for scalping"""
    return POPULAR_SCALPING_PAIRS.copy()
'''
        
        with open('symbols.py', 'w') as f:
            f.write(module_content)
        
        print("✅ Created symbols.py module")
        return True
        
    except Exception as e:
        print(f"❌ Error creating symbols module: {e}")
        return False

def display_summary(symbols_data):
    """Display summary of scraped symbols"""
    if not symbols_data:
        return
    
    print("\n" + "=" * 60)
    print("  BYBIT PERPETUAL SYMBOLS SUMMARY")
    print("=" * 60)
    
    print(f"📊 Total Symbols: {symbols_data['total_count']}")
    print(f"💰 USDT Perpetuals: {len(symbols_data['usdt_symbols'])}")
    print(f"💵 USDC Perpetuals: {len(symbols_data['usdc_symbols'])}")
    print(f"🔄 Other Perpetuals: {len(symbols_data['other_symbols'])}")
    
    print(f"\n🔍 Checking for SIRENUSDT...")
    if 'SIRENUSDT' in symbols_data['usdt_symbols']:
        print("✅ SIRENUSDT is available for trading!")
        
        # Find detailed info for SIRENUSDT
        for symbol_info in symbols_data['all_symbols']:
            if symbol_info['symbol'] == 'SIRENUSDT':
                print(f"   Min Order Qty: {symbol_info['minOrderQty']}")
                print(f"   Max Leverage: {symbol_info['maxLeverage']}x")
                print(f"   Tick Size: {symbol_info['tickSize']}")
                break
    else:
        print("❌ SIRENUSDT not found in available symbols")
    
    print(f"\n📈 Popular USDT Pairs (first 20):")
    for i, symbol in enumerate(symbols_data['usdt_symbols'][:20]):
        print(f"   {i+1:2d}. {symbol}")
    
    if len(symbols_data['usdt_symbols']) > 20:
        print(f"   ... and {len(symbols_data['usdt_symbols']) - 20} more")

if __name__ == "__main__":
    print("=" * 60)
    print("  BYBIT PERPETUAL SYMBOLS SCRAPER")
    print("=" * 60)
    
    # Scrape symbols
    symbols_data = get_all_perpetual_symbols()
    
    if symbols_data:
        # Display summary
        display_summary(symbols_data)
        
        # Save to files
        if save_symbols_to_file(symbols_data):
            create_symbols_module()
            
            print(f"\n✅ Successfully scraped {symbols_data['total_count']} symbols!")
            print("📁 Files created:")
            print("   - bybit_symbols.json (detailed data)")
            print("   - symbols.py (Python module)")
            
        else:
            print("\n❌ Failed to save symbols data")
    else:
        print("\n❌ Failed to scrape symbols")
    
    print("\n" + "=" * 60)