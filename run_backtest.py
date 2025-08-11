#!/usr/bin/env python3
"""
Simple script to run backtests and optimization
"""

import os
import pandas as pd
import logging
from backtester import TradingBacktester
from optimizer import ParameterOptimizer

def main():
    print("=" * 80)
    print("TRADING BOT BACKTESTING & OPTIMIZATION SUITE")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # List available data files
    data_files = [f for f in os.listdir('.') if f.endswith(('.csv', '.json', '.parquet'))]
    
    if not data_files:
        print("❌ No data files found in current directory.")
        print("Please add your market data files (.csv, .json, or .parquet)")
        print("\nExpected format:")
        print("- timestamp, open, high, low, close, volume")
        print("- OR any similar OHLCV format")
        return
    
    print(f"📁 Found {len(data_files)} data files:")
    for i, file in enumerate(data_files, 1):
        print(f"  {i}. {file}")
    
    # Get user choice
    print("\nWhat would you like to do?")
    print("1. Run simple backtest")
    print("2. Run parameter optimization")
    print("3. Load and inspect data")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_simple_backtest(data_files)
    elif choice == "2":
        run_optimization(data_files)
    elif choice == "3":
        inspect_data(data_files)
    else:
        print("Invalid choice. Exiting.")

def run_simple_backtest(data_files):
    """Run a simple backtest"""
    print("\n" + "=" * 50)
    print("SIMPLE BACKTEST")
    print("=" * 50)
    
    # Select data file
    file_choice = select_data_file(data_files)
    if not file_choice:
        return
    
    try:
        # Initialize backtester
        backtester = TradingBacktester(initial_balance=1000.0)
        
        # Load data
        print(f"📊 Loading data from {file_choice}...")
        data = backtester.load_data(file_choice, "SIRENUSDT")
        
        if data.empty:
            print("❌ Failed to load data")
            return
        
        print(f"✅ Loaded {len(data)} data points")
        print(f"📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        # Run backtest
        print("\n🚀 Running backtest...")
        results = backtester.run_backtest(data, strategy='technical')
        
        # Generate report
        report = backtester.generate_report(results, 'backtest_report.txt')
        
        print("\n✅ Backtest completed!")
        print("📄 Detailed report saved to 'backtest_report.txt'")
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        logging.error(f"Backtest error: {e}", exc_info=True)

def run_optimization(data_files):
    """Run parameter optimization"""
    print("\n" + "=" * 50)
    print("PARAMETER OPTIMIZATION")
    print("=" * 50)
    
    # Select data file
    file_choice = select_data_file(data_files)
    if not file_choice:
        return
    
    try:
        # Initialize optimizer
        print(f"📊 Initializing optimizer with {file_choice}...")
        optimizer = ParameterOptimizer(file_choice, "SIRENUSDT")
        
        # Get optimization preferences
        print("\nOptimization settings:")
        print("1. Quick optimization (fewer parameters)")
        print("2. Comprehensive optimization (more parameters, slower)")
        
        opt_choice = input("Choose optimization type (1-2): ").strip()
        
        if opt_choice == "1":
            # Quick optimization
            print("\n🚀 Running quick optimization...")
            results = optimizer.optimize_technical_strategy(
                rsi_periods=[14, 21],
                rsi_oversold=[25, 30],
                rsi_overbought=[70, 75],
                ema_fast=[9, 12],
                ema_slow=[21, 26],
                bb_periods=[20],
                bb_std=[2.0],
                confidence_threshold=[80, 85]
            )
        else:
            # Comprehensive optimization
            print("\n🚀 Running comprehensive optimization (this may take a while)...")
            results = optimizer.optimize_technical_strategy()
        
        # Analyze results
        top_results = optimizer.analyze_results(results)
        
        # Save results
        optimizer.save_results(results)
        
        # Create visualizations
        try:
            optimizer.create_visualizations(results)
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")
        
        print("\n✅ Optimization completed!")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        logging.error(f"Optimization error: {e}", exc_info=True)

def inspect_data(data_files):
    """Inspect data file structure"""
    print("\n" + "=" * 50)
    print("DATA INSPECTION")
    print("=" * 50)
    
    # Select data file
    file_choice = select_data_file(data_files)
    if not file_choice:
        return
    
    try:
        # Load data
        print(f"📊 Loading {file_choice}...")
        
        if file_choice.endswith('.csv'):
            df = pd.read_csv(file_choice)
        elif file_choice.endswith('.json'):
            df = pd.read_json(file_choice)
        elif file_choice.endswith('.parquet'):
            df = pd.read_parquet(file_choice)
        
        print(f"\n📈 Data Overview:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        print(f"\n📋 Column Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\n🔍 First 5 rows:")
        print(df.head())
        
        print(f"\n📊 Data Types:")
        print(df.dtypes)
        
        print(f"\n📈 Numeric Summary:")
        print(df.describe())
        
        # Check for timestamp column
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            print(f"\n⏰ Potential timestamp columns: {timestamp_cols}")
            
            for col in timestamp_cols:
                try:
                    ts = pd.to_datetime(df[col])
                    print(f"  {col}: {ts.min()} to {ts.max()}")
                except:
                    print(f"  {col}: Could not parse as timestamp")
        
        # Check for OHLCV columns
        ohlcv_mapping = {
            'open': [col for col in df.columns if 'open' in col.lower()],
            'high': [col for col in df.columns if 'high' in col.lower()],
            'low': [col for col in df.columns if 'low' in col.lower()],
            'close': [col for col in df.columns if 'close' in col.lower()],
            'volume': [col for col in df.columns if 'vol' in col.lower()]
        }
        
        print(f"\n💹 OHLCV Column Detection:")
        for ohlcv_type, candidates in ohlcv_mapping.items():
            if candidates:
                print(f"  {ohlcv_type.upper()}: {candidates}")
            else:
                print(f"  {ohlcv_type.upper()}: Not found")
        
        print(f"\n✅ Data inspection completed!")
        
    except Exception as e:
        print(f"❌ Error inspecting data: {e}")
        logging.error(f"Data inspection error: {e}", exc_info=True)

def select_data_file(data_files):
    """Helper to select data file"""
    if len(data_files) == 1:
        return data_files[0]
    
    print(f"\nSelect data file:")
    for i, file in enumerate(data_files, 1):
        print(f"  {i}. {file}")
    
    try:
        choice = int(input(f"Enter choice (1-{len(data_files)}): ").strip())
        if 1 <= choice <= len(data_files):
            return data_files[choice - 1]
        else:
            print("Invalid choice.")
            return None
    except ValueError:
        print("Invalid input.")
        return None

if __name__ == "__main__":
    main()