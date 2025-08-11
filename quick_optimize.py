#!/usr/bin/env python3
"""
Quick optimization script for SIRENUSDT data
"""

from optimizer import ParameterOptimizer
import logging

def main():
    print("🚀 Running SIRENUSDT Parameter Optimization...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize optimizer with your data file
        optimizer = ParameterOptimizer("BYBIT_SIRENUSDT.P, 5_e106e.csv", "SIRENUSDT")
        
        # Run quick optimization
        print("⚡ Running quick optimization...")
        results = optimizer.optimize_technical_strategy(
            rsi_periods=[10, 14, 18, 21],
            rsi_oversold=[20, 25, 30],
            rsi_overbought=[70, 75, 80],
            ema_fast=[5, 9, 12],
            ema_slow=[18, 21, 26],
            bb_periods=[15, 20],
            bb_std=[2.0, 2.5],
            confidence_threshold=[75, 80, 85]
        )
        
        # Analyze results
        print("\n📊 Analyzing results...")
        top_results = optimizer.analyze_results(results, top_n=5)
        
        # Save results
        optimizer.save_results(results, "sirenusdt_optimization.csv")
        
        print("\n✅ Optimization completed!")
        print("📁 Results saved to sirenusdt_optimization.csv")
        print("📁 Best parameters saved to best_parameters.json")
        
        # Show best parameters
        if len(results) > 0:
            best = results.iloc[0]
            print(f"\n🏆 BEST PARAMETERS FOUND:")
            print(f"   RSI: {best['rsi_period']} period, {best['rsi_oversold']}/{best['rsi_overbought']} levels")
            print(f"   EMA: {best['ema_fast']}/{best['ema_slow']} periods")
            print(f"   BB: {best['bb_period']} period, {best['bb_std']} std")
            print(f"   Confidence: {best['confidence_threshold']}%")
            print(f"   📈 Expected Return: {best['total_return']:.2f}%")
            print(f"   🎯 Win Rate: {best['win_rate']:.1f}%")
            print(f"   📉 Max Drawdown: {best['max_drawdown']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()