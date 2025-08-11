#!/usr/bin/env python3
"""
Data chunking utility for backtesting with different segmentation strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Generator
import logging

class DataChunker:
    """
    Utility class for chunking trading data into segments for backtesting
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with trading data
        
        Args:
            data: DataFrame with OHLC data and signals
        """
        self.data = data.copy()
        self.data['time'] = pd.to_datetime(self.data['time'], unit='s')
        self.data = self.data.sort_values('time').reset_index(drop=True)
        
    def chunk_by_rows(self, chunk_size: int = 100, overlap: int = 0) -> Generator[pd.DataFrame, None, None]:
        """
        Split data into fixed-size chunks by number of rows
        
        Args:
            chunk_size: Number of rows per chunk
            overlap: Number of overlapping rows between chunks
            
        Yields:
            DataFrame chunks
        """
        total_rows = len(self.data)
        start_idx = 0
        
        while start_idx < total_rows:
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = self.data.iloc[start_idx:end_idx].copy()
            
            if len(chunk) > 0:
                yield chunk
                
            # Move to next chunk with overlap consideration
            start_idx = end_idx - overlap
            if start_idx >= total_rows:
                break
                
    def chunk_by_time(self, time_window: str = '1D', overlap_hours: int = 0) -> Generator[pd.DataFrame, None, None]:
        """
        Split data into time-based chunks
        
        Args:
            time_window: Time window ('1H', '4H', '1D', '1W', etc.)
            overlap_hours: Hours of overlap between chunks
            
        Yields:
            DataFrame chunks
        """
        # Convert time window to timedelta
        window_map = {
            '1H': timedelta(hours=1),
            '4H': timedelta(hours=4),
            '1D': timedelta(days=1),
            '1W': timedelta(weeks=1),
            '1M': timedelta(days=30)
        }
        
        window_delta = window_map.get(time_window, timedelta(days=1))
        overlap_delta = timedelta(hours=overlap_hours)
        
        start_time = self.data['time'].min()
        end_time = self.data['time'].max()
        
        current_start = start_time
        
        while current_start < end_time:
            current_end = current_start + window_delta
            
            # Get data for this time window
            mask = (self.data['time'] >= current_start) & (self.data['time'] < current_end)
            chunk = self.data[mask].copy()
            
            if len(chunk) > 0:
                yield chunk
                
            # Move to next window with overlap
            current_start = current_end - overlap_delta
            
    def chunk_by_signals(self, signal_columns: List[str] = None, min_chunk_size: int = 20) -> Generator[pd.DataFrame, None, None]:
        """
        Split data based on signal occurrences (BOS, CHOCH, etc.)
        
        Args:
            signal_columns: List of signal column names to trigger new chunks
            min_chunk_size: Minimum number of rows per chunk
            
        Yields:
            DataFrame chunks
        """
        if signal_columns is None:
            # Check which signal columns exist in the data
            available_signals = []
            for col in ['Bullish BOS', 'Bearish BOS', 'Bullish CHOCH', 'Bearish CHOCH']:
                if col in self.data.columns:
                    available_signals.append(col)
            signal_columns = available_signals if available_signals else ['Bullish BOS', 'Bearish BOS']
            
        # Find signal trigger points
        signal_mask = self.data[signal_columns].any(axis=1)
        signal_indices = self.data[signal_mask].index.tolist()
        
        if not signal_indices:
            # No signals found, return entire dataset
            yield self.data
            return
            
        # Add start and end indices
        chunk_starts = [0] + signal_indices
        chunk_ends = signal_indices + [len(self.data)]
        
        for start_idx, end_idx in zip(chunk_starts, chunk_ends):
            chunk = self.data.iloc[start_idx:end_idx].copy()
            
            # Only yield chunks that meet minimum size requirement
            if len(chunk) >= min_chunk_size:
                yield chunk
                
    def chunk_by_volatility(self, volatility_window: int = 20, threshold_percentile: float = 75) -> Generator[pd.DataFrame, None, None]:
        """
        Split data based on volatility regimes
        
        Args:
            volatility_window: Window for calculating volatility
            threshold_percentile: Percentile threshold for regime changes
            
        Yields:
            DataFrame chunks
        """
        # Calculate rolling volatility
        self.data['returns'] = self.data['close'].pct_change()
        self.data['volatility'] = self.data['returns'].rolling(volatility_window).std()
        
        # Determine volatility threshold
        vol_threshold = self.data['volatility'].quantile(threshold_percentile / 100)
        
        # Find regime changes
        high_vol = self.data['volatility'] > vol_threshold
        regime_changes = high_vol != high_vol.shift(1)
        change_indices = self.data[regime_changes].index.tolist()
        
        if not change_indices:
            yield self.data
            return
            
        # Create chunks based on regime changes
        chunk_starts = [0] + change_indices
        chunk_ends = change_indices + [len(self.data)]
        
        for start_idx, end_idx in zip(chunk_starts, chunk_ends):
            chunk = self.data.iloc[start_idx:end_idx].copy()
            if len(chunk) > 0:
                yield chunk
                
    def get_chunk_summary(self, chunks: List[pd.DataFrame]) -> Dict:
        """
        Get summary statistics for chunks
        
        Args:
            chunks: List of DataFrame chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        summary = {
            'total_chunks': len(chunks),
            'total_rows': sum(chunk_sizes),
            'avg_chunk_size': np.mean(chunk_sizes),
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'chunk_sizes': chunk_sizes
        }
        
        return summary

def demo_chunking():
    """
    Demonstrate different chunking strategies
    """
    # Load sample data
    try:
        data = pd.read_csv('BYBIT_SIRENUSDT.P, 5_e106e.csv')
        chunker = DataChunker(data)
        
        print("=== Data Chunking Demo ===")
        print(f"Total data points: {len(data)}")
        print(f"Time range: {data['time'].min()} to {data['time'].max()}")
        
        # Test different chunking methods
        methods = [
            ("Fixed rows (100)", lambda: list(chunker.chunk_by_rows(100))),
            ("Time-based (1D)", lambda: list(chunker.chunk_by_time('1D'))),
            ("Signal-based", lambda: list(chunker.chunk_by_signals())),
            ("Volatility-based", lambda: list(chunker.chunk_by_volatility()))
        ]
        
        for method_name, method_func in methods:
            try:
                chunks = method_func()
                summary = chunker.get_chunk_summary(chunks)
                
                print(f"\n{method_name}:")
                print(f"  Chunks created: {summary['total_chunks']}")
                print(f"  Avg chunk size: {summary['avg_chunk_size']:.1f}")
                print(f"  Size range: {summary['min_chunk_size']} - {summary['max_chunk_size']}")
                
            except Exception as e:
                print(f"  Error with {method_name}: {e}")
                
    except FileNotFoundError:
        print("Sample data file not found. Please ensure CSV file exists.")
    except Exception as e:
        print(f"Error in demo: {e}")

if __name__ == "__main__":
    demo_chunking()