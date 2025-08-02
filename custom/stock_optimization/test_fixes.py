#!/usr/bin/env python3
"""
Unit tests for stock optimization fixes
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from initial_program import run_stock_optimization, optimize_strategy, generate_signals


class TestStockOptimizationFixes(unittest.TestCase):
    """Test the fixed stock optimization implementation"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample stock data with realistic price movements
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)  # Random walk
        
        self.stock_data = pd.DataFrame({
            'Close': prices,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        self.prices = self.stock_data['Close']
    
    def test_signal_format_constraints(self):
        """Test that signals meet format constraints"""
        signals = run_stock_optimization(self.stock_data)
        
        # Test 1: Must be pandas Series
        self.assertIsInstance(signals, pd.Series, "Signals must be pandas Series")
        
        # Test 2: Must have same length as input data
        self.assertEqual(len(signals), len(self.stock_data), 
                        f"Signal length {len(signals)} != stock data length {len(self.stock_data)}")
        
        # Test 3: Must contain only valid integer signals
        unique_signals = set(signals.unique())
        valid_signals = {-1, 0, 1}
        self.assertTrue(unique_signals.issubset(valid_signals), 
                       f"Invalid signals found: {unique_signals - valid_signals}")
        
        # Test 4: Must be integer dtype
        self.assertTrue(pd.api.types.is_integer_dtype(signals.dtype), 
                       f"Signals must be integer type, got {signals.dtype}")
    
    def test_index_alignment(self):
        """Test that signal index matches stock data index"""
        signals = run_stock_optimization(self.stock_data)
        
        # Index must match exactly
        pd.testing.assert_index_equal(signals.index, self.stock_data.index,
                                    "Signal index must match stock data index")
    
    def test_no_nan_values(self):
        """Test that signals contain no NaN values"""
        signals = run_stock_optimization(self.stock_data)
        
        self.assertFalse(signals.isna().any(), "Signals must not contain NaN values")
    
    def test_generate_signals_robustness(self):
        """Test that generate_signals handles edge cases"""
        # Test with very short series
        short_prices = self.prices.head(5)
        signals = generate_signals(short_prices, short_window=2, long_window=3)
        
        self.assertEqual(len(signals), len(short_prices))
        self.assertTrue(all(s in [-1, 0, 1] for s in signals.unique()))
        
        # Test with NaN values in input
        prices_with_nan = self.prices.copy()
        prices_with_nan.iloc[10:15] = np.nan
        signals = generate_signals(prices_with_nan)
        
        self.assertEqual(len(signals), len(prices_with_nan))
        self.assertFalse(signals.isna().any())
    
    def test_optimize_strategy_constraints(self):
        """Test that optimize_strategy meets all constraints"""
        signals = optimize_strategy(self.prices)
        
        # All constraint checks
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.prices))
        self.assertTrue(pd.api.types.is_integer_dtype(signals.dtype))
        self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))
        self.assertFalse(signals.isna().any())
        pd.testing.assert_index_equal(signals.index, self.prices.index)
    
    def test_real_market_data_simulation(self):
        """Test with realistic market scenarios"""
        # Trending up market
        trending_up = pd.Series(100 + np.cumsum(np.abs(np.random.randn(50)) * 0.3), 
                               index=pd.date_range('2023-01-01', periods=50))
        signals_up = optimize_strategy(trending_up)
        self.assertEqual(len(signals_up), 50)
        
        # Trending down market  
        trending_down = pd.Series(100 - np.cumsum(np.abs(np.random.randn(50)) * 0.3),
                                 index=pd.date_range('2023-01-01', periods=50))
        signals_down = optimize_strategy(trending_down)
        self.assertEqual(len(signals_down), 50)
        
        # Sideways market
        sideways = pd.Series(100 + np.random.randn(50) * 0.1,
                           index=pd.date_range('2023-01-01', periods=50))
        signals_sideways = optimize_strategy(sideways)
        self.assertEqual(len(signals_sideways), 50)
    
    def test_no_iloc_on_scalars(self):
        """Test that code doesn't use .iloc on numpy scalars"""
        # This should not raise AttributeError
        try:
            signals = run_stock_optimization(self.stock_data)
            self.assertTrue(True, "No .iloc on scalar errors")
        except AttributeError as e:
            if "iloc" in str(e):
                self.fail(f"Found .iloc on scalar error: {e}")
            else:
                raise
    
    def test_performance_basic(self):
        """Basic performance test - should complete quickly"""
        import time
        
        start = time.time()
        signals = run_stock_optimization(self.stock_data)
        duration = time.time() - start
        
        self.assertLess(duration, 1.0, "Basic optimization should complete in <1 second")
        self.assertEqual(len(signals), len(self.stock_data))


def run_tests():
    """Run all tests and report results"""
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestStockOptimizationFixes)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("✅ ALL TESTS PASSED - Stock optimization fixes work correctly!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Issues still exist")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)