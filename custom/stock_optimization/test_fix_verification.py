#!/usr/bin/env python3
"""
Simple test to verify the evaluation fix works correctly
"""

import unittest
import sys
import os
import importlib.util
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from evaluator import load_stock_data, backtest_strategy


class TestEvaluationFix(unittest.TestCase):
    """Verify the evaluation fix works correctly"""
    
    def setUp(self):
        """Set up test data"""
        self.stock_data = load_stock_data("SPY", "5y")
        self.prices = self.stock_data['Close']
        
        # Load programs
        spec = importlib.util.spec_from_file_location("simple", "simple_buy_hold.py")
        self.simple_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.simple_program)
        
        spec = importlib.util.spec_from_file_location("evolved", 
            "openevolve_output/checkpoints/checkpoint_20/best_program.py")
        self.evolved_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.evolved_program)
    
    def test_buy_and_hold_gets_market_return(self):
        """Test that buy-and-hold now gets the correct market return"""
        # Get buy-and-hold signals
        signals = self.simple_program.run_stock_optimization(self.stock_data)
        results = backtest_strategy(self.prices, signals)
        
        # Calculate expected market return
        expected_return = (self.prices.iloc[-1] / self.prices.iloc[0]) - 1
        actual_return = results['total_return']
        
        print(f"\nBuy-and-Hold Performance:")
        print(f"  Expected market return: {expected_return:.1%}")
        print(f"  Actual calculated return: {actual_return:.1%}")
        print(f"  Difference: {abs(actual_return - expected_return):.4f}")
        
        # Should match within 0.1%
        self.assertAlmostEqual(actual_return, expected_return, places=3,
                              msg="Buy-and-hold should match market return")
    
    def test_buy_and_hold_outperforms_evolved(self):
        """Test that buy-and-hold now correctly outperforms the evolved program"""
        # Get results for both
        simple_signals = self.simple_program.run_stock_optimization(self.stock_data)
        simple_results = backtest_strategy(self.prices, simple_signals)
        
        evolved_signals = self.evolved_program.run_stock_optimization(self.stock_data)
        evolved_results = backtest_strategy(self.prices, evolved_signals)
        
        simple_return = simple_results['total_return']
        evolved_return = evolved_results['total_return']
        
        print(f"\nPerformance Comparison:")
        print(f"  Buy-and-hold: {simple_return:.1%}")
        print(f"  Evolved program: {evolved_return:.1%}")
        print(f"  Buy-and-hold advantage: {(simple_return - evolved_return):.1%}")
        
        self.assertGreater(simple_return, evolved_return,
                          "Buy-and-hold should outperform evolved program")
        
        # Should be a significant difference (>50%)
        self.assertGreater(simple_return - evolved_return, 0.5,
                          "Buy-and-hold should significantly outperform evolved program")


if __name__ == "__main__":
    unittest.main(verbosity=2)