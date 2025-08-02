#!/usr/bin/env python3
"""
Unit tests to identify and verify the evaluation scoring bug
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import importlib.util
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from evaluator import load_stock_data, backtest_strategy, evaluate_strategy_performance


class TestEvaluationBug(unittest.TestCase):
    """Test suite to identify the evaluation scoring bug"""
    
    def setUp(self):
        """Set up test data and programs"""
        # Load real stock data for consistent testing
        self.stock_data = load_stock_data("SPY", "5y")
        self.prices = self.stock_data['Close']
        
        # Load both programs
        self.simple_program = self._load_program('simple_buy_hold.py')
        self.evolved_program = self._load_program(
            'openevolve_output/checkpoints/checkpoint_20/best_program.py'
        )
    
    def _load_program(self, program_path):
        """Helper to load a program module"""
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        return program
    
    def test_simple_buy_hold_performance(self):
        """Test simple buy-and-hold strategy performance"""
        signals = self.simple_program.run_stock_optimization(self.stock_data)
        results = backtest_strategy(self.prices, signals)
        scores = evaluate_strategy_performance(results)
        
        # Store results for comparison
        self.simple_results = results
        self.simple_scores = scores
        
        # Basic sanity checks
        self.assertIsInstance(results['total_return'], (int, float))
        self.assertIsInstance(scores['return_score'], (int, float))
        self.assertIsInstance(scores['combined_score'], (int, float))
        
        # Return should be positive for buy-and-hold over 5 years
        self.assertGreater(results['total_return'], 0, 
                          "Buy-and-hold should have positive returns over 5 years")
    
    def test_evolved_program_performance(self):
        """Test evolved program performance"""
        signals = self.evolved_program.run_stock_optimization(self.stock_data)
        results = backtest_strategy(self.prices, signals)
        scores = evaluate_strategy_performance(results)
        
        # Store results for comparison
        self.evolved_results = results
        self.evolved_scores = scores
        
        # Basic sanity checks
        self.assertIsInstance(results['total_return'], (int, float))
        self.assertIsInstance(scores['return_score'], (int, float))
        self.assertIsInstance(scores['combined_score'], (int, float))
    
    def test_performance_consistency(self):
        """Test that better performance gets better scores"""
        # Run both tests to populate results
        self.test_simple_buy_hold_performance()
        self.test_evolved_program_performance()
        
        # Compare raw performance
        simple_return = self.simple_results['total_return']
        evolved_return = self.evolved_results['total_return']
        
        # Compare scores
        simple_score = self.simple_scores['return_score']
        evolved_score = self.evolved_scores['return_score']
        
        # Print detailed comparison for debugging
        print(f"\n=== Performance Comparison ===")
        print(f"Simple Buy-and-Hold:")
        print(f"  Raw Total Return: {simple_return:.6f}")
        print(f"  Return Score: {simple_score:.6f}")
        print(f"  Combined Score: {self.simple_scores['combined_score']:.6f}")
        
        print(f"\nEvolved Program:")
        print(f"  Raw Total Return: {evolved_return:.6f}")
        print(f"  Return Score: {evolved_score:.6f}")
        print(f"  Combined Score: {self.evolved_scores['combined_score']:.6f}")
        
        print(f"\nCheckpoint 20 Reported:")
        print(f"  Raw Total Return: 0.04830623736936435")
        print(f"  Return Score: 0.01610207912312145")
        print(f"  Combined Score: 0.18734515937729695")
        print(f"  Runs Successfully: 0.0")
        
        # The core test: if one program has better raw performance, 
        # it should have a better return score
        if simple_return > evolved_return:
            self.assertGreaterEqual(simple_score, evolved_score,
                f"Simple buy-and-hold has better return ({simple_return:.6f} > {evolved_return:.6f}) "
                f"but worse score ({simple_score:.6f} < {evolved_score:.6f})")
        elif evolved_return > simple_return:
            self.assertGreaterEqual(evolved_score, simple_score,
                f"Evolved program has better return ({evolved_return:.6f} > {simple_return:.6f}) "
                f"but worse score ({evolved_score:.6f} < {simple_score:.6f})")
    
    def test_checkpoint_20_discrepancy(self):
        """Test if checkpoint 20 results match actual performance"""
        # Run evolved program
        signals = self.evolved_program.run_stock_optimization(self.stock_data)
        results = backtest_strategy(self.prices, signals)
        scores = evaluate_strategy_performance(results)
        
        # Reported checkpoint 20 metrics
        reported_return = 0.04830623736936435
        reported_return_score = 0.01610207912312145
        reported_combined_score = 0.18734515937729695
        reported_runs_successfully = 0.0
        
        # Check if actual results match reported results
        actual_return = results['total_return']
        actual_return_score = scores['return_score']
        actual_combined_score = scores['combined_score']
        
        print(f"\n=== Checkpoint 20 Verification ===")
        print(f"Reported vs Actual:")
        print(f"  Total Return: {reported_return:.6f} vs {actual_return:.6f}")
        print(f"  Return Score: {reported_return_score:.6f} vs {actual_return_score:.6f}")
        print(f"  Combined Score: {reported_combined_score:.6f} vs {actual_combined_score:.6f}")
        
        # Allow for small floating point differences (1e-6)
        self.assertAlmostEqual(actual_return, reported_return, places=6,
            msg="Actual return doesn't match reported checkpoint 20 return")
        
        # The critical bug: runs_successfully = 0.0 means the program failed,
        # but it still got scored as if it succeeded
        if reported_runs_successfully == 0.0:
            self.fail("CRITICAL BUG: Checkpoint 20 has 'runs_successfully': 0.0 but still has scores. "
                     "Failed programs should not be scored.")
    
    def test_buy_hold_signals_debug(self):
        """Debug what signals buy-and-hold is actually generating"""
        signals = self.simple_program.run_stock_optimization(self.stock_data)
        
        print(f"\n=== Buy-and-Hold Signal Debug ===")
        print(f"Signal length: {len(signals)}")
        print(f"Signal distribution: {signals.value_counts().to_dict()}")
        print(f"First 10 signals: {signals.head(10).tolist()}")
        print(f"Last 10 signals: {signals.tail(10).tolist()}")
        print(f"Signal sum: {signals.sum()}")
        
        # Check stock data period
        print(f"\nStock data period: {self.stock_data.index[0]} to {self.stock_data.index[-1]}")
        print(f"Stock data length: {len(self.stock_data)}")
        
        # Check price movement
        start_price = self.stock_data['Close'].iloc[0]
        end_price = self.stock_data['Close'].iloc[-1]
        actual_return = (end_price / start_price) - 1
        print(f"Actual SPY price movement: {start_price:.2f} -> {end_price:.2f}")
        print(f"Actual SPY return (buy-and-hold): {actual_return:.4f} ({actual_return*100:.2f}%)")
        
        # Test what backtest_strategy calculates
        results = backtest_strategy(self.prices, signals)
        print(f"Backtest calculated return: {results['total_return']:.6f}")
        
        # Manual calculation of what buy-and-hold should return
        # Buy-and-hold: buy on first day, hold forever
        daily_returns = self.prices.pct_change().dropna()
        
        # If signals are: [1, 0, 0, 0, ...] (buy first day, hold)
        # Strategy returns should be: [0, r1, r2, r3, ...] where ri are daily returns
        if signals.iloc[0] == 1 and (signals.iloc[1:] == 0).all():
            print("✓ Correct buy-and-hold signals detected")
            # This should give us the total market return
            expected_return = (1 + daily_returns).prod() - 1
            print(f"Expected buy-and-hold return: {expected_return:.6f}")
        else:
            print("❌ Incorrect buy-and-hold signals!")
        
        # Verify the fix works correctly
        print(f"\n✅ Fixed backtest matches expected return: {abs(results['total_return'] - expected_return) < 0.001}")
        
        # Show the old broken calculation for comparison
        old_strategy_returns = daily_returns * signals.shift(1)
        old_return = (1 + old_strategy_returns.dropna()).prod() - 1
        print(f"Old broken method would give: {old_return:.6f}")
        print(f"Difference from correct: {abs(old_return - expected_return):.6f}")
    
    def test_scoring_logic_sanity(self):
        """Test that the scoring logic itself makes sense"""
        # Test with known good performance
        good_results = {
            'total_return': 1.0,  # 100% return
            'annualized_volatility': 0.15,  # 15% volatility
            'sharpe_ratio': 1.5,  # Good Sharpe ratio
            'max_drawdown': -0.1  # 10% max drawdown
        }
        
        good_scores = evaluate_strategy_performance(good_results)
        
        # Test with known bad performance
        bad_results = {
            'total_return': -0.5,  # -50% return
            'annualized_volatility': 0.5,  # 50% volatility
            'sharpe_ratio': -1.0,  # Bad Sharpe ratio
            'max_drawdown': -0.4  # 40% max drawdown
        }
        
        bad_scores = evaluate_strategy_performance(bad_results)
        
        # Good performance should score better than bad performance
        self.assertGreater(good_scores['return_score'], bad_scores['return_score'])
        self.assertGreater(good_scores['combined_score'], bad_scores['combined_score'])
        
        # Negative returns should get zero return score
        self.assertEqual(bad_scores['return_score'], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)