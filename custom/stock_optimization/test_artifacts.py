"""
Unit tests for the artifacts channel in the stock optimization evaluator
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import patch

# Add the openevolve directory to the path
sys.path.insert(0, '/Users/matthew.khoriaty/Desktop/research/openevolve')

from evaluator import evaluate, evaluate_stage1, EvaluationResult, ExecutionError


class TestArtifactsChannel(unittest.TestCase):
    """Test cases for the artifacts channel functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up any temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_program(self, content):
        """Helper to create a temporary test program"""
        fd, path = tempfile.mkstemp(suffix='.py', dir=self.temp_dir)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_successful_program_artifacts(self):
        """Test that successful programs return proper artifacts"""
        program_content = '''
import pandas as pd
import numpy as np

def run_stock_optimization(stock_data):
    """Simple buy and hold strategy"""
    signals = pd.Series(1, index=stock_data.index)
    return signals
'''
        program_path = self.create_test_program(program_content)
        result = evaluate(program_path)
        
        # Check that we got a plain dictionary (for backward compatibility)
        self.assertIsInstance(result, dict)
        
        # Check that we have expected metrics
        self.assertIn('combined_score', result)
        self.assertIn('eval_time', result)
        
        # Artifacts are processed internally but not exposed in the returned dict
        # This maintains backward compatibility with OpenEvolve framework
        expected_metrics = {'combined_score', 'eval_time', 'return_score', 'volatility_score', 'sharpe_score', 'drawdown_score'}
        actual_keys = set(result.keys())
        self.assertTrue(expected_metrics.issubset(actual_keys))

    def test_syntax_error_artifacts(self):
        """Test that syntax errors are captured in artifacts"""
        program_content = '''
import pandas as pd

def run_stock_optimization(stock_data):
    # Syntax error - missing closing parenthesis
    return broken_function(
'''
        program_path = self.create_test_program(program_content)
        result = evaluate(program_path)
        
        # Should return plain dict with error metrics (artifacts processed internally)
        self.assertIsInstance(result, dict)
        
        # Should have zero scores
        self.assertEqual(result.metrics['combined_score'], 0.0)
        
        # Should have error artifacts
        self.assertTrue(result.has_artifacts())
        self.assertIn('error', result.artifacts)
        self.assertIn('error_type', result.artifacts)
        self.assertIn('failure_stage', result.artifacts)
        self.assertIn('stderr', result.artifacts)
        
        # Check error details
        self.assertEqual(result.artifacts['error_type'], 'ExecutionError')
        self.assertEqual(result.artifacts['failure_stage'], 'execution')
        self.assertIn('SyntaxError', result.artifacts['stderr'])

    def test_runtime_error_artifacts(self):
        """Test that runtime errors are captured in artifacts"""
        program_content = '''
import pandas as pd

def run_stock_optimization(stock_data):
    # Runtime error - undefined variable
    return undefined_variable
'''
        program_path = self.create_test_program(program_content)
        result = evaluate(program_path)
        
        # Should return plain dict with error metrics (artifacts processed internally)
        self.assertIsInstance(result, dict)
        
        # Should have zero scores
        self.assertEqual(result.metrics['combined_score'], 0.0)
        
        # Should have error artifacts
        self.assertTrue(result.has_artifacts())
        self.assertIn('error', result.artifacts)
        self.assertIn('traceback', result.artifacts)
        
        # Check that we captured the NameError
        self.assertIn('NameError', result.artifacts['stderr'])

    def test_invalid_return_type_artifacts(self):
        """Test that invalid return types are captured in artifacts"""
        program_content = '''
import pandas as pd

def run_stock_optimization(stock_data):
    # Invalid return type - should return pandas Series
    return [1, -1, 0, 1]
'''
        program_path = self.create_test_program(program_content)
        result = evaluate(program_path)
        
        # Should return plain dict with validation error metrics
        self.assertIsInstance(result, dict)
        
        # Should have zero scores
        self.assertEqual(result.metrics['combined_score'], 0.0)
        
        # Should have validation error artifacts
        self.assertTrue(result.has_artifacts())
        self.assertIn('error', result.artifacts)
        self.assertEqual(result.artifacts['error_type'], 'ValidationError')
        self.assertIn('Invalid signals format', result.artifacts['error'])

    def test_invalid_signal_values_artifacts(self):
        """Test that invalid signal values are captured in artifacts"""
        program_content = '''
import pandas as pd
import numpy as np

def run_stock_optimization(stock_data):
    # Invalid signal values - should be -1, 0, or 1
    signals = pd.Series([2, 3, -2, 5], index=stock_data.index[:4])
    return signals
'''
        program_path = self.create_test_program(program_content)
        result = evaluate(program_path)
        
        # Should return plain dict with validation error metrics
        self.assertIsInstance(result, dict)
        
        # Should have zero scores
        self.assertEqual(result.metrics['combined_score'], 0.0)
        
        # Should have validation error artifacts
        self.assertTrue(result.has_artifacts())
        self.assertIn('error', result.artifacts)
        self.assertEqual(result.artifacts['error_type'], 'ValidationError')
        self.assertIn('Invalid signal values', result.artifacts['error'])
        self.assertIn('invalid_signals', result.artifacts)

    def test_stage1_successful_artifacts(self):
        """Test that stage1 evaluation returns proper artifacts"""
        program_content = '''
import pandas as pd
import numpy as np

def run_stock_optimization(stock_data):
    """Simple buy and hold strategy"""
    signals = pd.Series(1, index=stock_data.index)
    return signals
'''
        program_path = self.create_test_program(program_content)
        result = evaluate_stage1(program_path)
        
        # Check that we got a plain dictionary (for backward compatibility)
        self.assertIsInstance(result, dict)
        
        # Check that we have expected metrics
        self.assertIn('runs_successfully', result.metrics)
        self.assertEqual(result.metrics['runs_successfully'], 1.0)
        
        # Check that we have artifacts
        self.assertTrue(result.has_artifacts())
        
        # Check for expected artifact keys
        expected_keys = {'stdout', 'stderr', 'signal_distribution', 'num_signals', 'test_data_length'}
        actual_keys = set(result.get_artifact_keys())
        self.assertTrue(expected_keys.issubset(actual_keys))

    def test_stage1_error_artifacts(self):
        """Test that stage1 evaluation captures error artifacts"""
        program_content = '''
import pandas as pd

def run_stock_optimization(stock_data):
    # This will cause a runtime error
    raise ValueError("Test error")
'''
        program_path = self.create_test_program(program_content)
        result = evaluate_stage1(program_path)
        
        # Should return plain dict with error metrics (artifacts processed internally)
        self.assertIsInstance(result, dict)
        
        # Should have failure
        self.assertEqual(result.metrics['runs_successfully'], 0.0)
        
        # Should have error artifacts
        self.assertTrue(result.has_artifacts())
        self.assertIn('error', result.artifacts)
        self.assertEqual(result.artifacts['failure_stage'], 'stage1_execution')

    def test_execution_timeout_artifacts(self):
        """Test that execution timeouts are captured in artifacts"""
        program_content = '''
import pandas as pd
import time

def run_stock_optimization(stock_data):
    # This will timeout
    time.sleep(100)  # Sleep longer than timeout
    signals = pd.Series(1, index=stock_data.index)
    return signals
'''
        program_path = self.create_test_program(program_content)
        
        # Use a very short timeout for testing
        with patch('evaluator.run_with_timeout') as mock_run:
            from evaluator import ExecutionError
            mock_run.side_effect = ExecutionError("Process timed out", {
                "timeout": True,
                "stderr": "Process timed out after 1 seconds",
                "stdout": "",
                "exit_code": -1
            })
            
            result = evaluate(program_path)
            
            # Should return EvaluationResult with timeout artifacts
            self.assertIsInstance(result, EvaluationResult)
            self.assertEqual(result.metrics['combined_score'], 0.0)
            self.assertTrue(result.has_artifacts())
            self.assertTrue(result.artifacts['timeout'])

    def test_artifact_size_calculation(self):
        """Test that artifact sizes are calculated correctly"""
        program_content = '''
import pandas as pd
import numpy as np

def run_stock_optimization(stock_data):
    signals = pd.Series(1, index=stock_data.index)
    return signals
'''
        program_path = self.create_test_program(program_content)
        result = evaluate(program_path)
        
        # Check artifact size methods
        self.assertGreater(result.get_total_artifact_size(), 0)
        
        for key in result.get_artifact_keys():
            size = result.get_artifact_size(key)
            self.assertGreaterEqual(size, 0)
            
        # Test non-existent key
        self.assertEqual(result.get_artifact_size('nonexistent'), 0)

    def test_backward_compatibility(self):
        """Test that the evaluator still works with legacy code expecting dict"""
        program_content = '''
import pandas as pd
import numpy as np

def run_stock_optimization(stock_data):
    signals = pd.Series(1, index=stock_data.index)
    return signals
'''
        program_path = self.create_test_program(program_content)
        result = evaluate(program_path)
        
        # Should be able to access as dict for backward compatibility
        metrics_dict = result.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('combined_score', metrics_dict)
        
        # Should also be able to access metrics directly
        self.assertIn('combined_score', result.metrics)

    def test_execution_error_class(self):
        """Test the ExecutionError class"""
        test_info = {"stdout": "test output", "stderr": "test error"}
        error = ExecutionError("Test message", test_info)
        
        self.assertEqual(str(error), "Test message")
        self.assertEqual(error.execution_info, test_info)
        
        # Test with no execution info
        error_no_info = ExecutionError("Test message")
        self.assertEqual(error_no_info.execution_info, {})


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)