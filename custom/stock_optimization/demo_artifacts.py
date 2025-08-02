#!/usr/bin/env python3
"""
Demonstration of the artifacts channel in stock optimization evaluator

This script shows how the artifacts channel provides detailed feedback
about program execution, compilation errors, runtime errors, and performance metrics.
"""

import sys
import json
import tempfile
import os

# Add the openevolve directory to the path
sys.path.insert(0, '/Users/matthew.khoriaty/Desktop/research/openevolve')

from evaluator import evaluate, evaluate_stage1


def create_demo_program(content, name):
    """Create a temporary program file for demo"""
    fd, path = tempfile.mkstemp(suffix=f'_{name}.py')
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


def show_result_summary(result, title):
    """Display a summary of evaluation results and artifacts"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    print(f"\nResult Type: {type(result).__name__}")
    print(f"Combined Score: {result.metrics.get('combined_score', 'N/A')}")
    print(f"Evaluation Time: {result.metrics.get('eval_time', 'N/A'):.3f}s")
    
    print(f"\nHas Artifacts: {result.has_artifacts()}")
    if result.has_artifacts():
        print(f"Total Artifact Size: {result.get_total_artifact_size()} bytes")
        print(f"Artifact Keys: {', '.join(result.get_artifact_keys())}")
        
        # Show key artifacts
        key_artifacts = ['error', 'error_type', 'failure_stage', 'signal_distribution']
        for key in key_artifacts:
            if key in result.artifacts:
                value = result.artifacts[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")


def demo_successful_program():
    """Demo: Successful program with performance artifacts"""
    program = '''
import pandas as pd
import numpy as np

def run_stock_optimization(stock_data):
    """Moving average crossover strategy"""
    prices = stock_data['Close']
    short_ma = prices.rolling(window=10).mean()
    long_ma = prices.rolling(window=50).mean()
    
    signals = pd.Series(0, index=prices.index)
    signals[short_ma > long_ma] = 1
    signals[short_ma < long_ma] = -1
    
    return signals
'''
    
    path = create_demo_program(program, "successful")
    try:
        result = evaluate(path)
        show_result_summary(result, "SUCCESSFUL PROGRAM - Performance Artifacts")
        
        if result.has_artifacts():
            print(f"\nPerformance Details:")
            print(f"  Number of signals: {result.artifacts.get('num_signals', 'N/A')}")
            print(f"  Number of trades: {result.artifacts.get('num_trades', 'N/A')}")
            print(f"  Signal distribution: {result.artifacts.get('signal_distribution', 'N/A')}")
            
    finally:
        os.unlink(path)


def demo_syntax_error():
    """Demo: Syntax error captured in artifacts"""
    program = '''
import pandas as pd

def run_stock_optimization(stock_data):
    # Syntax error - missing closing parenthesis
    return some_function(
        stock_data['Close'],
        window=10
    # Missing closing parenthesis here
'''
    
    path = create_demo_program(program, "syntax_error")
    try:
        result = evaluate(path)
        show_result_summary(result, "SYNTAX ERROR - Compilation Artifacts")
        
        if 'stderr' in result.artifacts:
            print(f"\nSyntax Error Details:")
            stderr_lines = result.artifacts['stderr'].split('\n')
            for line in stderr_lines[-5:]:  # Show last 5 lines
                if line.strip():
                    print(f"  {line}")
                    
    finally:
        os.unlink(path)


def demo_runtime_error():
    """Demo: Runtime error captured in artifacts"""
    program = '''
import pandas as pd

def run_stock_optimization(stock_data):
    # Runtime error - trying to access undefined variable
    prices = stock_data['Close']
    
    # This will cause a NameError
    signals = undefined_variable * prices
    
    return signals
'''
    
    path = create_demo_program(program, "runtime_error")
    try:
        result = evaluate(path)
        show_result_summary(result, "RUNTIME ERROR - Execution Artifacts")
        
        if 'traceback' in result.artifacts:
            print(f"\nRuntime Error Traceback:")
            tb_lines = result.artifacts['traceback'].split('\n')
            for line in tb_lines[-8:]:  # Show last 8 lines
                if line.strip():
                    print(f"  {line}")
                    
    finally:
        os.unlink(path)


def demo_validation_error():
    """Demo: Validation error captured in artifacts"""
    program = '''
import pandas as pd

def run_stock_optimization(stock_data):
    # Returns wrong type - should return pandas Series
    return [1, -1, 0, 1, 1, -1]
'''
    
    path = create_demo_program(program, "validation_error")
    try:
        result = evaluate(path)
        show_result_summary(result, "VALIDATION ERROR - Format Artifacts")
        
    finally:
        os.unlink(path)


def demo_stage1_evaluation():
    """Demo: Stage 1 evaluation with artifacts"""
    program = '''
import pandas as pd
import numpy as np

def run_stock_optimization(stock_data):
    """Simple buy and hold strategy"""
    signals = pd.Series(1, index=stock_data.index)
    return signals
'''
    
    path = create_demo_program(program, "stage1")
    try:
        result = evaluate_stage1(path)
        show_result_summary(result, "STAGE 1 EVALUATION - Quick Validation")
        
        if result.has_artifacts():
            print(f"\nStage 1 Details:")
            print(f"  Test data length: {result.artifacts.get('test_data_length', 'N/A')}")
            print(f"  Runs successfully: {result.metrics.get('runs_successfully', 'N/A')}")
            
    finally:
        os.unlink(path)


def main():
    """Run all artifact demonstrations"""
    print("OpenEvolve Stock Optimization - Artifacts Channel Demo")
    print("This demo shows how the artifacts channel captures detailed")
    print("execution information for better LLM feedback in evolution.")
    
    try:
        demo_successful_program()
        demo_syntax_error()
        demo_runtime_error()
        demo_validation_error()
        demo_stage1_evaluation()
        
        print(f"\n{'='*60}")
        print("  DEMO COMPLETE")
        print(f"{'='*60}")
        print("\nKey Benefits of Artifacts Channel:")
        print("• Captures compilation/syntax errors with exact line numbers")
        print("• Provides runtime error tracebacks for debugging")
        print("• Records performance metrics (execution time, signal stats)")
        print("• Enables detailed validation error reporting")
        print("• Maintains backward compatibility with existing evaluators")
        print("\nThis information helps the LLM understand what went wrong")
        print("and how to fix it in subsequent evolution generations.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()