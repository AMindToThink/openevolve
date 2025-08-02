"""
Evaluator for the stock optimization example
"""

import importlib.util
import pandas as pd
import numpy as np
import yfinance as yf
import traceback
import warnings
import os
import signal
import subprocess
import tempfile
import pickle
import sys
import time
from datetime import datetime, timedelta

# Suppress yfinance warnings
warnings.filterwarnings('ignore')


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def load_stock_data(symbol='SPY', period='5y'):
    """
    Load historical stock data using yfinance
    
    Args:
        symbol: Stock symbol to load (default: SPY for S&P 500)
        period: Time period to load (5y = 5 years)
        
    Returns:
        DataFrame with historical stock data
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    
    if data.empty:
        raise ValueError(f"No data found for symbol {symbol}")
        
    return data


def calculate_returns(prices):
    """Calculate daily returns from price series"""
    return prices.pct_change().dropna()


def backtest_strategy(prices, signals):
    """
    Backtest the trading strategy and calculate performance metrics.
    
    Args:
        prices: Series of stock prices
        signals: Series of trading signals
        
    Returns:
        Dictionary with performance metrics
    """
    # Calculate daily returns
    daily_returns = calculate_returns(prices)
    
    # Calculate strategy returns (shift signals to avoid look-ahead bias)
    strategy_returns = daily_returns * signals.shift(1)
    
    # Remove NaN values
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_volatility': float('inf'),
            'sharpe_ratio': -float('inf'),
            'max_drawdown': 1.0
        }
    
    # Calculate performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    annualized_volatility = strategy_returns.std() * np.sqrt(252)  # 252 trading days
    
    # Sharpe ratio (assuming risk-free rate of 0)
    if annualized_volatility > 0:
        sharpe_ratio = (strategy_returns.mean() * 252) / annualized_volatility
    else:
        sharpe_ratio = -float('inf')  # Better than 0.0 for invalid strategies
    
    # Maximum drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'total_return': float(total_return),
        'annualized_volatility': float(annualized_volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown)
    }


def evaluate_strategy_performance(results):
    """
    Evaluate strategy performance and return normalized scores
    
    Args:
        results: Dictionary with strategy performance metrics
        
    Returns:
        Dictionary with normalized scores
    """
    total_return = results.get('total_return', 0.0)
    annualized_volatility = results.get('annualized_volatility', float('inf'))
    sharpe_ratio = results.get('sharpe_ratio', -float('inf'))
    max_drawdown = results.get('max_drawdown', -1.0)
    
    # Return score: normalize to [0, 1] range
    # Assume good strategies can achieve 10-50% total returns over 5 years
    return_score = np.clip(total_return / 0.5, 0.0, 1.0)
    
    # Volatility score: lower volatility is better
    # Assume good strategies have 10-30% annualized volatility
    if annualized_volatility == float('inf') or annualized_volatility <= 0:
        volatility_score = 0.0
    else:
        volatility_score = np.clip(1.0 - (annualized_volatility - 0.1) / 0.3, 0.0, 1.0)
    
    # Sharpe ratio score: higher is better
    # Good strategies typically have Sharpe > 0.5, excellent > 1.0
    if sharpe_ratio == -float('inf'):
        sharpe_score = 0.0
    else:
        sharpe_score = np.clip(sharpe_ratio / 1.5, 0.0, 1.0)
    
    # Drawdown score: smaller drawdowns are better
    # Good strategies keep max drawdown under 20%
    drawdown_score = np.clip(1.0 + max_drawdown / 0.2, 0.0, 1.0)
    
    # Combined score with emphasis on returns and low risk
    # Weights: 40% returns, 25% low volatility, 20% Sharpe ratio, 15% low drawdown
    combined_score = (
        0.40 * return_score +
        0.25 * volatility_score +
        0.20 * sharpe_score +
        0.15 * drawdown_score
    )
    
    return {
        'return_score': float(return_score),
        'volatility_score': float(volatility_score),
        'sharpe_score': float(sharpe_score),
        'drawdown_score': float(drawdown_score),
        'combined_score': float(combined_score),
        'raw_total_return': float(total_return),
        'raw_volatility': float(annualized_volatility),
        'raw_sharpe_ratio': float(sharpe_ratio),
        'raw_max_drawdown': float(max_drawdown)
    }


def run_with_timeout(program_path, timeout_seconds=60):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        signals from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback
import pandas as pd

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Load stock data
    import yfinance as yf
    import warnings
    warnings.filterwarnings('ignore')
    
    ticker = yf.Ticker('SPY')
    stock_data = ticker.history(period='5y')
    
    if stock_data.empty:
        raise ValueError("No data found for symbol SPY")
    
    # Run the strategy function
    print("Calling run_stock_optimization()...")
    signals = program.run_stock_optimization(stock_data)
    print(f"run_stock_optimization() returned successfully")

    # Save results to a file
    results = {{
        'signals': signals
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(program_path))
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["signals"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path):
    """
    Evaluate the stock trading strategy by running it on historical data
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of performance metrics
    """
    try:
        # Use subprocess to run with timeout and isolation
        start_time = time.time()
        signals = run_with_timeout(program_path, timeout_seconds=60)
        end_time = time.time()
        eval_time = end_time - start_time
        
        # Load stock data for backtesting
        stock_data = load_stock_data('SPY', '5y')
        
        # Validate signals format
        try:
            if not isinstance(signals, pd.Series):
                print(f"Error: Strategy must return a pandas Series of signals")
                return {
                    "return_score": 0.0,
                    "volatility_score": 0.0,
                    "sharpe_score": 0.0,
                    "drawdown_score": 0.0,
                    "combined_score": 0.0,
                    "error": "Invalid signals format - must be pandas Series",
                }
            
            # Validate signal values are in expected range
            unique_signals = signals.unique()
            valid_signals = set([-1, 0, 1])
            if not all(sig in valid_signals for sig in unique_signals):
                print(f"Error: Signals must be -1 (sell), 0 (hold), or 1 (buy). Got: {unique_signals}")
                return {
                    "return_score": 0.0,
                    "volatility_score": 0.0,
                    "sharpe_score": 0.0,
                    "drawdown_score": 0.0,
                    "combined_score": 0.0,
                    "error": f"Invalid signal values: {unique_signals}",
                }
            
            # Backtest the strategy using evaluator's backtesting logic
            results = backtest_strategy(stock_data['Close'], signals)
            
            # Evaluate and score the strategy
            scores = evaluate_strategy_performance(results)
            
            # Add eval_time to scores
            scores['eval_time'] = float(eval_time)
            return scores
            
        except Exception as e:
            print(f"Error validating/backtesting strategy: {str(e)}")
            print(traceback.format_exc())
            return {
                "return_score": 0.0,
                "volatility_score": 0.0,
                "sharpe_score": 0.0,
                "drawdown_score": 0.0,
                "combined_score": 0.0,
                "eval_time": float(eval_time) if 'eval_time' in locals() else 0.0,
                "error": str(e),
            }
        
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "return_score": 0.0,
            "volatility_score": 0.0,
            "sharpe_score": 0.0,
            "drawdown_score": 0.0,
            "combined_score": 0.0,
            "eval_time": 0.0,
            "error": str(e),
        }


def evaluate_stage1(program_path):
    """First stage evaluation with basic validation"""
    try:
        # Use subprocess with shorter timeout for stage 1
        try:
            signals = run_with_timeout(program_path, timeout_seconds=30)
            
            # Load minimal test data for validation
            test_data = load_stock_data('SPY', '1y').tail(100)  # Last 100 days only for quick test
            
            # Basic validation
            if not isinstance(signals, pd.Series):
                print(f"Stage 1: Invalid signals format")
                return {"runs_successfully": 0.0, "error": "Invalid signals format"}
            
            # Validate signal values
            unique_signals = signals.unique()
            valid_signals = set([-1, 0, 1])
            if not all(sig in valid_signals for sig in unique_signals):
                print(f"Stage 1: Invalid signal values: {unique_signals}")
                return {"runs_successfully": 0.5, "error": f"Invalid signal values: {unique_signals}"}
            
            # Align signals with test data (signals might be from full dataset)
            aligned_signals = signals.reindex(test_data.index, fill_value=0)
            
            # Backtest to get performance metrics
            results = backtest_strategy(test_data['Close'], aligned_signals)
            
            # Check for reasonable values
            total_return = results.get('total_return', 0.0)
            volatility = results.get('annualized_volatility', float('inf'))
            
            # Sanity checks
            if not isinstance(total_return, (int, float)) or not np.isfinite(total_return):
                return {"runs_successfully": 0.5, "error": "Invalid total_return"}
            
            if not isinstance(volatility, (int, float)) or volatility < 0:
                return {"runs_successfully": 0.5, "error": "Invalid volatility"}
            
            # Calculate basic score
            basic_scores = evaluate_strategy_performance(results)
            
            return {
                "runs_successfully": 1.0,
                "basic_score": basic_scores['combined_score'],
                "return_score": basic_scores['return_score'],
                "volatility_score": basic_scores['volatility_score'],
            }
            
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            print(traceback.format_exc())
            return {"runs_successfully": 0.0, "error": str(e)}
            
    except Exception as e:
        print(f"Stage 1 evaluation failed completely: {e}")
        print(traceback.format_exc())
        return {"runs_successfully": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """Second stage evaluation with full testing"""
    return evaluate(program_path)