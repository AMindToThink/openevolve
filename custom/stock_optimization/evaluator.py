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

# Import OpenEvolve's EvaluationResult for artifacts support
from openevolve.evaluation_result import EvaluationResult

# Suppress yfinance warnings
warnings.filterwarnings("ignore")


class TimeoutError(Exception):
    pass


class ExecutionError(Exception):
    """Exception that carries execution artifacts"""

    def __init__(self, message, execution_info=None):
        super().__init__(message)
        self.execution_info = execution_info or {}


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def load_stock_data(symbol="SPY", period="5y"):
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
    
    Standard pandas backtesting approach:
    - Signals: 1=buy, -1=sell, 0=hold current position
    - Position tracking: accumulate signals over time
    - Returns: position * daily_returns

    Args:
        prices: Series of stock prices
        signals: Series of trading signals (1=buy, -1=sell, 0=hold)

    Returns:
        Dictionary with performance metrics
    """
    # Calculate daily returns
    daily_returns = calculate_returns(prices)

    # Convert signals to positions using standard approach
    # 1=buy (enter long), -1=sell (exit long), 0=hold current position
    position = 0
    positions = []
    
    for signal in signals:
        if signal == 1:  # Buy signal - enter long position
            position = 1
        elif signal == -1:  # Sell signal - exit position
            position = 0
        # signal == 0 means hold current position (no change)
        positions.append(position)
    
    positions = pd.Series(positions, index=signals.index)
    
    # Calculate strategy returns: position held during each period * daily return
    # Shift positions to avoid look-ahead bias (use yesterday's position for today's return)
    strategy_returns = daily_returns * positions.shift(1)
    
    # Remove NaN values (first day has no prior position)
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {
            "total_return": 0.0,
            "annualized_volatility": 1e6,  # Very high volatility instead of infinity
            "sharpe_ratio": -1e6,  # Very bad Sharpe ratio instead of negative infinity
            "max_drawdown": 1.0,
        }

    # Calculate performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    annualized_volatility = strategy_returns.std() * np.sqrt(252)  # 252 trading days

    # Sharpe ratio (assuming risk-free rate of 0)
    if annualized_volatility > 0:
        sharpe_ratio = (strategy_returns.mean() * 252) / annualized_volatility
    else:
        sharpe_ratio = -1e6  # Very bad Sharpe ratio for invalid strategies

    # Maximum drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "total_return": float(total_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
    }


def evaluate_strategy_performance(results):
    """
    Evaluate strategy performance and return normalized scores

    Args:
        results: Dictionary with strategy performance metrics

    Returns:
        Dictionary with normalized scores
    """
    total_return = results.get("total_return", 0.0)
    annualized_volatility = results.get("annualized_volatility", float("inf"))
    sharpe_ratio = results.get("sharpe_ratio", -float("inf"))
    max_drawdown = results.get("max_drawdown", -1.0)

    # Return score: normalize to [0, 1] range with strong emphasis on positive returns
    # Use realistic thresholds: 0% = 0.0, 300% over 5 years (~25% annualized) = 1.0
    # Negative returns get heavily penalized
    if total_return <= 0:
        return_score = 0.0  # Zero score for any losses or no gains
    else:
        return_score = np.clip(total_return / 3.0, 0.0, 1.0)

    # Volatility score: lower volatility is better, but zero volatility means no trading
    # Realistic range: 10% annualized (score=1.0) to 50% annualized (score=0.0)
    # Zero volatility (no trading) gets moderate score, not perfect
    if annualized_volatility >= 1e6:
        volatility_score = 0.0
    elif annualized_volatility <= 0:
        volatility_score = 0.5  # Moderate score for no-risk strategies
    else:
        volatility_score = np.clip(1.0 - (annualized_volatility - 0.10) / 0.40, 0.0, 1.0)

    # Sharpe ratio score: higher is better
    # Realistic thresholds: 0 = 0.0, 2.5 = 1.0 (excellent strategies rarely exceed 2.5)
    # This prevents saturation at modest Sharpe ratios
    if sharpe_ratio <= -1e6:
        sharpe_score = 0.0
    else:
        sharpe_score = np.clip(sharpe_ratio / 2.5, 0.0, 1.0)

    # Drawdown score: smaller drawdowns are better
    # Realistic range: 0% drawdown (score=1.0) to 50% drawdown (score=0.0)
    # Most strategies experience 10-30% max drawdowns
    drawdown_score = np.clip(1.0 + max_drawdown / 0.5, 0.0, 1.0)

    # Combined score with heavy emphasis on returns and profitable trading
    # Weights: 70% returns, 10% low volatility, 15% Sharpe ratio, 5% low drawdown
    # This prioritizes profitable strategies over low-risk "do nothing" strategies
    combined_score = (
        0.70 * return_score + 0.10 * volatility_score + 0.15 * sharpe_score + 0.05 * drawdown_score
    )

    # Ensure no infinite values are returned for JSON serialization
    def safe_float(value):
        if value == float("inf"):
            return 1e6  # Very large number instead of infinity
        elif value == -float("inf"):
            return -1e6  # Very small number instead of negative infinity
        elif np.isnan(value):
            return 0.0  # Zero instead of NaN
        else:
            return float(value)

    return {
        "return_score": safe_float(return_score),
        "volatility_score": safe_float(volatility_score),
        "sharpe_score": safe_float(sharpe_score),
        "drawdown_score": safe_float(drawdown_score),
        "combined_score": safe_float(combined_score),
        "raw_total_return": safe_float(total_return),
        "raw_volatility": safe_float(annualized_volatility),
        "raw_sharpe_ratio": safe_float(sharpe_ratio),
        "raw_max_drawdown": safe_float(max_drawdown),
    }


def run_with_timeout(program_path, timeout_seconds=60):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        tuple: (signals from the program, execution_info dict with artifacts)
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
    # If an error occurs, save the error with full traceback
    print(f"Error in subprocess: {{str(e)}}")
    tb = traceback.format_exc()
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e), 'traceback': tb, 'error_type': type(e).__name__}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    execution_info = {"stdout": "", "stderr": "", "exit_code": -1, "timeout": False}

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(program_path)),
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            execution_info["stdout"] = stdout.decode()
            execution_info["stderr"] = stderr.decode()
            execution_info["exit_code"] = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {execution_info['stdout']}")
            if execution_info["stderr"]:
                print(f"Subprocess stderr: {execution_info['stderr']}")

            # Load the results (even on error to get artifacts)
            results = {}
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

            # Merge execution info with program results
            results.update(execution_info)

            # Still raise an error for non-zero exit codes, but preserve all info
            if execution_info["exit_code"] != 0:
                raise ExecutionError(
                    f"Process exited with code {execution_info['exit_code']}", results
                )

            # Check if an error was returned
            if "error" in results:
                raise ExecutionError(f"Program execution failed: {results['error']}", results)

            return results["signals"], results

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            execution_info["timeout"] = True
            execution_info["stderr"] = f"Process timed out after {timeout_seconds} seconds"
            raise ExecutionError(
                f"Process timed out after {timeout_seconds} seconds", execution_info
            )

    except Exception as e:
        # For any other exceptions, preserve execution info
        if isinstance(e, ExecutionError):
            raise  # Re-raise as-is
        else:
            raise ExecutionError(str(e), execution_info)

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
        Dictionary of performance metrics (maintains backward compatibility)
        Artifacts are processed internally but not exposed for framework compatibility
    """
    execution_info = {}
    start_time = time.time()

    try:
        # Use subprocess to run with timeout and isolation
        signals, execution_info = run_with_timeout(program_path, timeout_seconds=60)
        end_time = time.time()
        eval_time = end_time - start_time

        # Load stock data for backtesting
        stock_data = load_stock_data("SPY", "5y")

        # Validate signals format
        try:
            if not isinstance(signals, pd.Series):
                print(f"Error: Strategy must return a pandas Series of signals")
                metrics = {
                    "return_score": 0.0,
                    "volatility_score": 0.0,
                    "sharpe_score": 0.0,
                    "drawdown_score": 0.0,
                    "combined_score": 0.0,
                    "eval_time": float(eval_time),
                }
                artifacts = {
                    "error": "Invalid signals format - must be pandas Series",
                    "error_type": "ValidationError",
                    "stdout": execution_info.get("stdout", ""),
                    "stderr": execution_info.get("stderr", ""),
                }
                # Create EvaluationResult for internal processing but return plain dict for compatibility
                result = EvaluationResult(metrics=metrics, artifacts=artifacts)
                return result.to_dict()  # Return plain dict for OpenEvolve compatibility

            # Validate signal values are in expected range
            unique_signals = signals.unique()
            valid_signals = set([-1, 0, 1])
            if not all(sig in valid_signals for sig in unique_signals):
                print(
                    f"Error: Signals must be -1 (sell), 0 (hold), or 1 (buy). Got: {unique_signals}"
                )
                metrics = {
                    "return_score": 0.0,
                    "volatility_score": 0.0,
                    "sharpe_score": 0.0,
                    "drawdown_score": 0.0,
                    "combined_score": 0.0,
                    "eval_time": float(eval_time),
                }
                artifacts = {
                    "error": f"Invalid signal values: {unique_signals}",
                    "error_type": "ValidationError",
                    "stdout": execution_info.get("stdout", ""),
                    "stderr": execution_info.get("stderr", ""),
                    "invalid_signals": str(unique_signals),
                }
                # Create EvaluationResult for internal processing but return plain dict for compatibility
                result = EvaluationResult(metrics=metrics, artifacts=artifacts)
                return result.to_dict()  # Return plain dict for OpenEvolve compatibility

            # Validate signal length matches stock data length
            if len(signals) != len(stock_data):
                print(
                    f"Error: Signal length ({len(signals)}) does not match stock data length ({len(stock_data)})"
                )
                metrics = {
                    "return_score": 0.0,
                    "volatility_score": 0.0,
                    "sharpe_score": 0.0,
                    "drawdown_score": 0.0,
                    "combined_score": 0.0,
                    "eval_time": float(eval_time),
                }
                artifacts = {
                    "error": f"Signal length mismatch: got {len(signals)}, expected {len(stock_data)}",
                    "error_type": "ValidationError",
                    "stdout": execution_info.get("stdout", ""),
                    "stderr": execution_info.get("stderr", ""),
                    "signal_length": len(signals),
                    "expected_length": len(stock_data),
                }
                # Create EvaluationResult for internal processing but return plain dict for compatibility
                result = EvaluationResult(metrics=metrics, artifacts=artifacts)
                return result.to_dict()  # Return plain dict for OpenEvolve compatibility

            # Backtest the strategy using evaluator's backtesting logic
            results = backtest_strategy(stock_data["Close"], signals)

            # Evaluate and score the strategy
            scores = evaluate_strategy_performance(results)

            # Add eval_time to scores
            scores["eval_time"] = float(eval_time)

            # Create performance artifacts
            artifacts = {
                "stdout": execution_info.get("stdout", ""),
                "stderr": execution_info.get("stderr", ""),
                "signal_distribution": str(signals.value_counts().to_dict()),
                "num_signals": len(signals),
                "num_trades": int((signals.diff() != 0).sum()),
                "execution_time": float(eval_time),
            }

            # Create EvaluationResult for internal processing but return plain dict for compatibility
            result = EvaluationResult(metrics=scores, artifacts=artifacts)
            return result.to_dict()  # Return plain dict for OpenEvolve compatibility

        except Exception as e:
            print(f"Error validating/backtesting strategy: {str(e)}")
            print(traceback.format_exc())
            end_time = time.time()
            eval_time = end_time - start_time

            metrics = {
                "return_score": 0.0,
                "volatility_score": 0.0,
                "sharpe_score": 0.0,
                "drawdown_score": 0.0,
                "combined_score": 0.0,
                "eval_time": float(eval_time),
            }
            artifacts = {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "failure_stage": "validation_or_backtesting",
                "stdout": execution_info.get("stdout", ""),
                "stderr": execution_info.get("stderr", ""),
            }
            # Create EvaluationResult for internal processing but return plain dict for compatibility
            result = EvaluationResult(metrics=metrics, artifacts=artifacts)
            return result.to_dict()  # Return plain dict for OpenEvolve compatibility

    except ExecutionError as e:
        print(f"Execution failed: {str(e)}")
        print(traceback.format_exc())
        end_time = time.time()
        eval_time = end_time - start_time

        metrics = {
            "return_score": 0.0,
            "volatility_score": 0.0,
            "sharpe_score": 0.0,
            "drawdown_score": 0.0,
            "combined_score": 0.0,
            "eval_time": float(eval_time),
        }

        # Extract artifacts from execution info
        artifacts = {
            "error": str(e),
            "error_type": "ExecutionError",
            "failure_stage": "execution",
            "stdout": e.execution_info.get("stdout", ""),
            "stderr": e.execution_info.get("stderr", ""),
            "exit_code": e.execution_info.get("exit_code", -1),
            "timeout": e.execution_info.get("timeout", False),
        }

        # Add additional error details if available
        if "traceback" in e.execution_info:
            artifacts["traceback"] = e.execution_info["traceback"]
        if "error_type" in e.execution_info:
            artifacts["program_error_type"] = e.execution_info["error_type"]

        # Create EvaluationResult for internal processing but return plain dict for compatibility
        result = EvaluationResult(metrics=metrics, artifacts=artifacts)
        return result.to_dict()  # Return plain dict for OpenEvolve compatibility

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        end_time = time.time()
        eval_time = end_time - start_time

        metrics = {
            "return_score": 0.0,
            "volatility_score": 0.0,
            "sharpe_score": 0.0,
            "drawdown_score": 0.0,
            "combined_score": 0.0,
            "eval_time": float(eval_time),
        }
        artifacts = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "failure_stage": "evaluator_setup",
        }
        # Create EvaluationResult for internal processing but return plain dict for compatibility
        result = EvaluationResult(metrics=metrics, artifacts=artifacts)
        return result.to_dict()  # Return plain dict for OpenEvolve compatibility


def evaluate_stage1(program_path):
    """
    First stage evaluation with basic validation

    Returns:
        Dictionary of performance metrics (maintains backward compatibility)
        Artifacts are processed internally but not exposed for framework compatibility
    """
    start_time = time.time()
    execution_info = {}

    try:
        # Use subprocess with shorter timeout for stage 1
        try:
            signals, execution_info = run_with_timeout(program_path, timeout_seconds=30)
            end_time = time.time()
            eval_time = end_time - start_time

            # Load minimal test data for validation
            test_data = load_stock_data("SPY", "1y").tail(100)  # Last 100 days only for quick test

            # Basic validation
            if not isinstance(signals, pd.Series):
                print(f"Stage 1: Invalid signals format")
                metrics = {"runs_successfully": 0.0, "eval_time": float(eval_time)}
                artifacts = {
                    "error": "Invalid signals format",
                    "error_type": "ValidationError",
                    "failure_stage": "stage1_validation",
                    "stdout": execution_info.get("stdout", ""),
                    "stderr": execution_info.get("stderr", ""),
                }
                # Create EvaluationResult for internal processing but return plain dict for compatibility
                result = EvaluationResult(metrics=metrics, artifacts=artifacts)
                return result.to_dict()  # Return plain dict for OpenEvolve compatibility

            # Validate signal values
            unique_signals = signals.unique()
            valid_signals = set([-1, 0, 1])
            if not all(sig in valid_signals for sig in unique_signals):
                print(f"Stage 1: Invalid signal values: {unique_signals}")
                metrics = {"runs_successfully": 0.5, "eval_time": float(eval_time)}
                artifacts = {
                    "error": f"Invalid signal values: {unique_signals}",
                    "error_type": "ValidationError",
                    "failure_stage": "stage1_validation",
                    "invalid_signals": str(unique_signals),
                    "stdout": execution_info.get("stdout", ""),
                    "stderr": execution_info.get("stderr", ""),
                }
                # Create EvaluationResult for internal processing but return plain dict for compatibility
                result = EvaluationResult(metrics=metrics, artifacts=artifacts)
                return result.to_dict()  # Return plain dict for OpenEvolve compatibility

            # Validate signal length matches test data length  
            if len(signals) != len(test_data):
                print(f"Stage 1: Signal length mismatch: {len(signals)} vs {len(test_data)}")
                metrics = {"runs_successfully": 0.0, "eval_time": float(eval_time)}
                artifacts = {
                    "error": f"Signal length mismatch: got {len(signals)}, expected {len(test_data)}",
                    "error_type": "ValidationError",
                    "failure_stage": "stage1_validation",
                    "signal_length": len(signals),
                    "expected_length": len(test_data),
                    "stdout": execution_info.get("stdout", ""),
                    "stderr": execution_info.get("stderr", ""),
                }
                # Create EvaluationResult for internal processing but return plain dict for compatibility
                result = EvaluationResult(metrics=metrics, artifacts=artifacts)
                return result.to_dict()  # Return plain dict for OpenEvolve compatibility

            # Align signals with test data (signals might be from full dataset)
            aligned_signals = signals.reindex(test_data.index, fill_value=0)

            # Backtest to get performance metrics
            results = backtest_strategy(test_data["Close"], aligned_signals)

            # Check for reasonable values
            total_return = results.get("total_return", 0.0)
            volatility = results.get("annualized_volatility", float("inf"))

            # Sanity checks
            if not isinstance(total_return, (int, float)) or not np.isfinite(total_return):
                metrics = {"runs_successfully": 0.5, "eval_time": float(eval_time)}
                artifacts = {
                    "error": "Invalid total_return",
                    "error_type": "ValidationError",
                    "failure_stage": "stage1_metrics",
                    "total_return": str(total_return),
                    "stdout": execution_info.get("stdout", ""),
                    "stderr": execution_info.get("stderr", ""),
                }
                # Create EvaluationResult for internal processing but return plain dict for compatibility
                result = EvaluationResult(metrics=metrics, artifacts=artifacts)
                return result.to_dict()  # Return plain dict for OpenEvolve compatibility

            if not isinstance(volatility, (int, float)) or volatility < 0:
                metrics = {"runs_successfully": 0.5, "eval_time": float(eval_time)}
                artifacts = {
                    "error": "Invalid volatility",
                    "error_type": "ValidationError",
                    "failure_stage": "stage1_metrics",
                    "volatility": str(volatility),
                    "stdout": execution_info.get("stdout", ""),
                    "stderr": execution_info.get("stderr", ""),
                }
                # Create EvaluationResult for internal processing but return plain dict for compatibility
                result = EvaluationResult(metrics=metrics, artifacts=artifacts)
                return result.to_dict()  # Return plain dict for OpenEvolve compatibility

            # Calculate basic score
            basic_scores = evaluate_strategy_performance(results)

            metrics = {
                "runs_successfully": 1.0,
                "basic_score": basic_scores["combined_score"],
                "return_score": basic_scores["return_score"],
                "volatility_score": basic_scores["volatility_score"],
                "eval_time": float(eval_time),
            }

            artifacts = {
                "stdout": execution_info.get("stdout", ""),
                "stderr": execution_info.get("stderr", ""),
                "signal_distribution": str(aligned_signals.value_counts().to_dict()),
                "num_signals": len(aligned_signals),
                "test_data_length": len(test_data),
            }

            # Create EvaluationResult for internal processing but return plain dict for compatibility
            result = EvaluationResult(metrics=metrics, artifacts=artifacts)
            return result.to_dict()  # Return plain dict for OpenEvolve compatibility

        except ExecutionError as e:
            print(f"Stage 1 execution failed: {e}")
            print(traceback.format_exc())
            end_time = time.time()
            eval_time = end_time - start_time

            metrics = {"runs_successfully": 0.0, "eval_time": float(eval_time)}
            artifacts = {
                "error": str(e),
                "error_type": "ExecutionError",
                "failure_stage": "stage1_execution",
                "stdout": e.execution_info.get("stdout", ""),
                "stderr": e.execution_info.get("stderr", ""),
                "exit_code": e.execution_info.get("exit_code", -1),
                "timeout": e.execution_info.get("timeout", False),
            }

            if "traceback" in e.execution_info:
                artifacts["traceback"] = e.execution_info["traceback"]
            if "error_type" in e.execution_info:
                artifacts["program_error_type"] = e.execution_info["error_type"]

            # Create EvaluationResult for internal processing but return plain dict for compatibility
            result = EvaluationResult(metrics=metrics, artifacts=artifacts)
            return result.to_dict()  # Return plain dict for OpenEvolve compatibility

        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            print(traceback.format_exc())
            end_time = time.time()
            eval_time = end_time - start_time

            metrics = {"runs_successfully": 0.0, "eval_time": float(eval_time)}
            artifacts = {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "failure_stage": "stage1_other",
                "stdout": execution_info.get("stdout", ""),
                "stderr": execution_info.get("stderr", ""),
            }
            # Create EvaluationResult for internal processing but return plain dict for compatibility
            result = EvaluationResult(metrics=metrics, artifacts=artifacts)
            return result.to_dict()  # Return plain dict for OpenEvolve compatibility

    except Exception as e:
        print(f"Stage 1 evaluation failed completely: {e}")
        print(traceback.format_exc())
        end_time = time.time()
        eval_time = end_time - start_time

        metrics = {"runs_successfully": 0.0, "eval_time": float(eval_time)}
        artifacts = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "failure_stage": "stage1_setup",
        }
        # Create EvaluationResult for internal processing but return plain dict for compatibility
        result = EvaluationResult(metrics=metrics, artifacts=artifacts)
        return result.to_dict()  # Return plain dict for OpenEvolve compatibility


def evaluate_stage2(program_path):
    """Second stage evaluation with full testing"""
    return evaluate(program_path)
