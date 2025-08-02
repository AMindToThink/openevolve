"""
Evaluator for the prime number optimization example
"""

import importlib.util
import time
import concurrent.futures
import traceback
import math


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=1.5):
    """
    Run a function with a timeout using concurrent.futures
    
    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds
        
    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def is_prime(n):
    """
    Verify if a number is prime using trial division
    
    Args:
        n: Number to test
        
    Returns:
        True if prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    
    return True


def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and scoring based on:
    1. Size of the largest prime found
    2. Speed of execution (staying under 1 second)
    3. Consistency across multiple runs
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of metrics
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if the required function exists
        if not hasattr(program, "run_prime_search"):
            print(f"Error: program does not have 'run_prime_search' function")
            return {
                "prime_size_score": 0.0,
                "speed_score": 0.0,
                "consistency_score": 0.0,
                "validity_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing run_prime_search function",
            }
        
        # Run multiple trials
        num_trials = 5
        primes = []
        execution_times = []
        success_count = 0
        
        for trial in range(num_trials):
            try:
                # Run with timeout (1.5 seconds to allow for some overhead)
                result = run_with_timeout(program.run_prime_search, timeout_seconds=1.5)
                
                # Handle different result formats
                if isinstance(result, tuple) and len(result) == 2:
                    prime, exec_time = result
                elif isinstance(result, (int, float)):
                    # If only prime is returned, measure our own execution time
                    prime = result
                    exec_time = 1.0  # Default penalty for not reporting time
                else:
                    print(f"Trial {trial}: Invalid result format, expected tuple or number")
                    continue
                
                # Convert to appropriate types
                prime = int(prime)
                exec_time = float(exec_time)
                
                # Validate the prime
                if not is_prime(prime):
                    print(f"Trial {trial}: {prime} is not a valid prime number")
                    continue
                
                # Check execution time constraint
                if exec_time > 1.0:
                    print(f"Trial {trial}: Execution time {exec_time:.4f}s exceeds 1 second limit")
                    # Still record but with penalty
                    exec_time = 1.0  # Cap at limit for scoring
                
                primes.append(prime)
                execution_times.append(exec_time)
                success_count += 1
                
            except TimeoutError as e:
                print(f"Trial {trial}: {str(e)}")
                continue
            except Exception as e:
                print(f"Trial {trial}: Error - {str(e)}")
                print(traceback.format_exc())
                continue
        
        # If all trials failed, return zero scores
        if success_count == 0:
            return {
                "prime_size_score": 0.0,
                "speed_score": 0.0,
                "consistency_score": 0.0,
                "validity_score": 0.0,
                "combined_score": 0.0,
                "error": "All trials failed",
            }
        
        # Calculate metrics
        max_prime = max(primes)
        avg_prime = sum(primes) / len(primes)
        avg_time = sum(execution_times) / len(execution_times)
        
        # Prime size score: logarithmic scaling to reward larger primes
        # Using log base 10 to make the scoring more reasonable
        prime_size_score = math.log10(max(max_prime, 10)) / 10.0  # Normalize to reasonable range
        
        # Speed score: reward faster execution (inverse relationship)
        # Perfect score (1.0) for instant execution, decreasing as time approaches 1 second
        speed_score = max(0.0, (1.0 - avg_time))
        
        # Consistency score: reward consistent prime finding across trials
        if len(primes) > 1:
            prime_std = math.sqrt(sum((p - avg_prime) ** 2 for p in primes) / len(primes))
            consistency_score = 1.0 / (1.0 + prime_std / avg_prime)  # Normalized by average
        else:
            consistency_score = 1.0 if success_count == 1 else 0.0
        
        # Validity score: based on success rate
        validity_score = success_count / num_trials
        
        # Combined score with weights:
        # - Prime size is most important (50%)
        # - Speed is important (25%) 
        # - Consistency matters (15%)
        # - Validity is baseline requirement (10%)
        combined_score = (
            0.50 * prime_size_score +
            0.25 * speed_score +
            0.15 * consistency_score +
            0.10 * validity_score
        )
        
        return {
            "prime_size_score": float(prime_size_score),
            "speed_score": float(speed_score),
            "consistency_score": float(consistency_score),
            "validity_score": float(validity_score),
            "combined_score": float(combined_score),
            "max_prime": int(max_prime),
            "avg_prime": float(avg_prime),
            "avg_execution_time": float(avg_time),
            "success_rate": float(validity_score),
        }
        
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "prime_size_score": 0.0,
            "speed_score": 0.0,
            "consistency_score": 0.0,
            "validity_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
        }


def evaluate_stage1(program_path):
    """First stage evaluation with basic validation"""
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if the required function exists
        if not hasattr(program, "run_prime_search"):
            print(f"Stage 1 validation: Program does not have 'run_prime_search' function")
            return {"runs_successfully": 0.0, "error": "Missing run_prime_search function"}
        
        try:
            # Run a single trial with timeout
            result = run_with_timeout(program.run_prime_search, timeout_seconds=1.5)
            
            # Handle different result formats
            if isinstance(result, tuple) and len(result) == 2:
                prime, exec_time = result
            elif isinstance(result, (int, float)):
                prime = result
                exec_time = 1.0
            else:
                print(f"Stage 1: Invalid result format")
                return {"runs_successfully": 0.0, "error": "Invalid result format"}
            
            # Convert and validate
            prime = int(prime)
            exec_time = float(exec_time)
            
            # Basic validation
            if not is_prime(prime):
                print(f"Stage 1: {prime} is not a valid prime")
                return {"runs_successfully": 0.5, "error": "Invalid prime"}
            
            if exec_time > 1.0:
                print(f"Stage 1: Execution time {exec_time:.4f}s exceeds limit")
                return {"runs_successfully": 0.7, "timeout_violation": True}
            
            # Calculate basic scores
            prime_size_score = math.log10(max(prime, 10)) / 10.0
            speed_score = max(0.0, (1.0 - exec_time))
            
            return {
                "runs_successfully": 1.0,
                "prime_size_score": float(prime_size_score),
                "speed_score": float(speed_score),
                "basic_score": float((prime_size_score + speed_score) / 2),
            }
            
        except TimeoutError as e:
            print(f"Stage 1 evaluation timed out: {e}")
            return {"runs_successfully": 0.0, "error": "Timeout"}
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            print(traceback.format_exc())
            return {"runs_successfully": 0.0, "error": str(e)}
            
    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        print(traceback.format_exc())
        return {"runs_successfully": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """Second stage evaluation with full testing"""
    return evaluate(program_path)