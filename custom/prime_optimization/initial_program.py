# EVOLVE-BLOCK-START
"""Prime number optimization for OpenEvolve"""
import time


def find_largest_prime(max_time=1.0):
    """
    Find the largest prime number within the given time limit.
    
    Args:
        max_time: Maximum time in seconds to search for primes
        
    Returns:
        Tuple of (largest_prime, execution_time)
    """
    start_time = time.time()
    largest_prime = 2
    candidate = 3
    
    while time.time() - start_time < max_time:
        if is_prime(candidate):
            largest_prime = candidate
        candidate += 2  # Skip even numbers after 2
        
        # Break if we're getting close to timeout
        if time.time() - start_time > max_time * 0.95:
            break
    
    execution_time = time.time() - start_time
    return largest_prime, execution_time


def is_prime(n):
    """
    Basic primality test using trial division.
    
    Args:
        n: Number to test for primality
        
    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    
    return True


# EVOLVE-BLOCK-END


def run_prime_search():
    """Entry point for prime search"""
    return find_largest_prime()


if __name__ == "__main__":
    prime, exec_time = run_prime_search()
    print(f"Found largest prime: {prime} in {exec_time:.4f} seconds")