#!/usr/bin/env python3

import sys
import os
import subprocess

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

def run_specific_test():
    """Run the specific MAP-Elites test"""
    try:
        # Change to the correct directory
        os.chdir('/home/runner/work/openevolve/openevolve')
        
        # Run the test
        result = subprocess.run([
            sys.executable, '-m', 'unittest', 
            'tests.test_map_elites_fix.TestMapElitesFix.test_map_elites_replacement_basic',
            '-v'
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running test: {e}")
        return False

def run_all_tests():
    """Run all tests in the test suite"""
    try:
        os.chdir('/home/runner/work/openevolve/openevolve')
        
        result = subprocess.run([
            sys.executable, '-m', 'unittest', 
            'discover', 'tests', '-v'
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    print("Testing MAP-Elites fix...")
    print("=" * 60)
    
    # First run the specific test
    print("1. Running MAP-Elites replacement test...")
    specific_passed = run_specific_test()
    
    print("\n" + "=" * 60)
    print("2. Running all tests...")
    all_passed = run_all_tests()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  MAP-Elites specific test: {'PASS' if specific_passed else 'FAIL'}")
    print(f"  All tests: {'PASS' if all_passed else 'FAIL'}")
    
    if all_passed:
        print("\n✅ All tests passed! The deterministic fixes are working correctly.")
    else:
        print("\n❌ Some tests failed! Check the output above for details.")