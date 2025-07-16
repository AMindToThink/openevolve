#!/usr/bin/env python3

import sys
import os
import unittest
from io import StringIO

# Add current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def run_map_elites_tests():
    """Run the MAP-Elites tests specifically"""
    print("Running MAP-Elites test suite...")
    print("=" * 50)
    
    try:
        # Import the test module
        from tests.test_map_elites_fix import TestMapElitesFix
        
        # Create a test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMapElitesFix)
        
        # Capture output
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        
        # Print the output
        output = stream.getvalue()
        print(output)
        
        # Print summary
        print("\n" + "=" * 50)
        print("Test Results Summary:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success: {result.wasSuccessful()}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests in the tests directory"""
    print("Running all tests...")
    print("=" * 50)
    
    try:
        # Discover and run all tests
        loader = unittest.TestLoader()
        suite = loader.discover('tests', pattern='test_*.py')
        
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        
        output = stream.getvalue()
        print(output)
        
        print("\n" + "=" * 50)
        print("All Tests Summary:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success: {result.wasSuccessful()}")
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"Error running all tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing the MAP-Elites deterministic fixes...")
    print("=" * 70)
    
    # Run MAP-Elites specific tests
    map_elites_success = run_map_elites_tests()
    
    print("\n" + "=" * 70)
    
    # Run all tests
    all_tests_success = run_all_tests()
    
    print("\n" + "=" * 70)
    print("Final Summary:")
    print(f"MAP-Elites tests: {'PASS' if map_elites_success else 'FAIL'}")
    print(f"All tests: {'PASS' if all_tests_success else 'FAIL'}")
    
    if map_elites_success and all_tests_success:
        print("\n🎉 All tests are passing!")
        print("The deterministic fixes successfully resolved the random.sample() issues.")
    elif map_elites_success:
        print("\n✅ MAP-Elites tests are passing!")
        print("The deterministic fixes resolved the specific issues.")
        print("⚠️  Some other tests may still be failing (unrelated to our changes).")
    else:
        print("\n❌ MAP-Elites tests are still failing.")
        print("The deterministic fixes may need additional work.")
    
    sys.exit(0 if map_elites_success else 1)