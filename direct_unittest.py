#!/usr/bin/env python3

import sys
import os
import unittest

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

# Set up the environment
os.chdir('/home/runner/work/openevolve/openevolve')

def run_tests():
    """Run the tests directly"""
    print("Running MAP-Elites tests directly...")
    print("=" * 50)
    
    try:
        # Import test module
        from tests.test_map_elites_fix import TestMapElitesFix
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add specific tests
        suite.addTest(TestMapElitesFix('test_map_elites_replacement_basic'))
        suite.addTest(TestMapElitesFix('test_map_elites_population_limit_respects_diversity'))
        suite.addTest(TestMapElitesFix('test_map_elites_best_program_protection'))
        suite.addTest(TestMapElitesFix('test_map_elites_feature_map_consistency'))
        suite.addTest(TestMapElitesFix('test_remove_program_from_database_method'))
        suite.addTest(TestMapElitesFix('test_map_elites_non_elite_program_removal_priority'))
        
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Print summary
        print("\n" + "=" * 50)
        print("Test Summary:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success: {result.wasSuccessful()}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"\n{test}:")
                print(traceback)
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"\n{test}:")
                print(traceback)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_tests()
    
    if success:
        print("\n✅ All MAP-Elites tests passed!")
        print("The deterministic fixes are working correctly.")
    else:
        print("\n❌ Some tests failed.")
        print("Check the output above for details.")
    
    sys.exit(0 if success else 1)