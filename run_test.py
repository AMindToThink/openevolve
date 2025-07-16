#!/usr/bin/env python3

import sys
import unittest
from tests.test_map_elites_fix import TestMapElitesFix

if __name__ == "__main__":
    # Create a test suite with just the failing test
    test_suite = unittest.TestSuite()
    test_case = TestMapElitesFix('test_map_elites_replacement_basic')
    test_suite.addTest(test_case)
    
    # Run the test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print('\nTEST PASSED!')
        sys.exit(0)
    else:
        print('\nTEST FAILED!')
        for failure in result.failures:
            print(f'FAILURE: {failure[0]}')
            print(failure[1])
        for error in result.errors:
            print(f'ERROR: {error[0]}')
            print(error[1])
        sys.exit(1)