#!/usr/bin/env python3
"""
Test runner for the Premier League MLOps System.
"""

import importlib
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_tests():
    """Run all tests in the tests directory."""
    print("🧪 Running tests...")
    
    # Check if tests directory exists
    tests_dir = Path(__file__).parent.parent / "tests"
    if not tests_dir.exists():
        print(f"❌ Tests directory not found: {tests_dir}")
        return False
    
    # Find all test files
    test_files = list(tests_dir.glob("test_*.py"))
    if not test_files:
        print("❌ No test files found in tests directory.")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add tests from each file
    for test_file in test_files:
        print(f"📄 Loading tests from: {test_file.name}")
        
        # Import the module
        module_name = f"tests.{test_file.stem}"
        try:
            module = importlib.import_module(module_name)
            
            # Add all tests from the module
            tests = unittest.defaultTestLoader.loadTestsFromModule(module)
            test_suite.addTest(tests)
        except Exception as e:
            print(f"❌ Error loading tests from {test_file.name}: {e}")
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Print summary
    print("\n🔍 Test Summary:")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 