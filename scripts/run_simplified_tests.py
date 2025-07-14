#!/usr/bin/env python3
"""
Simple test runner for the simplified Premier League MLOps System.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests():
    """Run all tests in the tests_simplified directory."""
    print("🧪 Running simplified tests...")
    
    # Check if tests_simplified directory exists
    tests_dir = Path(__file__).parent.parent / "tests_simplified"
    if not tests_dir.exists():
        print("❌ Tests directory not found. Run cleanup.py first to create it.")
        return False
    
    # Find all test files
    test_files = [f for f in tests_dir.glob("test_*.py")]
    if not test_files:
        print("❌ No test files found in tests_simplified directory.")
        return False
    
    # Print test files
    print(f"📋 Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    
    # Run tests
    print("\n🚀 Running tests...")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests from each file
    for test_file in test_files:
        module_name = f"tests_simplified.{test_file.stem}"
        try:
            # Import the module dynamically
            test_module = __import__(module_name, fromlist=["*"])
            tests = loader.loadTestsFromModule(test_module)
            suite.addTests(tests)
        except ImportError as e:
            print(f"❌ Error importing {module_name}: {e}")
    
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"  - Total tests: {result.testsRun}")
    print(f"  - Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  - Failed: {len(result.failures)}")
    print(f"  - Errors: {len(result.errors)}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 