#!/usr/bin/env python3
"""Run all tests for the project"""

import subprocess

# Add parent directory to path

def run_test(test_file):
    """Run a single test file"""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, test_file],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )
    
    return result.returncode == 0

def main():
    """Run all tests"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = [
        'test_basic_sted.py',
        'test_dataset_analysis.py',
        'test_variation_progression.py',
        'test_llm_results.py'
    ]
    
    print("="*60)
    print("FIELD-AWARE CONSISTENCY EVALUATION FRAMEWORK - TEST SUITE")
    print("="*60)
    
    results = {}
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            results[test_file] = run_test(test_path)
        else:
            print(f"⚠ Test file not found: {test_file}")
            results[test_file] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_file, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_file}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
