#!/usr/bin/env python3
"""
Phase 1 Complete Test Suite
Runs all Phase 1 tests in sequence to verify all fixes
"""

import asyncio
import sys
import os
import time
import subprocess

def log(message, status="INFO"):
    """Log test messages"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{status}] {message}")

def run_test_script(script_name, description):
    """Run a test script and return success status"""
    log(f"=== {description} ===")
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            log(f"{description} completed successfully", "PASS")
            return True
        else:
            log(f"{description} failed with return code {result.returncode}", "FAIL")
            return False
            
    except subprocess.TimeoutExpired:
        log(f"{description} timed out after 5 minutes", "FAIL")
        return False
    except Exception as e:
        log(f"{description} crashed: {e}", "FAIL")
        return False

async def main():
    """Main test runner"""
    print("PyCluster Phase 1 Complete Test Suite")
    print("=" * 60)
    print("This will test all Phase 1 fixes:")
    print("1. LLM Serving Implementation")
    print("2. Worker Discovery")
    print("3. API Integration")
    print("4. Head Node Functionality")
    print("5. Worker Functionality")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Head Node Tests
    log("Starting Phase 1 Head Node Tests...")
    test_results["head_node"] = run_test_script("phase1_head_test.py", "Head Node Tests")
    
    if not test_results["head_node"]:
        log("Head node tests failed. Worker tests may not work properly.", "WARN")
    
    # Brief pause between test suites
    time.sleep(2)
    
    # Test 2: Worker Tests
    log("Starting Phase 1 Worker Tests...")
    test_results["worker"] = run_test_script("phase1_worker_test.py", "Worker Tests")
    
    # Test 3: API Tests (if available)
    log("Checking for API tests...")
    if os.path.exists("test_api.py"):
        test_results["api"] = run_test_script("test_api.py", "API Integration Tests")
    else:
        log("API tests not found, skipping", "SKIP")
        test_results["api"] = None
    
    # Print final summary
    log("=== PHASE 1 COMPLETE TEST SUMMARY ===")
    
    total_tests = len([k for k, v in test_results.items() if v is not None])
    passed_tests = sum(1 for v in test_results.values() if v is True)
    failed_tests = sum(1 for v in test_results.values() if v is False)
    skipped_tests = sum(1 for v in test_results.values() if v is None)
    
    log(f"Total Test Suites: {total_tests}")
    log(f"Passed: {passed_tests}")
    log(f"Failed: {failed_tests}")
    log(f"Skipped: {skipped_tests}")
    
    log("\nDetailed Results:")
    for test_name, result in test_results.items():
        if result is True:
            status = "PASS"
        elif result is False:
            status = "FAIL"
        else:
            status = "SKIP"
        log(f"  {test_name}: {status}")
    
    if failed_tests == 0:
        log("\n*** ALL PHASE 1 TESTS PASSED! ***", "PASS")
        log("Phase 1 fixes have been successfully implemented and tested.", "PASS")
        log("PyCluster is now significantly more functional than before.", "PASS")
        log("\nNext steps:", "INFO")
        log("1. Test the API integration (if not already done)", "INFO")
        log("2. Try deploying and running LLM models", "INFO")
        log("3. Test multi-worker scenarios", "INFO")
        log("4. Proceed to Phase 2 improvements", "INFO")
        return True
    else:
        log(f"\n*** {failed_tests} TEST SUITES FAILED ***", "FAIL")
        log("Please check the failed tests and fix any issues.", "FAIL")
        log("Phase 1 fixes may not be complete.", "FAIL")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        log("\n*** Tests interrupted by user ***", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"\n*** Test suite crashed: {e} ***", "FAIL")
        sys.exit(1)
