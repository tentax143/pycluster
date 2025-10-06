#!/usr/bin/env python3
"""
Phase 1 Coordinated Test
Runs head node test and waits for external worker to join
"""

import asyncio
import sys
import os
import time
import subprocess
import threading

def log(message, status="INFO"):
    """Log test messages"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{status}] {message}")

def run_head_node_test():
    """Run head node test in a separate process"""
    log("Starting head node test...")
    
    try:
        # Run the head node test
        process = subprocess.Popen([sys.executable, "phase1_head_test.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True,
                                 bufsize=1,
                                 universal_newlines=True)
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return process.returncode == 0
        
    except Exception as e:
        log(f"Head node test failed: {e}", "FAIL")
        return False

def run_worker_test(scheduler_address):
    """Run worker test with scheduler address"""
    log(f"Starting worker test with scheduler: {scheduler_address}")
    
    try:
        # Run the worker test with scheduler address
        result = subprocess.run([sys.executable, "phase1_worker_test.py", "--scheduler", scheduler_address],
                              capture_output=True,
                              text=True,
                              timeout=300)  # 5 minute timeout
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        log("Worker test timed out", "FAIL")
        return False
    except Exception as e:
        log(f"Worker test failed: {e}", "FAIL")
        return False

def get_scheduler_address():
    """Get scheduler address from discovery"""
    try:
        from pycluster.worker_discovery import EasyWorkerJoin
        
        easy_join = EasyWorkerJoin()
        clusters = easy_join.discover_clusters(timeout=10.0)
        
        if clusters:
            return clusters[0].scheduler_address
        else:
            return None
            
    except Exception as e:
        log(f"Discovery failed: {e}", "WARN")
        return None

async def main():
    """Main coordinated test function"""
    print("PyCluster Phase 1 Coordinated Test")
    print("=" * 50)
    print("This test will:")
    print("1. Start head node and wait for external workers")
    print("2. Run worker test in another terminal")
    print("3. Verify the worker joins the head node")
    print("=" * 50)
    
    # Check if we should run head node or worker test
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        # Run worker test
        scheduler_address = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not scheduler_address:
            log("Getting scheduler address via discovery...")
            scheduler_address = get_scheduler_address()
            
            if not scheduler_address:
                log("No scheduler found. Make sure head node is running.", "FAIL")
                sys.exit(1)
        
        log(f"Running worker test with scheduler: {scheduler_address}")
        success = run_worker_test(scheduler_address)
        
        if success:
            log("Worker test completed successfully!", "PASS")
        else:
            log("Worker test failed!", "FAIL")
            sys.exit(1)
    
    else:
        # Run head node test
        log("Running head node test...")
        log("The head node will wait for external workers to join.")
        log("In another terminal, run:")
        log("  python phase1_coordinated_test.py --worker")
        log("Or:")
        log("  python phase1_worker_test.py --scheduler <address>")
        log("")
        
        success = run_head_node_test()
        
        if success:
            log("Head node test completed successfully!", "PASS")
        else:
            log("Head node test failed!", "FAIL")
            sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Test interrupted by user", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"Test crashed: {e}", "FAIL")
        sys.exit(1)
