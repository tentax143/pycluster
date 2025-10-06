#!/usr/bin/env python3
"""
Phase 1 Worker Comprehensive Test
Tests all worker functionality including discovery, connection, and task execution
"""

import asyncio
import sys
import os
import time
import socket
import threading

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

from pycluster.node import WorkerNode
from pycluster.worker_discovery import EasyWorkerJoin

class Phase1WorkerTest:
    def __init__(self, scheduler_address=None):
        self.worker_node = None
        self.scheduler_address = scheduler_address
        self.test_results = {}
        
    def log(self, message, status="INFO"):
        """Log test messages"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{status}] {message}")
        
    def test_worker_discovery(self):
        """Test 1: Worker discovery functionality"""
        self.log("=== Test 1: Worker Discovery ===")
        
        try:
            easy_join = EasyWorkerJoin()
            
            # Test cluster discovery
            self.log("Searching for clusters on network...")
            clusters = easy_join.discover_clusters(timeout=10.0)
            
            if clusters:
                self.log(f"Found {len(clusters)} cluster(s)", "PASS")
                for i, cluster in enumerate(clusters, 1):
                    self.log(f"  {i}. {cluster.name} - {cluster.scheduler_address}")
                
                # Use the first cluster found
                self.scheduler_address = clusters[0].scheduler_address
                self.test_results["worker_discovery"] = True
                return True
            else:
                self.log("No clusters found on network", "WARN")
                self.log("This is expected if no head node is running", "INFO")
                
                # Use provided scheduler address or default
                if not self.scheduler_address:
                    self.scheduler_address = "tcp://172.16.71.183:8786"
                    self.log(f"Using default scheduler address: {self.scheduler_address}", "INFO")
                else:
                    self.log(f"Using provided scheduler address: {self.scheduler_address}", "INFO")
                
                self.test_results["worker_discovery"] = False
                return False
                
        except Exception as e:
            self.log(f"Worker discovery failed: {e}", "FAIL")
            self.test_results["worker_discovery"] = False
            return False
    
    def test_connection_validation(self):
        """Test 2: Connection validation"""
        self.log("=== Test 2: Connection Validation ===")
        
        try:
            if not self.scheduler_address:
                self.log("No scheduler address available", "FAIL")
                self.test_results["connection_validation"] = False
                return False
            
            easy_join = EasyWorkerJoin()
            
            if easy_join.test_connection(self.scheduler_address):
                self.log("Connection test successful", "PASS")
                self.test_results["connection_validation"] = True
                return True
            else:
                self.log("Connection test failed", "FAIL")
                self.test_results["connection_validation"] = False
                return False
                
        except Exception as e:
            self.log(f"Connection validation error: {e}", "FAIL")
            self.test_results["connection_validation"] = False
            return False
    
    def test_dask_client_connection(self):
        """Test 3: Dask client connection"""
        self.log("=== Test 3: Dask Client Connection ===")
        
        try:
            from dask.distributed import Client
            
            self.log(f"Connecting to {self.scheduler_address}...")
            client = Client(self.scheduler_address, timeout=10)
            
            # Test basic operations
            info = client.scheduler_info()
            self.log("Dask client connected successfully", "PASS")
            self.log(f"  Scheduler ID: {info.get('id', 'unknown')}")
            self.log(f"  Current workers: {info.get('n_workers', 0)}")
            
            # Test task execution
            future = client.submit(lambda x: x * 3, 14)
            result = future.result(timeout=10)
            self.log(f"Task execution test: 14 * 3 = {result}", "PASS")
            
            client.close()
            self.test_results["dask_client"] = True
            return True
            
        except Exception as e:
            self.log(f"Dask client connection failed: {e}", "FAIL")
            self.test_results["dask_client"] = False
            return False
    
    async def test_worker_node_creation(self):
        """Test 4: Worker node creation"""
        self.log("=== Test 4: Worker Node Creation ===")
        
        try:
            self.worker_node = WorkerNode(
                scheduler_address=self.scheduler_address,
                worker_name="phase1-test-worker"
            )
            
            self.log("Worker node created successfully", "PASS")
            self.test_results["worker_creation"] = True
            return True
            
        except Exception as e:
            self.log(f"Worker node creation failed: {e}", "FAIL")
            self.test_results["worker_creation"] = False
            return False
    
    async def test_worker_connection(self):
        """Test 5: Worker connection"""
        self.log("=== Test 5: Worker Connection ===")
        
        try:
            if not self.worker_node:
                self.log("Worker node not created", "FAIL")
                self.test_results["worker_connection"] = False
                return False
            
            self.log("Starting worker connection...")
            result = await self.worker_node.start(
                n_workers=1,
                threads_per_worker=2,
                memory_limit="1GB"
            )
            
            if result['status'] == 'success':
                self.log("Worker connected successfully", "PASS")
                self.test_results["worker_connection"] = True
                return True
            else:
                self.log(f"Worker connection failed: {result}", "FAIL")
                self.test_results["worker_connection"] = False
                return False
                
        except Exception as e:
            self.log(f"Worker connection error: {e}", "FAIL")
            self.test_results["worker_connection"] = False
            return False
    
    def test_worker_status(self):
        """Test 6: Worker status"""
        self.log("=== Test 6: Worker Status ===")
        
        try:
            if not self.worker_node:
                self.log("Worker node not available", "FAIL")
                self.test_results["worker_status"] = False
                return False
            
            status = self.worker_node.get_status()
            
            if status.get('status') == 'running':
                self.log("Worker status healthy", "PASS")
                self.log(f"  Worker name: {status.get('worker_name', 'unknown')}")
                self.log(f"  Worker count: {status.get('worker_count', 0)}")
                self.log(f"  Scheduler: {status.get('scheduler_address', 'unknown')}")
                self.test_results["worker_status"] = True
                return True
            else:
                self.log(f"Worker status unhealthy: {status}", "FAIL")
                self.test_results["worker_status"] = False
                return False
                
        except Exception as e:
            self.log(f"Worker status error: {e}", "FAIL")
            self.test_results["worker_status"] = False
            return False
    
    def test_task_execution(self):
        """Test 7: Task execution on worker"""
        self.log("=== Test 7: Task Execution ===")
        
        try:
            from dask.distributed import Client
            
            client = Client(self.scheduler_address, timeout=10)
            
            # Test multiple tasks
            tasks = [
                (lambda x: x * 2, 21),
                (lambda x: x + 10, 5),
                (lambda x: x ** 2, 4),
                (lambda x: x / 2, 20)
            ]
            
            results = []
            for func, arg in tasks:
                future = client.submit(func, arg)
                result = future.result(timeout=10)
                results.append(result)
                self.log(f"Task {func.__name__}({arg}) = {result}", "PASS")
            
            # Test parallel execution
            futures = [client.submit(lambda x: x * i, 10) for i in range(1, 6)]
            parallel_results = [f.result(timeout=10) for f in futures]
            self.log(f"Parallel execution: {parallel_results}", "PASS")
            
            client.close()
            self.test_results["task_execution"] = True
            return True
            
        except Exception as e:
            self.log(f"Task execution failed: {e}", "FAIL")
            self.test_results["task_execution"] = False
            return False
    
    def test_cluster_info(self):
        """Test 8: Cluster information"""
        self.log("=== Test 8: Cluster Information ===")
        
        try:
            from dask.distributed import Client
            
            client = Client(self.scheduler_address, timeout=10)
            info = client.scheduler_info()
            
            self.log("Cluster information retrieved", "PASS")
            self.log(f"  Scheduler: {info.get('id', 'unknown')}")
            self.log(f"  Workers: {info.get('n_workers', 0)}")
            self.log(f"  Total threads: {info.get('total_threads', 0)}")
            self.log(f"  Total memory: {info.get('total_memory', 0) / (1024**3):.1f} GB")
            
            # List workers
            workers = info.get('workers', {})
            if workers:
                self.log("  Worker details:")
                for worker_addr, worker_info in workers.items():
                    self.log(f"    - {worker_info.get('name', 'unknown')}: {worker_addr}")
            
            client.close()
            self.test_results["cluster_info"] = True
            return True
            
        except Exception as e:
            self.log(f"Cluster info retrieval failed: {e}", "FAIL")
            self.test_results["cluster_info"] = False
            return False
    
    def test_llm_functionality(self):
        """Test 9: LLM functionality (if available)"""
        self.log("=== Test 9: LLM Functionality ===")
        
        try:
            # Check if LLM dependencies are available
            try:
                import torch
                import transformers
                llm_available = True
                self.log("LLM dependencies available", "PASS")
            except ImportError:
                llm_available = False
                self.log("LLM dependencies not available", "WARN")
            
            if llm_available:
                from pycluster.llm_serving import LLMClusterManager
                
                # Test LLM manager creation
                if self.worker_node and self.worker_node.cluster_manager:
                    llm_manager = LLMClusterManager(self.worker_node.cluster_manager)
                    self.log("LLM manager created successfully", "PASS")
                    self.test_results["llm_functionality"] = True
                    return True
                else:
                    self.log("Cluster manager not available for LLM", "FAIL")
                    self.test_results["llm_functionality"] = False
                    return False
            else:
                self.log("LLM functionality test skipped (dependencies not available)", "SKIP")
                self.test_results["llm_functionality"] = None
                return True
                
        except Exception as e:
            self.log(f"LLM functionality test failed: {e}", "FAIL")
            self.test_results["llm_functionality"] = False
            return False
    
    def print_summary(self):
        """Print test summary"""
        self.log("=== PHASE 1 WORKER TEST SUMMARY ===")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result is True)
        skipped_tests = sum(1 for result in self.test_results.values() if result is None)
        failed_tests = total_tests - passed_tests - skipped_tests
        
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {passed_tests}")
        self.log(f"Failed: {failed_tests}")
        self.log(f"Skipped: {skipped_tests}")
        
        self.log("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            if result is True:
                status = "PASS"
            elif result is False:
                status = "FAIL"
            else:
                status = "SKIP"
            self.log(f"  {test_name}: {status}")
        
        if failed_tests == 0:
            self.log("\n*** ALL TESTS PASSED! ***", "PASS")
            return True
        else:
            self.log(f"\n*** {failed_tests} TESTS FAILED ***", "FAIL")
            return False
    
    async def run_all_tests(self):
        """Run all worker tests"""
        self.log("Starting Phase 1 Worker Tests...")
        self.log("=" * 50)
        
        # Run tests in order
        tests = [
            self.test_worker_discovery,
            self.test_connection_validation,
            self.test_dask_client_connection,
            self.test_worker_node_creation,
            self.test_worker_connection,
            self.test_worker_status,
            self.test_task_execution,
            self.test_cluster_info,
            self.test_llm_functionality
        ]
        
        for test in tests:
            try:
                if asyncio.iscoroutinefunction(test):
                    await test()
                else:
                    test()
            except Exception as e:
                self.log(f"Test {test.__name__} crashed: {e}", "FAIL")
                self.test_results[test.__name__] = False
            
            time.sleep(1)  # Brief pause between tests
        
        return self.print_summary()

async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PyCluster Phase 1 Worker Test")
    parser.add_argument("--scheduler", help="Scheduler address (e.g., tcp://192.168.1.100:8786)")
    args = parser.parse_args()
    
    print("PyCluster Phase 1 Worker Comprehensive Test")
    print("=" * 60)
    
    tester = Phase1WorkerTest(scheduler_address=args.scheduler)
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            print("\n*** PHASE 1 WORKER TESTS COMPLETED SUCCESSFULLY! ***")
            print("\nThe worker functionality is working correctly.")
            print("Phase 1 fixes have been successfully implemented!")
        else:
            print("\n*** SOME TESTS FAILED ***")
            print("Please check the failed tests and fix any issues.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n*** Tests interrupted by user ***")
        if tester.worker_node:
            tester.worker_node.shutdown()
    except Exception as e:
        print(f"\n*** Test suite crashed: {e} ***")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
