#!/usr/bin/env python3
"""
Phase 1 Head Node Comprehensive Test
Tests all head node functionality including discovery broadcasting
"""

import asyncio
import sys
import os
import time
import socket
import threading

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

from pycluster.node import HeadNode

class Phase1HeadTest:
    def __init__(self):
        self.head_node = None
        self.test_results = {}
        
    def log(self, message, status="INFO"):
        """Log test messages"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{status}] {message}")
        
    async def test_head_node_creation(self):
        """Test 1: Head node creation"""
        self.log("=== Test 1: Head Node Creation ===")
        
        try:
            self.head_node = HeadNode(
                cluster_name='phase1-test-cluster',
                host='0.0.0.0',
                scheduler_port=8786,
                dashboard_port=8787
            )
            
            self.log("Head node created successfully", "PASS")
            self.test_results["head_creation"] = True
            return True
            
        except Exception as e:
            self.log(f"Head node creation failed: {e}", "FAIL")
            self.test_results["head_creation"] = False
            return False
    
    async def test_head_node_startup(self):
        """Test 2: Head node startup"""
        self.log("=== Test 2: Head Node Startup ===")
        
        try:
            result = await self.head_node.start(n_local_workers=1)
            
            if result['status'] == 'success':
                self.log("Head node started successfully", "PASS")
                self.log(f"  Scheduler: {result.get('scheduler_address', 'N/A')}")
                self.log(f"  Dashboard: {result.get('dashboard_address', 'N/A')}")
                self.test_results["head_startup"] = True
                return True
            else:
                self.log(f"Head node startup failed: {result}", "FAIL")
                self.test_results["head_startup"] = False
                return False
                
        except Exception as e:
            self.log(f"Head node startup error: {e}", "FAIL")
            self.test_results["head_startup"] = False
            return False
    
    def test_connection_info(self):
        """Test 3: Connection information"""
        self.log("=== Test 3: Connection Information ===")
        
        try:
            conn_info = self.head_node.get_connection_info()
            
            required_fields = ['cluster_name', 'scheduler_address', 'dashboard_url', 'host_ip']
            missing_fields = [field for field in required_fields if field not in conn_info]
            
            if not missing_fields:
                self.log("Connection info complete", "PASS")
                self.log(f"  Cluster: {conn_info['cluster_name']}")
                self.log(f"  Scheduler: {conn_info['scheduler_address']}")
                self.log(f"  Dashboard: {conn_info['dashboard_url']}")
                self.log(f"  Host IP: {conn_info['host_ip']}")
                self.test_results["connection_info"] = True
                return True
            else:
                self.log(f"Missing connection fields: {missing_fields}", "FAIL")
                self.test_results["connection_info"] = False
                return False
                
        except Exception as e:
            self.log(f"Connection info error: {e}", "FAIL")
            self.test_results["connection_info"] = False
            return False
    
    def test_cluster_status(self):
        """Test 4: Cluster status"""
        self.log("=== Test 4: Cluster Status ===")
        
        try:
            status = self.head_node.get_cluster_status()
            
            if status.get('status') == 'connected':
                workers = status.get('workers', [])
                self.log("Cluster status healthy", "PASS")
                self.log(f"  Workers: {len(workers)}")
                self.log(f"  Status: {status.get('status')}")
                self.test_results["cluster_status"] = True
                return True
            else:
                self.log(f"Cluster status unhealthy: {status}", "FAIL")
                self.test_results["cluster_status"] = False
                return False
                
        except Exception as e:
            self.log(f"Cluster status error: {e}", "FAIL")
            self.test_results["cluster_status"] = False
            return False
    
    def test_port_connectivity(self):
        """Test 5: Port connectivity"""
        self.log("=== Test 5: Port Connectivity ===")
        
        try:
            conn_info = self.head_node.get_connection_info()
            host = conn_info['host_ip']
            
            # Test scheduler port
            scheduler_port = 8786
            if self._test_port(host, scheduler_port):
                self.log(f"Scheduler port {scheduler_port} accessible", "PASS")
            else:
                self.log(f"Scheduler port {scheduler_port} not accessible", "FAIL")
                self.test_results["port_connectivity"] = False
                return False
            
            # Test dashboard port
            dashboard_port = 8787
            if self._test_port(host, dashboard_port):
                self.log(f"Dashboard port {dashboard_port} accessible", "PASS")
            else:
                self.log(f"Dashboard port {dashboard_port} not accessible", "FAIL")
                self.test_results["port_connectivity"] = False
                return False
            
            self.test_results["port_connectivity"] = True
            return True
            
        except Exception as e:
            self.log(f"Port connectivity error: {e}", "FAIL")
            self.test_results["port_connectivity"] = False
            return False
    
    def test_dask_client_connection(self):
        """Test 6: Dask client connection"""
        self.log("=== Test 6: Dask Client Connection ===")
        
        try:
            from dask.distributed import Client
            
            conn_info = self.head_node.get_connection_info()
            scheduler_address = conn_info['scheduler_address']
            
            client = Client(scheduler_address, timeout=10)
            
            # Test basic operations
            info = client.scheduler_info()
            self.log("Dask client connected successfully", "PASS")
            self.log(f"  Scheduler ID: {info.get('id', 'unknown')}")
            self.log(f"  Workers: {info.get('n_workers', 0)}")
            
            # Test task execution
            future = client.submit(lambda x: x * 2, 42)
            result = future.result(timeout=10)
            self.log(f"Task execution test: 42 * 2 = {result}", "PASS")
            
            client.close()
            self.test_results["dask_client"] = True
            return True
            
        except Exception as e:
            self.log(f"Dask client connection failed: {e}", "FAIL")
            self.test_results["dask_client"] = False
            return False
    
    def test_discovery_broadcasting(self):
        """Test 7: Discovery broadcasting"""
        self.log("=== Test 7: Discovery Broadcasting ===")
        
        try:
            if self.head_node.discovery:
                self.log("Discovery system initialized", "PASS")
                
                # Check if broadcasting is active
                if hasattr(self.head_node.discovery, 'running') and self.head_node.discovery.running:
                    self.log("Discovery broadcasting active", "PASS")
                    self.test_results["discovery_broadcasting"] = True
                    return True
                else:
                    self.log("Discovery broadcasting not active", "FAIL")
                    self.test_results["discovery_broadcasting"] = False
                    return False
            else:
                self.log("Discovery system not available", "WARN")
                self.test_results["discovery_broadcasting"] = False
                return False
                
        except Exception as e:
            self.log(f"Discovery broadcasting error: {e}", "FAIL")
            self.test_results["discovery_broadcasting"] = False
            return False
    
    def test_llm_manager_initialization(self):
        """Test 8: LLM manager initialization"""
        self.log("=== Test 8: LLM Manager Initialization ===")
        
        try:
            # Check if LLM serving is available
            from pycluster.llm_serving import LLMClusterManager
            
            if self.head_node.cluster_manager:
                llm_manager = LLMClusterManager(self.head_node.cluster_manager)
                self.log("LLM manager initialized successfully", "PASS")
                self.test_results["llm_manager"] = True
                return True
            else:
                self.log("Cluster manager not available for LLM", "FAIL")
                self.test_results["llm_manager"] = False
                return False
                
        except Exception as e:
            self.log(f"LLM manager initialization failed: {e}", "FAIL")
            self.test_results["llm_manager"] = False
            return False
    
    def _test_port(self, host, port, timeout=5):
        """Test if a port is accessible"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def print_summary(self):
        """Print test summary"""
        self.log("=== PHASE 1 HEAD NODE TEST SUMMARY ===")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {passed_tests}")
        self.log(f"Failed: {total_tests - passed_tests}")
        
        self.log("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            self.log(f"  {test_name}: {status}")
        
        if passed_tests == total_tests:
            self.log("\n*** ALL TESTS PASSED! ***", "PASS")
            return True
        else:
            self.log(f"\n*** {total_tests - passed_tests} TESTS FAILED ***", "FAIL")
            return False
    
    async def run_all_tests(self):
        """Run all head node tests"""
        self.log("Starting Phase 1 Head Node Tests...")
        self.log("=" * 50)
        
        # Run tests in order
        tests = [
            self.test_head_node_creation,
            self.test_head_node_startup,
            self.test_connection_info,
            self.test_cluster_status,
            self.test_port_connectivity,
            self.test_dask_client_connection,
            self.test_discovery_broadcasting,
            self.test_llm_manager_initialization
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
    print("PyCluster Phase 1 Head Node Comprehensive Test")
    print("=" * 60)
    
    tester = Phase1HeadTest()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            print("\n*** PHASE 1 HEAD NODE TESTS COMPLETED SUCCESSFULLY! ***")
            print("\nThe head node is ready for worker connections.")
            print("You can now run phase1_worker_test.py to test worker functionality.")
        else:
            print("\n*** SOME TESTS FAILED ***")
            print("Please check the failed tests and fix any issues.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n*** Tests interrupted by user ***")
        if tester.head_node:
            tester.head_node.shutdown()
    except Exception as e:
        print(f"\n*** Test suite crashed: {e} ***")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
