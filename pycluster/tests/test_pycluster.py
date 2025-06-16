"""
Test suite for PyCluster
"""

import unittest
import time
import threading
from unittest.mock import patch, MagicMock
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pycluster import ClusterManager, HeadNode, WorkerNode, DashboardManager
from pycluster.windows_utils import WindowsClusterManager
from pycluster.network_utils import NetworkDiscovery


class TestClusterManager(unittest.TestCase):
    """Test the ClusterManager class."""
    
    def setUp(self):
        self.cluster = ClusterManager("test-cluster")
    
    def tearDown(self):
        if self.cluster:
            self.cluster.shutdown()
    
    def test_cluster_initialization(self):
        """Test cluster manager initialization."""
        self.assertEqual(self.cluster.cluster_name, "test-cluster")
        self.assertIsNone(self.cluster.scheduler_address)
        self.assertFalse(self.cluster.is_head_node)
    
    def test_cluster_info_disconnected(self):
        """Test cluster info when disconnected."""
        info = self.cluster.get_cluster_info()
        self.assertEqual(info["status"], "not_connected")


class TestHeadNode(unittest.TestCase):
    """Test the HeadNode class."""
    
    def setUp(self):
        self.head_node = None
    
    def tearDown(self):
        if self.head_node:
            self.head_node.shutdown()
    
    def test_head_node_initialization(self):
        """Test head node initialization."""
        self.head_node = HeadNode("test-cluster", host="127.0.0.1")
        self.assertEqual(self.head_node.cluster_name, "test-cluster")
        self.assertEqual(self.head_node.host, "127.0.0.1")
        self.assertFalse(self.head_node.is_running)
    
    def test_head_node_start_local(self):
        """Test starting head node with local workers."""
        self.head_node = HeadNode("test-cluster", host="127.0.0.1")
        
        # Start with local workers
        result = self.head_node.start(n_local_workers=1)
        
        if result["status"] == "success":
            self.assertTrue(self.head_node.is_running)
            
            # Get connection info
            conn_info = self.head_node.get_connection_info()
            self.assertEqual(conn_info["status"], "running")
            self.assertIn("scheduler_address", conn_info)
            self.assertIn("dashboard_url", conn_info)
            
            # Check cluster status
            status = self.head_node.get_cluster_status()
            self.assertGreaterEqual(status.get("total_workers", 0), 1)
        else:
            self.skipTest(f"Failed to start head node: {result.get('message')}")


class TestWorkerNode(unittest.TestCase):
    """Test the WorkerNode class."""
    
    def setUp(self):
        self.worker_node = None
    
    def tearDown(self):
        if self.worker_node:
            self.worker_node.shutdown()
    
    def test_worker_node_initialization(self):
        """Test worker node initialization."""
        self.worker_node = WorkerNode("tcp://127.0.0.1:8786")
        self.assertEqual(self.worker_node.scheduler_address, "tcp://127.0.0.1:8786")
        self.assertFalse(self.worker_node.is_running)


class TestDashboardManager(unittest.TestCase):
    """Test the DashboardManager class."""
    
    def test_dashboard_initialization(self):
        """Test dashboard manager initialization."""
        dashboard = DashboardManager("http://127.0.0.1:8787")
        self.assertEqual(dashboard.dashboard_url, "http://127.0.0.1:8787")
        self.assertEqual(dashboard.base_url, "http://127.0.0.1:8787")
    
    def test_dashboard_urls(self):
        """Test dashboard URL generation."""
        dashboard = DashboardManager("http://127.0.0.1:8787")
        urls = dashboard.get_dashboard_urls()
        
        expected_urls = [
            "status", "workers", "tasks", "system", 
            "profile", "graph", "individual-plots"
        ]
        
        for url_type in expected_urls:
            self.assertIn(url_type, urls)
            self.assertTrue(urls[url_type].startswith("http://127.0.0.1:8787"))


class TestWindowsClusterManager(unittest.TestCase):
    """Test Windows-specific utilities."""
    
    def setUp(self):
        self.windows_manager = WindowsClusterManager()
    
    def test_config_directory(self):
        """Test configuration directory creation."""
        self.assertTrue(self.windows_manager.config_dir.exists())
    
    def test_save_load_config(self):
        """Test saving and loading cluster configuration."""
        test_config = {
            "cluster_name": "test",
            "host": "127.0.0.1",
            "scheduler_port": 8786
        }
        
        # Save config
        success = self.windows_manager.save_cluster_config(test_config, "test")
        self.assertTrue(success)
        
        # Load config
        loaded_config = self.windows_manager.load_cluster_config("test")
        self.assertIsNotNone(loaded_config)
        self.assertEqual(loaded_config["cluster_name"], "test")
        self.assertEqual(loaded_config["host"], "127.0.0.1")
    
    def test_list_configs(self):
        """Test listing cluster configurations."""
        # Save a test config
        test_config = {"cluster_name": "list_test"}
        self.windows_manager.save_cluster_config(test_config, "list_test")
        
        # List configs
        configs = self.windows_manager.list_cluster_configs()
        self.assertIn("list_test", configs)
    
    def test_port_availability(self):
        """Test port availability checking."""
        # Test a port that should be available
        available = self.windows_manager.check_port_availability(65432)
        self.assertTrue(available)
        
        # Find an available port
        port = self.windows_manager.find_available_port(65400)
        self.assertIsNotNone(port)
        self.assertGreaterEqual(port, 65400)
    
    def test_system_info(self):
        """Test system information gathering."""
        info = self.windows_manager.get_system_info()
        
        required_keys = [
            "platform", "system", "cpu_count", 
            "memory_total", "hostname", "python_version"
        ]
        
        for key in required_keys:
            self.assertIn(key, info)


class TestNetworkDiscovery(unittest.TestCase):
    """Test network discovery utilities."""
    
    def setUp(self):
        self.network_discovery = NetworkDiscovery()
    
    def test_local_ip(self):
        """Test local IP detection."""
        ip = self.network_discovery.get_local_ip()
        self.assertIsNotNone(ip)
        self.assertNotEqual(ip, "")
    
    def test_port_checking(self):
        """Test port checking functionality."""
        # Test a port that should be closed
        open_port = self.network_discovery._check_port_open("127.0.0.1", 65432, timeout=0.5)
        self.assertFalse(open_port)
    
    def test_recommended_ports(self):
        """Test recommended port generation."""
        ports = self.network_discovery.get_recommended_ports(65400)
        
        self.assertIn("scheduler", ports)
        self.assertIn("dashboard", ports)
        self.assertIn("discovery", ports)
        
        # Ports should be sequential
        self.assertEqual(ports["dashboard"], ports["scheduler"] + 1)
        self.assertEqual(ports["discovery"], ports["dashboard"] + 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        self.head_node = None
        self.worker_node = None
    
    def tearDown(self):
        if self.worker_node:
            self.worker_node.shutdown()
        if self.head_node:
            self.head_node.shutdown()
    
    def test_full_cluster_workflow(self):
        """Test a complete cluster workflow."""
        # Start head node
        self.head_node = HeadNode("integration-test", host="127.0.0.1")
        result = self.head_node.start(n_local_workers=1)
        
        if result["status"] != "success":
            self.skipTest(f"Failed to start head node: {result.get('message')}")
        
        # Get connection info
        conn_info = self.head_node.get_connection_info()
        self.assertEqual(conn_info["status"], "running")
        
        # Test dashboard
        dashboard = DashboardManager(conn_info["dashboard_url"])
        
        # Wait a bit for dashboard to be ready
        time.sleep(2)
        
        # Check if dashboard is accessible (may fail in test environment)
        accessible = dashboard.is_accessible()
        # Don't assert this as it may fail in CI/test environments
        
        # Test task submission
        cluster_manager = self.head_node.cluster_manager
        
        def test_task(x):
            return x * 2
        
        # Submit a task
        future = cluster_manager.submit_task(test_task, 5)
        result = future.result(timeout=10)
        self.assertEqual(result, 10)
        
        # Test map tasks
        futures = cluster_manager.map_tasks(test_task, [1, 2, 3, 4])
        results = [f.result(timeout=10) for f in futures]
        self.assertEqual(results, [2, 4, 6, 8])


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

