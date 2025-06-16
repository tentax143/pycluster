"""
Enhanced worker discovery and auto-join functionality for PyCluster
"""

import socket
import threading
import time
import json
import logging
import subprocess
import platform
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Information about a discovered cluster"""
    name: str
    scheduler_address: str
    dashboard_url: str
    host_ip: str
    port: int
    timestamp: float
    workers_count: int = 0
    status: str = "active"


class ClusterDiscovery:
    """
    Handles automatic discovery of PyCluster head nodes on the local network
    """
    
    def __init__(self, discovery_port: int = 8788):
        self.discovery_port = discovery_port
        self.broadcast_interval = 5.0  # seconds
        self.discovery_timeout = 30.0  # seconds
        self.running = False
        self.discovered_clusters: Dict[str, ClusterInfo] = {}
        self._broadcast_thread = None
        self._discovery_thread = None
        self._lock = threading.Lock()
    
    def start_broadcasting(self, cluster_info: ClusterInfo):
        """Start broadcasting cluster information for discovery"""
        self.cluster_info = cluster_info
        self.running = True
        
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            daemon=True
        )
        self._broadcast_thread.start()
        logger.info(f"Started cluster broadcasting on port {self.discovery_port}")
    
    def start_discovery(self):
        """Start listening for cluster broadcasts"""
        self.running = True
        
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop,
            daemon=True
        )
        self._discovery_thread.start()
        logger.info(f"Started cluster discovery on port {self.discovery_port}")
    
    def stop(self):
        """Stop broadcasting and discovery"""
        self.running = False
        if self._broadcast_thread:
            self._broadcast_thread.join(timeout=1.0)
        if self._discovery_thread:
            self._discovery_thread.join(timeout=1.0)
        logger.info("Stopped cluster discovery")
    
    def _broadcast_loop(self):
        """Broadcast cluster information periodically"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            # Windows-specific socket options
            if platform.system() == "Windows":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            broadcast_data = {
                "type": "pycluster_announcement",
                "cluster_name": self.cluster_info.name,
                "scheduler_address": self.cluster_info.scheduler_address,
                "dashboard_url": self.cluster_info.dashboard_url,
                "host_ip": self.cluster_info.host_ip,
                "timestamp": time.time(),
                "workers_count": self.cluster_info.workers_count,
                "status": self.cluster_info.status
            }
            
            message = json.dumps(broadcast_data).encode('utf-8')
            
            while self.running:
                try:
                    # Broadcast to local network
                    sock.sendto(message, ('<broadcast>', self.discovery_port))
                    
                    # Also try specific broadcast addresses
                    for broadcast_addr in self._get_broadcast_addresses():
                        try:
                            sock.sendto(message, (broadcast_addr, self.discovery_port))
                        except Exception:
                            pass  # Ignore individual broadcast failures
                    
                    time.sleep(self.broadcast_interval)
                    
                except Exception as e:
                    logger.warning(f"Broadcast error: {e}")
                    time.sleep(self.broadcast_interval)
            
            sock.close()
            
        except Exception as e:
            logger.error(f"Failed to start broadcasting: {e}")
    
    def _discovery_loop(self):
        """Listen for cluster broadcasts"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Windows-specific socket options
            if platform.system() == "Windows":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            sock.bind(('', self.discovery_port))
            sock.settimeout(1.0)  # Non-blocking with timeout
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)
                    message = json.loads(data.decode('utf-8'))
                    
                    if message.get("type") == "pycluster_announcement":
                        cluster_info = ClusterInfo(
                            name=message["cluster_name"],
                            scheduler_address=message["scheduler_address"],
                            dashboard_url=message["dashboard_url"],
                            host_ip=message["host_ip"],
                            port=self.discovery_port,
                            timestamp=message["timestamp"],
                            workers_count=message.get("workers_count", 0),
                            status=message.get("status", "active")
                        )
                        
                        with self._lock:
                            self.discovered_clusters[cluster_info.name] = cluster_info
                        
                        logger.debug(f"Discovered cluster: {cluster_info.name} at {cluster_info.scheduler_address}")
                
                except socket.timeout:
                    # Clean up old discoveries
                    self._cleanup_old_discoveries()
                    continue
                except Exception as e:
                    logger.debug(f"Discovery error: {e}")
                    continue
            
            sock.close()
            
        except Exception as e:
            logger.error(f"Failed to start discovery: {e}")
    
    def _get_broadcast_addresses(self) -> List[str]:
        """Get list of broadcast addresses for local network interfaces"""
        broadcast_addrs = []
        
        try:
            if platform.system() == "Windows":
                # Use ipconfig on Windows
                result = subprocess.run(['ipconfig'], capture_output=True, text=True)
                # Parse output to find network interfaces and calculate broadcast addresses
                # This is a simplified approach - in production, you might want more robust parsing
                broadcast_addrs.extend(['192.168.1.255', '192.168.0.255', '10.0.0.255'])
            else:
                # Use ip or ifconfig on Unix-like systems
                try:
                    result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
                    # Parse routing table to find broadcast addresses
                except FileNotFoundError:
                    result = subprocess.run(['route', '-n'], capture_output=True, text=True)
                
                # Fallback to common broadcast addresses
                broadcast_addrs.extend(['192.168.1.255', '192.168.0.255', '10.0.0.255'])
        
        except Exception as e:
            logger.debug(f"Could not determine broadcast addresses: {e}")
            # Fallback to common broadcast addresses
            broadcast_addrs = ['192.168.1.255', '192.168.0.255', '10.0.0.255']
        
        return broadcast_addrs
    
    def _cleanup_old_discoveries(self):
        """Remove old cluster discoveries"""
        current_time = time.time()
        with self._lock:
            expired_clusters = [
                name for name, info in self.discovered_clusters.items()
                if current_time - info.timestamp > self.discovery_timeout
            ]
            for name in expired_clusters:
                del self.discovered_clusters[name]
                logger.debug(f"Removed expired cluster: {name}")
    
    def get_discovered_clusters(self) -> Dict[str, ClusterInfo]:
        """Get currently discovered clusters"""
        with self._lock:
            return self.discovered_clusters.copy()
    
    def find_cluster_by_name(self, name: str) -> Optional[ClusterInfo]:
        """Find a specific cluster by name"""
        with self._lock:
            return self.discovered_clusters.get(name)


class EasyWorkerJoin:
    """
    Simplified worker joining with multiple discovery methods
    """
    
    def __init__(self):
        self.discovery = ClusterDiscovery()
    
    def discover_clusters(self, timeout: float = 10.0) -> List[ClusterInfo]:
        """
        Discover available clusters on the network
        
        Args:
            timeout: How long to search for clusters
            
        Returns:
            List of discovered clusters
        """
        logger.info("Searching for PyCluster head nodes...")
        
        self.discovery.start_discovery()
        time.sleep(timeout)
        self.discovery.stop()
        
        clusters = list(self.discovery.get_discovered_clusters().values())
        logger.info(f"Found {len(clusters)} cluster(s)")
        
        return clusters
    
    def join_cluster_interactive(self) -> Optional[str]:
        """
        Interactive cluster joining with user selection
        
        Returns:
            Scheduler address of selected cluster, or None if cancelled
        """
        clusters = self.discover_clusters()
        
        if not clusters:
            print("❌ No PyCluster head nodes found on the network.")
            print("\nTroubleshooting:")
            print("1. Ensure the head node is running")
            print("2. Check Windows Firewall settings")
            print("3. Verify you're on the same network")
            print("4. Try manual connection with scheduler address")
            return None
        
        print(f"\n🔍 Found {len(clusters)} PyCluster cluster(s):")
        print("=" * 60)
        
        for i, cluster in enumerate(clusters, 1):
            print(f"{i}. {cluster.name}")
            print(f"   Scheduler: {cluster.scheduler_address}")
            print(f"   Dashboard: {cluster.dashboard_url}")
            print(f"   Workers: {cluster.workers_count}")
            print(f"   Status: {cluster.status}")
            print()
        
        while True:
            try:
                choice = input(f"Select cluster (1-{len(clusters)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                cluster_idx = int(choice) - 1
                if 0 <= cluster_idx < len(clusters):
                    selected_cluster = clusters[cluster_idx]
                    print(f"✅ Selected: {selected_cluster.name}")
                    return selected_cluster.scheduler_address
                else:
                    print(f"❌ Please enter a number between 1 and {len(clusters)}")
            
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n❌ Cancelled by user")
                return None
    
    def join_cluster_by_name(self, cluster_name: str, timeout: float = 10.0) -> Optional[str]:
        """
        Join a specific cluster by name
        
        Args:
            cluster_name: Name of the cluster to join
            timeout: How long to search for the cluster
            
        Returns:
            Scheduler address if found, None otherwise
        """
        logger.info(f"Searching for cluster: {cluster_name}")
        
        self.discovery.start_discovery()
        
        # Search for the specific cluster
        start_time = time.time()
        while time.time() - start_time < timeout:
            cluster = self.discovery.find_cluster_by_name(cluster_name)
            if cluster:
                self.discovery.stop()
                logger.info(f"Found cluster {cluster_name} at {cluster.scheduler_address}")
                return cluster.scheduler_address
            time.sleep(0.5)
        
        self.discovery.stop()
        logger.warning(f"Cluster {cluster_name} not found within {timeout} seconds")
        return None
    
    def test_connection(self, scheduler_address: str) -> bool:
        """
        Test if we can connect to a scheduler
        
        Args:
            scheduler_address: Address to test (e.g., "tcp://192.168.1.100:8786")
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Extract host and port from scheduler address
            if scheduler_address.startswith("tcp://"):
                host_port = scheduler_address[6:]
                host, port = host_port.split(":")
                port = int(port)
            else:
                logger.error("Invalid scheduler address format")
                return False
            
            # Test TCP connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"✅ Connection test successful: {scheduler_address}")
                return True
            else:
                logger.warning(f"❌ Connection test failed: {scheduler_address}")
                return False
                
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            return False
    
    def get_cluster_info(self, scheduler_address: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a cluster
        
        Args:
            scheduler_address: Scheduler address
            
        Returns:
            Cluster information dictionary or None if failed
        """
        try:
            # Try to get info from dashboard API
            # Extract host and port, then construct dashboard URL
            if scheduler_address.startswith("tcp://"):
                host_port = scheduler_address[6:]
                host, port = host_port.split(":")
                
                # Assume dashboard is on port 8787 (default)
                dashboard_url = f"http://{host}:8787"
                
                # Try to get cluster info from dashboard
                response = requests.get(f"{dashboard_url}/json/identity.json", timeout=5)
                if response.status_code == 200:
                    return response.json()
            
        except Exception as e:
            logger.debug(f"Could not get cluster info: {e}")
        
        return None


def create_worker_join_script():
    """
    Create a simple script for easy worker joining
    """
    script_content = '''#!/usr/bin/env python3
"""
PyCluster Easy Worker Join Script

This script makes it easy to join a PyCluster as a worker node.
It will automatically discover available clusters on your network.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pycluster.worker_discovery import EasyWorkerJoin
from pycluster.cli_enhanced import start_worker_node
import argparse

def main():
    parser = argparse.ArgumentParser(description="Easy PyCluster Worker Join")
    parser.add_argument("--cluster-name", help="Specific cluster name to join")
    parser.add_argument("--scheduler", help="Direct scheduler address (e.g., tcp://192.168.1.100:8786)")
    parser.add_argument("--auto", action="store_true", help="Auto-join first available cluster")
    parser.add_argument("--list", action="store_true", help="List available clusters and exit")
    parser.add_argument("--timeout", type=float, default=10.0, help="Discovery timeout in seconds")
    
    args = parser.parse_args()
    
    easy_join = EasyWorkerJoin()
    
    if args.list:
        # Just list available clusters
        clusters = easy_join.discover_clusters(timeout=args.timeout)
        if clusters:
            print(f"Found {len(clusters)} cluster(s):")
            for cluster in clusters:
                print(f"  - {cluster.name}: {cluster.scheduler_address}")
        else:
            print("No clusters found")
        return
    
    scheduler_address = None
    
    if args.scheduler:
        # Direct scheduler address provided
        scheduler_address = args.scheduler
        print(f"Using provided scheduler: {scheduler_address}")
        
    elif args.cluster_name:
        # Join specific cluster by name
        scheduler_address = easy_join.join_cluster_by_name(args.cluster_name, args.timeout)
        if not scheduler_address:
            print(f"❌ Cluster '{args.cluster_name}' not found")
            sys.exit(1)
            
    elif args.auto:
        # Auto-join first available cluster
        clusters = easy_join.discover_clusters(timeout=args.timeout)
        if clusters:
            scheduler_address = clusters[0].scheduler_address
            print(f"✅ Auto-joining cluster: {clusters[0].name}")
        else:
            print("❌ No clusters found for auto-join")
            sys.exit(1)
            
    else:
        # Interactive selection
        scheduler_address = easy_join.join_cluster_interactive()
        if not scheduler_address:
            print("❌ No cluster selected")
            sys.exit(1)
    
    # Test connection before starting worker
    if not easy_join.test_connection(scheduler_address):
        print(f"❌ Cannot connect to scheduler: {scheduler_address}")
        print("Please check:")
        print("1. Head node is running")
        print("2. Network connectivity")
        print("3. Firewall settings")
        sys.exit(1)
    
    print(f"🚀 Starting worker, connecting to: {scheduler_address}")
    
    # Create args object for start_worker_node
    class WorkerArgs:
        def __init__(self, scheduler_addr):
            self.scheduler = scheduler_addr
            self.nthreads = None
            self.memory_limit = "auto"
    
    worker_args = WorkerArgs(scheduler_address)
    
    try:
        start_worker_node(worker_args)
    except KeyboardInterrupt:
        print("\\n✅ Worker stopped by user")
    except Exception as e:
        print(f"❌ Worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    return script_content

