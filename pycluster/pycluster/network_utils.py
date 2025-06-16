"""
Network utilities for PyCluster
"""

import socket
import subprocess
import platform
import logging
import time
from typing import List, Dict, Optional, Tuple
import threading

logger = logging.getLogger(__name__)


class NetworkDiscovery:
    """
    Network discovery utilities for finding PyCluster nodes.
    """
    
    def __init__(self):
        self.is_windows = platform.system() == 'Windows'
        self.discovery_port = 8788  # Default discovery port
        self.discovery_timeout = 5
    
    def get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def scan_network_for_clusters(self, 
                                 network_range: str = None,
                                 ports: List[int] = None) -> List[Dict[str, str]]:
        """
        Scan the network for PyCluster head nodes.
        
        Args:
            network_range: Network range to scan (e.g., "192.168.1.0/24")
            ports: List of ports to check (default: [8786, 8787])
        
        Returns:
            List of discovered clusters
        """
        if ports is None:
            ports = [8786, 8787]  # Scheduler and dashboard ports
        
        if network_range is None:
            # Auto-detect network range
            local_ip = self.get_local_ip()
            network_range = '.'.join(local_ip.split('.')[:-1]) + '.0/24'
        
        discovered_clusters = []
        
        try:
            # Parse network range
            if '/' in network_range:
                base_ip, cidr = network_range.split('/')
                cidr = int(cidr)
            else:
                base_ip = network_range
                cidr = 24
            
            # Generate IP range
            base_parts = base_ip.split('.')
            base_int = (int(base_parts[0]) << 24) + (int(base_parts[1]) << 16) + \
                      (int(base_parts[2]) << 8) + int(base_parts[3])
            
            mask = (0xFFFFFFFF << (32 - cidr)) & 0xFFFFFFFF
            network = base_int & mask
            
            # Scan range (limit to reasonable size)
            max_hosts = min(2**(32-cidr), 254)
            
            for i in range(1, max_hosts):
                ip_int = network + i
                ip = f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"
                
                for port in ports:
                    if self._check_port_open(ip, port, timeout=1):
                        cluster_info = self._probe_cluster(ip, port)
                        if cluster_info:
                            discovered_clusters.append(cluster_info)
                        break  # Found one port, move to next IP
        
        except Exception as e:
            logger.error(f"Network scan failed: {e}")
        
        return discovered_clusters
    
    def _check_port_open(self, host: str, port: int, timeout: float = 1) -> bool:
        """Check if a port is open on a host."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                return result == 0
        except Exception:
            return False
    
    def _probe_cluster(self, host: str, port: int) -> Optional[Dict[str, str]]:
        """Probe a host to get cluster information."""
        try:
            # Try to get cluster info via HTTP (dashboard port)
            if port == 8787:
                import requests
                response = requests.get(f"http://{host}:{port}/json/identity.json", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'host': host,
                        'scheduler_port': '8786',  # Assume standard port
                        'dashboard_port': str(port),
                        'scheduler_address': f"tcp://{host}:8786",
                        'dashboard_url': f"http://{host}:{port}",
                        'cluster_id': data.get('id', 'unknown'),
                        'type': 'dask_cluster'
                    }
        except Exception:
            pass
        
        # If it's a scheduler port, assume it's a cluster
        if port == 8786:
            return {
                'host': host,
                'scheduler_port': str(port),
                'dashboard_port': '8787',  # Assume standard port
                'scheduler_address': f"tcp://{host}:{port}",
                'dashboard_url': f"http://{host}:8787",
                'cluster_id': 'unknown',
                'type': 'scheduler_port'
            }
        
        return None
    
    def start_discovery_service(self, cluster_info: Dict[str, str]) -> threading.Thread:
        """
        Start a discovery service that broadcasts cluster information.
        
        Args:
            cluster_info: Information about this cluster
        
        Returns:
            Thread running the discovery service
        """
        def discovery_server():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(('', self.discovery_port))
                    
                    logger.info(f"Discovery service started on port {self.discovery_port}")
                    
                    while True:
                        try:
                            data, addr = sock.recvfrom(1024)
                            if data == b'PYCLUSTER_DISCOVERY':
                                # Respond with cluster info
                                import json
                                response = json.dumps(cluster_info).encode()
                                sock.sendto(response, addr)
                                logger.debug(f"Responded to discovery request from {addr}")
                        except Exception as e:
                            logger.error(f"Discovery service error: {e}")
                            break
            
            except Exception as e:
                logger.error(f"Failed to start discovery service: {e}")
        
        thread = threading.Thread(target=discovery_server, daemon=True)
        thread.start()
        return thread
    
    def discover_clusters_broadcast(self, timeout: float = 5) -> List[Dict[str, str]]:
        """
        Discover clusters using UDP broadcast.
        
        Args:
            timeout: Discovery timeout in seconds
        
        Returns:
            List of discovered clusters
        """
        discovered = []
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(1)
                
                # Send broadcast discovery request
                message = b'PYCLUSTER_DISCOVERY'
                sock.sendto(message, ('<broadcast>', self.discovery_port))
                
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        data, addr = sock.recvfrom(1024)
                        import json
                        cluster_info = json.loads(data.decode())
                        cluster_info['discovered_from'] = addr[0]
                        discovered.append(cluster_info)
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.warning(f"Error parsing discovery response: {e}")
        
        except Exception as e:
            logger.error(f"Broadcast discovery failed: {e}")
        
        return discovered
    
    def test_connection(self, scheduler_address: str) -> Dict[str, bool]:
        """
        Test connection to a cluster.
        
        Args:
            scheduler_address: Scheduler address (e.g., "tcp://192.168.1.100:8786")
        
        Returns:
            Dictionary with connection test results
        """
        results = {
            'scheduler_reachable': False,
            'dashboard_reachable': False,
            'can_connect': False
        }
        
        try:
            # Parse scheduler address
            if scheduler_address.startswith('tcp://'):
                host_port = scheduler_address[6:]
                host, port = host_port.split(':')
                port = int(port)
            else:
                host, port = scheduler_address.split(':')
                port = int(port)
            
            # Test scheduler port
            results['scheduler_reachable'] = self._check_port_open(host, port, timeout=3)
            
            # Test dashboard port (assume +1 from scheduler)
            dashboard_port = port + 1
            results['dashboard_reachable'] = self._check_port_open(host, dashboard_port, timeout=3)
            
            # Test actual connection
            if results['scheduler_reachable']:
                try:
                    from dask.distributed import Client
                    client = Client(scheduler_address, timeout=5)
                    client.close()
                    results['can_connect'] = True
                except Exception as e:
                    logger.warning(f"Connection test failed: {e}")
        
        except Exception as e:
            logger.error(f"Connection test error: {e}")
        
        return results
    
    def get_recommended_ports(self, start_port: int = 8786) -> Dict[str, int]:
        """
        Get recommended ports for cluster components.
        
        Args:
            start_port: Starting port to check from
        
        Returns:
            Dictionary with recommended ports
        """
        ports = {}
        current_port = start_port
        
        # Find available ports
        for component in ['scheduler', 'dashboard', 'discovery']:
            while not self._check_port_available(current_port):
                current_port += 1
            ports[component] = current_port
            current_port += 1
        
        return ports
    
    def _check_port_available(self, port: int) -> bool:
        """Check if a port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('', port))
                return True
        except Exception:
            return False

