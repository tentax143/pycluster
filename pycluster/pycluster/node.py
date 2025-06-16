"""
Node management classes for PyCluster
"""

import os
import socket
import logging
from typing import Optional, Dict, Any
from .cluster import ClusterManager

logger = logging.getLogger(__name__)


class HeadNode:
    """
    Simplified interface for starting and managing a head node.
    """
    
    def __init__(self, 
                 cluster_name: str = "pycluster",
                 host: str = "0.0.0.0",
                 scheduler_port: int = 8786,
                 dashboard_port: int = 8787):
        """
        Initialize a head node.
        
        Args:
            cluster_name: Name of the cluster
            host: Host address to bind to
            scheduler_port: Port for the scheduler
            dashboard_port: Port for the dashboard
        """
        self.cluster_name = cluster_name
        self.host = host
        self.scheduler_port = scheduler_port
        self.dashboard_port = dashboard_port
        self.cluster_manager = ClusterManager(cluster_name)
        self.is_running = False
        
        # Initialize discovery for worker auto-join
        try:
            from .worker_discovery import ClusterDiscovery
            self.discovery = ClusterDiscovery()
        except ImportError:
            self.discovery = None
            logger.warning("Worker discovery not available")
    
    def start(self, n_local_workers: int = 0) -> Dict[str, Any]:
        """
        Start the head node.
        
        Args:
            n_local_workers: Number of local workers to start on this node
            
        Returns:
            Dictionary with startup information
        """
        if self.is_running:
            return {"status": "already_running"}
        
        result = self.cluster_manager.start_head_node(
            host=self.host,
            scheduler_port=self.scheduler_port,
            dashboard_port=self.dashboard_port,
            n_local_workers=n_local_workers
        )
        
        # Start broadcasting for discovery if successful
        if result["status"] == "success" and self.discovery:
            try:
                import time
                from .worker_discovery import ClusterInfo
                
                # Get actual IP for broadcasting
                actual_host = self.host
                if self.host == "0.0.0.0":
                    actual_host = socket.gethostbyname(socket.gethostname())
                
                cluster_info = ClusterInfo(
                    name=self.cluster_name,
                    scheduler_address=f"tcp://{actual_host}:{self.scheduler_port}",
                    dashboard_url=f"http://{actual_host}:{self.dashboard_port}",
                    host_ip=actual_host,
                    port=8788,  # Discovery port
                    timestamp=time.time(),
                    workers_count=n_local_workers,
                    status="active"
                )
                self.discovery.start_broadcasting(cluster_info)
                logger.info("Started cluster broadcasting for worker discovery")
            except Exception as e:
                logger.warning(f"Could not start discovery broadcasting: {e}")
        
        if result["status"] == "success":
            self.is_running = True
            logger.info(f"Head node '{self.cluster_name}' started successfully")
        
        return result
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information for workers to join this head node.
        
        Returns:
            Dictionary with connection details
        """
        if not self.is_running:
            return {"status": "not_running"}
        
        # Get the actual IP address if host is 0.0.0.0
        if self.host == "0.0.0.0":
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        else:
            local_ip = self.host
        
        return {
            "status": "running",
            "cluster_name": self.cluster_name,
            "scheduler_address": f"tcp://{local_ip}:{self.scheduler_port}",
            "dashboard_url": f"http://{local_ip}:{self.dashboard_port}",
            "host_ip": local_ip
        }
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get the current status of the cluster.
        
        Returns:
            Dictionary with cluster status
        """
        if not self.is_running:
            return {"status": "not_running"}
        
        return self.cluster_manager.get_cluster_info()
    
    def shutdown(self):
        """
        Shutdown the head node.
        """
        if self.is_running:
            # Stop discovery broadcasting
            if self.discovery:
                try:
                    self.discovery.stop()
                    logger.info("Stopped cluster broadcasting")
                except Exception as e:
                    logger.warning(f"Error stopping discovery: {e}")
            
            self.cluster_manager.shutdown()
            self.is_running = False
            logger.info(f"Head node '{self.cluster_name}' shutdown")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class WorkerNode:
    """
    Simplified interface for starting and managing worker nodes.
    """
    
    def __init__(self, scheduler_address: str, worker_name: Optional[str] = None):
        """
        Initialize a worker node.
        
        Args:
            scheduler_address: Address of the scheduler to connect to
            worker_name: Optional name for this worker
        """
        self.scheduler_address = scheduler_address
        self.worker_name = worker_name or f"worker-{socket.gethostname()}"
        self.cluster_manager = ClusterManager()
        self.is_running = False
    
    def start(self, 
              n_workers: int = 1,
              threads_per_worker: int = None,
              memory_limit: str = "auto") -> Dict[str, Any]:
        """
        Start the worker node(s).
        
        Args:
            n_workers: Number of workers to start
            threads_per_worker: Number of threads per worker
            memory_limit: Memory limit per worker
            
        Returns:
            Dictionary with startup information
        """
        if self.is_running:
            return {"status": "already_running"}
        
        result = self.cluster_manager.add_worker(
            scheduler_address=self.scheduler_address,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit
        )
        
        if result["status"] == "success":
            self.is_running = True
            logger.info(f"Worker node '{self.worker_name}' started successfully")
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of this worker node.
        
        Returns:
            Dictionary with worker status
        """
        if not self.is_running:
            return {"status": "not_running"}
        
        cluster_info = self.cluster_manager.get_cluster_info()
        
        # Filter to show only workers from this node
        my_workers = [
            worker for worker in cluster_info.get("workers", [])
            if self.worker_name in worker.get("id", "")
        ]
        
        return {
            "status": "running",
            "worker_name": self.worker_name,
            "scheduler_address": self.scheduler_address,
            "workers": my_workers,
            "worker_count": len(my_workers)
        }
    
    def shutdown(self):
        """
        Shutdown the worker node.
        """
        if self.is_running:
            self.cluster_manager.shutdown()
            self.is_running = False
            logger.info(f"Worker node '{self.worker_name}' shutdown")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

