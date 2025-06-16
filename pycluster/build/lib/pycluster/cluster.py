"""
Core cluster management functionality for PyCluster
"""

import os
import time
import socket
import logging
from typing import Optional, List, Dict, Any
from dask.distributed import Client, Scheduler, Worker, LocalCluster
from dask import delayed
import threading
import subprocess
import psutil

logger = logging.getLogger(__name__)


class ClusterManager:
    """
    Main cluster management class that provides a high-level interface
    for creating and managing distributed computing clusters.
    """
    
    def __init__(self, cluster_name: str = "pycluster"):
        """
        Initialize the cluster manager.
        
        Args:
            cluster_name: Name of the cluster for identification
        """
        self.cluster_name = cluster_name
        self.scheduler_address = None
        self.dashboard_address = None
        self.client = None
        self.workers = []
        self.is_head_node = False
        self.scheduler_process = None
        self.worker_processes = []
        
    def start_head_node(self, 
                       host: str = "0.0.0.0", 
                       scheduler_port: int = 8786,
                       dashboard_port: int = 8787,
                       n_local_workers: int = 0) -> Dict[str, Any]:
        """
        Start the head node (scheduler) of the cluster.
        
        Args:
            host: Host address to bind the scheduler to
            scheduler_port: Port for the scheduler
            dashboard_port: Port for the dashboard
            n_local_workers: Number of local workers to start on head node
            
        Returns:
            Dictionary with connection information
        """
        try:
            # Start scheduler
            self.scheduler_address = f"tcp://{host}:{scheduler_port}"
            self.dashboard_address = f"http://{host}:{dashboard_port}"
            
            # Use LocalCluster for simplicity, but configure it properly
            if n_local_workers > 0:
                self.cluster = LocalCluster(
                    n_workers=n_local_workers,
                    scheduler_port=scheduler_port,
                    dashboard_address=f":{dashboard_port}",
                    host=host,
                    processes=True
                )
                self.client = Client(self.cluster)
            else:
                # Start just the scheduler without local workers
                from .windows_fixes import start_scheduler_safely, fix_windows_event_loop
                from dask.distributed import Scheduler
                
                # Apply Windows fixes
                fix_windows_event_loop()
                
                self.scheduler = Scheduler(
                    host=host,
                    port=scheduler_port,
                    dashboard_address=f":{dashboard_port}"
                )
                
                # Use the safe scheduler startup method
                try:
                    self.scheduler = start_scheduler_safely(self.scheduler, timeout=60)
                    logger.info("Scheduler started successfully")
                except Exception as e:
                    logger.error(f"Failed to start scheduler: {e}")
                    raise
                
                # Wait a moment for scheduler to be ready
                import time
                time.sleep(2)
                
                self.client = Client(self.scheduler_address)
            
            self.is_head_node = True
            
            logger.info(f"Head node started successfully")
            logger.info(f"Scheduler address: {self.scheduler_address}")
            logger.info(f"Dashboard address: {self.dashboard_address}")
            
            return {
                "status": "success",
                "scheduler_address": self.scheduler_address,
                "dashboard_address": self.dashboard_address,
                "cluster_name": self.cluster_name
            }
            
        except Exception as e:
            logger.error(f"Failed to start head node: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def add_worker(self, 
                   scheduler_address: str,
                   n_workers: int = 1,
                   threads_per_worker: int = None,
                   memory_limit: str = "auto") -> Dict[str, Any]:
        """
        Add worker(s) to an existing cluster.
        
        Args:
            scheduler_address: Address of the scheduler to connect to
            n_workers: Number of workers to start
            threads_per_worker: Number of threads per worker
            memory_limit: Memory limit per worker
            
        Returns:
            Dictionary with worker information
        """
        try:
            if threads_per_worker is None:
                threads_per_worker = psutil.cpu_count()
            
            # Start workers
            for i in range(n_workers):
                worker = Worker(
                    scheduler_address,
                    nthreads=threads_per_worker,
                    memory_limit=memory_limit,
                    name=f"worker-{socket.gethostname()}-{i}"
                )
                worker.start()
                self.workers.append(worker)
            
            # Connect client if not already connected
            if self.client is None:
                self.client = Client(scheduler_address)
                self.scheduler_address = scheduler_address
            
            logger.info(f"Added {n_workers} worker(s) to cluster")
            
            return {
                "status": "success",
                "workers_added": n_workers,
                "total_workers": len(self.workers)
            }
            
        except Exception as e:
            logger.error(f"Failed to add workers: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get information about the current cluster state.
        
        Returns:
            Dictionary with cluster information
        """
        if self.client is None:
            return {"status": "not_connected"}
        
        try:
            info = self.client.scheduler_info()
            workers_info = []
            
            for worker_id, worker_info in info.get("workers", {}).items():
                workers_info.append({
                    "id": worker_id,
                    "host": worker_info.get("host"),
                    "nthreads": worker_info.get("nthreads"),
                    "memory_limit": worker_info.get("memory_limit"),
                    "status": worker_info.get("status")
                })
            
            return {
                "status": "connected",
                "cluster_name": self.cluster_name,
                "scheduler_address": self.scheduler_address,
                "dashboard_address": self.dashboard_address,
                "total_workers": len(workers_info),
                "workers": workers_info,
                "is_head_node": self.is_head_node
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster info: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def submit_task(self, func, *args, **kwargs):
        """
        Submit a task to the cluster for execution.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future object representing the computation
        """
        if self.client is None:
            raise RuntimeError("Not connected to a cluster")
        
        return self.client.submit(func, *args, **kwargs)
    
    def map_tasks(self, func, *iterables, **kwargs):
        """
        Map a function over iterables in parallel across the cluster.
        
        Args:
            func: Function to map
            *iterables: Iterables to map over
            **kwargs: Additional keyword arguments
            
        Returns:
            List of Future objects
        """
        if self.client is None:
            raise RuntimeError("Not connected to a cluster")
        
        return self.client.map(func, *iterables, **kwargs)
    
    def shutdown(self):
        """
        Shutdown the cluster and clean up resources.
        """
        try:
            if self.client:
                self.client.close()
            
            for worker in self.workers:
                worker.close()
            
            if hasattr(self, 'cluster'):
                self.cluster.close()
            
            if hasattr(self, 'scheduler'):
                self.scheduler.close()
            
            logger.info("Cluster shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

