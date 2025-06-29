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
        
    async def start_head_node(self, 
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
                    host=host
                )
                self.client = Client(self.cluster)
            else:
                # Start just the scheduler without local workers
                from .windows_fixes import start_scheduler_safely, fix_windows_event_loop
                from dask.distributed import Scheduler
                
                # Apply Windows fixes
                fix_windows_event_loop()
                self.scheduler = await start_scheduler_safely(
                    host=host,
                    port=scheduler_port,
                    dashboard_address=f":{dashboard_port}"
                )
                
            self.is_head_node = True
            
            # Get actual IP for connection info
            actual_host = host
            if host == "0.0.0.0":
                actual_host = socket.gethostbyname(socket.gethostname())
            
            return {
                "status": "success",
                "cluster_name": self.cluster_name,
                "scheduler_address": f"tcp://{actual_host}:{scheduler_port}",
                "dashboard_address": f"http://{actual_host}:{dashboard_port}",
                "host": actual_host,
                "n_local_workers": n_local_workers
            }
            
        except Exception as e:
            logger.error(f"Failed to start head node: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def add_worker(self, 
                   scheduler_address: str,
                   n_workers: int = 1,
                   threads_per_worker: int = None,
                   memory_limit: str = "auto") -> Dict[str, Any]:
        """
        Add worker nodes to the cluster.
        
        Args:
            scheduler_address: Address of the scheduler to connect to
            n_workers: Number of workers to start
            threads_per_worker: Number of threads per worker
            memory_limit: Memory limit per worker
            
        Returns:
            Dictionary with worker information
        """
        try:
            self.scheduler_address = scheduler_address
            
            # Create workers
            for i in range(n_workers):
                worker = Worker(
                    scheduler_address,
                    nthreads=threads_per_worker,
                    memory_limit=memory_limit,
                    name=f"worker-{socket.gethostname()}-{i}"
                )
                
                # Start worker in a separate thread
                worker_thread = threading.Thread(
                    target=self._start_worker_thread,
                    args=(worker,),
                    daemon=True
                )
                worker_thread.start()
                self.worker_processes.append(worker_thread)
                self.workers.append(worker)
            
            # Create client for task submission
            self.client = Client(scheduler_address)
            
            return {
                "status": "success",
                "workers_added": n_workers,
                "scheduler_address": scheduler_address
            }
            
        except Exception as e:
            logger.error(f"Failed to add workers: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _start_worker_thread(self, worker):
        """Start a worker in a separate thread."""
        try:
            worker.start()
        except Exception as e:
            logger.error(f"Worker failed to start: {e}")
    
    def submit_task(self, func, *args, **kwargs):
        """
        Submit a task to the cluster.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future object for the task
        """
        if not self.client:
            raise RuntimeError("No client connected to cluster")
        
        return self.client.submit(func, *args, **kwargs)
    
    def map_tasks(self, func, data):
        """
        Map a function over data using the cluster.
        
        Args:
            func: Function to apply
            data: Data to map over
            
        Returns:
            List of Future objects
        """
        if not self.client:
            raise RuntimeError("No client connected to cluster")
        
        return self.client.map(func, data)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get information about the cluster.
        
        Returns:
            Dictionary with cluster information
        """
        if not self.client:
            return {
                "status": "not_connected",
                "message": "No client connected to cluster"
            }
        
        try:
            info = self.client.scheduler_info()
            workers = list(info['workers'].values())
            
            return {
                "status": "connected",
                "cluster_name": self.cluster_name,
                "scheduler_address": self.scheduler_address,
                "dashboard_address": self.dashboard_address,
                "total_workers": len(workers),
                "total_cores": sum(w.get('nthreads', 1) for w in workers),
                "total_memory": sum(w.get('memory', 0) for w in workers),
                "workers": workers
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def shutdown(self):
        """Shutdown the cluster and cleanup resources."""
        try:
            if self.client:
                self.client.close()
            
            # Stop workers
            for worker in self.workers:
                try:
                    worker.close()
                except:
                    pass
            
            # Stop scheduler if we started it
            if hasattr(self, 'scheduler') and self.scheduler:
                try:
                    self.scheduler.close()
                except:
                    pass
            
            # Stop cluster if we created it
            if hasattr(self, 'cluster') and self.cluster:
                try:
                    self.cluster.close()
                except:
                    pass
            
            logger.info("Cluster shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()