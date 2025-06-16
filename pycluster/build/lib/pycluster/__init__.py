"""
PyCluster - A Windows-based Python clustering package with dashboard support

This package provides a simple interface for creating distributed computing clusters
using Dask as the underlying framework. It supports head node/worker architecture
similar to Ray but with better Windows compatibility and LLM serving capabilities.
"""

__version__ = "0.3.0"
__author__ = "PyCluster Development Team"

from .cluster import ClusterManager
from .node import HeadNode, WorkerNode
from .dashboard import DashboardManager
from .windows_utils import WindowsClusterManager
from .network_utils import NetworkDiscovery
from .gpu_monitor import GPUMonitor, LLMResourceManager
from .llm_serving import LLMClusterManager, LLMWorker, LLMRequest, LLMResponse
from .worker_discovery import EasyWorkerJoin, ClusterDiscovery

__all__ = [
    "ClusterManager", 
    "HeadNode", 
    "WorkerNode", 
    "DashboardManager",
    "WindowsClusterManager",
    "NetworkDiscovery",
    "GPUMonitor",
    "LLMResourceManager", 
    "LLMClusterManager",
    "LLMWorker",
    "LLMRequest",
    "LLMResponse",
    "EasyWorkerJoin",
    "ClusterDiscovery"
]

