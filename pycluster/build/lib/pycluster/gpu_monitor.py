"""
Enhanced GPU monitoring and management for PyCluster
"""

import os
import sys
import time
import threading
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import GPU monitoring libraries
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available. GPU monitoring will be limited.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. System monitoring will be limited.")


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    index: int
    name: str
    uuid: str
    memory_total: int
    memory_used: int
    memory_free: int
    utilization: int
    temperature: int
    power_usage: float
    power_limit: float
    driver_version: str
    cuda_version: str
    processes: List[Dict[str, Any]]


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_total: int
    memory_available: int
    disk_usage: float
    network_io: Dict[str, int]
    gpu_metrics: List[GPUInfo]


class GPUMonitor:
    """
    Monitor NVIDIA GPUs using NVML (NVIDIA Management Library).
    """
    
    def __init__(self):
        self.initialized = False
        self.gpu_count = 0
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.max_history = 1000  # Keep last 1000 measurements
        
        self._initialize_nvml()
    
    def _initialize_nvml(self):
        """Initialize NVML library."""
        if not PYNVML_AVAILABLE:
            logger.warning("NVML not available. GPU monitoring disabled.")
            return
        
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.initialized = True
            logger.info(f"NVML initialized. Found {self.gpu_count} GPU(s)")
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.initialized = False
    
    def get_gpu_info(self, gpu_index: int) -> Optional[GPUInfo]:
        """Get detailed information about a specific GPU."""
        if not self.initialized:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            
            # Basic device info
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            uuid = pynvml.nvmlDeviceGetUUID(handle).decode('utf-8')
            
            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = memory_info.total
            memory_used = memory_info.used
            memory_free = memory_info.free
            
            # Utilization
            utilization_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization = utilization_info.gpu
            
            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0
            
            # Power usage
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
            except:
                power_usage = 0.0
                power_limit = 0.0
            
            # Driver and CUDA version
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            except:
                driver_version = "Unknown"
                cuda_version = "Unknown"
            
            # Running processes
            processes = []
            try:
                process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in process_info:
                    try:
                        proc_name = psutil.Process(proc.pid).name() if PSUTIL_AVAILABLE else "Unknown"
                        processes.append({
                            'pid': proc.pid,
                            'name': proc_name,
                            'memory_used': proc.usedGpuMemory
                        })
                    except:
                        pass
            except:
                pass
            
            return GPUInfo(
                index=gpu_index,
                name=name,
                uuid=uuid,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_free=memory_free,
                utilization=utilization,
                temperature=temperature,
                power_usage=power_usage,
                power_limit=power_limit,
                driver_version=driver_version,
                cuda_version=cuda_version,
                processes=processes
            )
        
        except Exception as e:
            logger.error(f"Failed to get GPU {gpu_index} info: {e}")
            return None
    
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """Get information about all GPUs."""
        gpu_info_list = []
        for i in range(self.gpu_count):
            info = self.get_gpu_info(i)
            if info:
                gpu_info_list.append(info)
        return gpu_info_list
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get comprehensive system metrics including GPU info."""
        timestamp = datetime.now()
        
        # CPU and memory metrics
        if PSUTIL_AVAILABLE:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_total = memory.total
            memory_available = memory.available
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        else:
            cpu_usage = 0.0
            memory_usage = 0.0
            memory_total = 0
            memory_available = 0
            disk_usage = 0.0
            network_io = {}
        
        # GPU metrics
        gpu_metrics = self.get_all_gpu_info()
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_total=memory_total,
            memory_available=memory_available,
            disk_usage=disk_usage,
            network_io=network_io,
            gpu_metrics=gpu_metrics
        )
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous monitoring in a background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started GPU monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped GPU monitoring")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.get_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def get_recent_metrics(self, count: int = 60) -> List[SystemMetrics]:
        """Get recent metrics history."""
        return self.metrics_history[-count:] if self.metrics_history else []
    
    def get_gpu_summary(self) -> Dict[str, Any]:
        """Get a summary of GPU status."""
        if not self.initialized:
            return {
                'available': False,
                'count': 0,
                'message': 'NVML not available'
            }
        
        gpu_info = self.get_all_gpu_info()
        
        total_memory = sum(gpu.memory_total for gpu in gpu_info)
        used_memory = sum(gpu.memory_used for gpu in gpu_info)
        avg_utilization = sum(gpu.utilization for gpu in gpu_info) / len(gpu_info) if gpu_info else 0
        avg_temperature = sum(gpu.temperature for gpu in gpu_info) / len(gpu_info) if gpu_info else 0
        total_power = sum(gpu.power_usage for gpu in gpu_info)
        
        return {
            'available': True,
            'count': len(gpu_info),
            'total_memory_gb': total_memory / (1024**3),
            'used_memory_gb': used_memory / (1024**3),
            'memory_utilization': (used_memory / total_memory * 100) if total_memory > 0 else 0,
            'avg_gpu_utilization': avg_utilization,
            'avg_temperature': avg_temperature,
            'total_power_watts': total_power,
            'gpus': [
                {
                    'index': gpu.index,
                    'name': gpu.name,
                    'memory_used_gb': gpu.memory_used / (1024**3),
                    'memory_total_gb': gpu.memory_total / (1024**3),
                    'utilization': gpu.utilization,
                    'temperature': gpu.temperature,
                    'power_usage': gpu.power_usage
                }
                for gpu in gpu_info
            ]
        }
    
    def cleanup(self):
        """Cleanup NVML resources."""
        self.stop_monitoring()
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown completed")
            except Exception as e:
                logger.error(f"Error during NVML shutdown: {e}")


class LLMResourceManager:
    """
    Manage resources for LLM workloads across the cluster.
    """
    
    def __init__(self, gpu_monitor: GPUMonitor):
        self.gpu_monitor = gpu_monitor
        self.llm_instances = {}
        self.resource_allocations = {}
    
    def estimate_llm_memory_requirements(self, model_name: str, model_size: str) -> Dict[str, int]:
        """
        Estimate memory requirements for different LLM configurations.
        
        Args:
            model_name: Name of the model (e.g., "deepseek", "llama", "mistral")
            model_size: Size of the model (e.g., "7b", "13b", "70b")
        
        Returns:
            Dictionary with memory requirements in bytes
        """
        # Rough estimates based on model parameters and precision
        size_multipliers = {
            "1b": 1,
            "3b": 3,
            "7b": 7,
            "13b": 13,
            "30b": 30,
            "70b": 70,
            "175b": 175
        }
        
        base_memory_per_billion = {
            "fp32": 4 * 1024**3,  # 4GB per billion parameters for FP32
            "fp16": 2 * 1024**3,  # 2GB per billion parameters for FP16
            "int8": 1 * 1024**3,  # 1GB per billion parameters for INT8
            "int4": 0.5 * 1024**3  # 0.5GB per billion parameters for INT4
        }
        
        size_key = model_size.lower()
        if size_key not in size_multipliers:
            # Try to extract number from size string
            import re
            match = re.search(r'(\d+)b', size_key)
            if match:
                size_multipliers[size_key] = int(match.group(1))
            else:
                size_multipliers[size_key] = 7  # Default to 7B
        
        billion_params = size_multipliers[size_key]
        
        return {
            "model_memory_fp32": int(billion_params * base_memory_per_billion["fp32"]),
            "model_memory_fp16": int(billion_params * base_memory_per_billion["fp16"]),
            "model_memory_int8": int(billion_params * base_memory_per_billion["int8"]),
            "model_memory_int4": int(billion_params * base_memory_per_billion["int4"]),
            "kv_cache_memory": int(billion_params * 0.1 * 1024**3),  # Estimate for KV cache
            "activation_memory": int(billion_params * 0.2 * 1024**3),  # Estimate for activations
            "overhead_memory": int(1 * 1024**3)  # 1GB overhead
        }
    
    def find_suitable_gpus(self, memory_required: int, gpu_count: int = 1) -> List[int]:
        """
        Find GPUs with sufficient memory for LLM deployment.
        
        Args:
            memory_required: Required memory in bytes
            gpu_count: Number of GPUs needed
        
        Returns:
            List of GPU indices that meet requirements
        """
        gpu_info = self.gpu_monitor.get_all_gpu_info()
        suitable_gpus = []
        
        for gpu in gpu_info:
            if gpu.memory_free >= memory_required:
                suitable_gpus.append(gpu.index)
        
        return suitable_gpus[:gpu_count]
    
    def plan_llm_deployment(self, model_name: str, model_size: str, precision: str = "fp16") -> Dict[str, Any]:
        """
        Plan LLM deployment across available GPUs.
        
        Args:
            model_name: Name of the model
            model_size: Size of the model
            precision: Model precision (fp32, fp16, int8, int4)
        
        Returns:
            Deployment plan with GPU assignments and configuration
        """
        memory_reqs = self.estimate_llm_memory_requirements(model_name, model_size)
        model_memory_key = f"model_memory_{precision}"
        
        if model_memory_key not in memory_reqs:
            raise ValueError(f"Unsupported precision: {precision}")
        
        total_memory_needed = (
            memory_reqs[model_memory_key] +
            memory_reqs["kv_cache_memory"] +
            memory_reqs["activation_memory"] +
            memory_reqs["overhead_memory"]
        )
        
        # Check if single GPU can handle the model
        suitable_gpus = self.find_suitable_gpus(total_memory_needed, 1)
        
        if suitable_gpus:
            # Single GPU deployment
            return {
                "deployment_type": "single_gpu",
                "gpu_indices": suitable_gpus[:1],
                "memory_per_gpu": total_memory_needed,
                "total_memory": total_memory_needed,
                "model_sharding": False,
                "tensor_parallel": 1,
                "pipeline_parallel": 1
            }
        
        # Multi-GPU deployment with model sharding
        all_gpus = self.gpu_monitor.get_all_gpu_info()
        if not all_gpus:
            raise RuntimeError("No GPUs available")
        
        # Calculate how many GPUs we need for model sharding
        max_gpu_memory = max(gpu.memory_free for gpu in all_gpus)
        min_gpus_needed = (total_memory_needed + max_gpu_memory - 1) // max_gpu_memory
        
        available_gpus = [gpu.index for gpu in all_gpus if gpu.memory_free >= total_memory_needed // min_gpus_needed]
        
        if len(available_gpus) < min_gpus_needed:
            raise RuntimeError(f"Insufficient GPU memory. Need {min_gpus_needed} GPUs with at least {total_memory_needed // min_gpus_needed / (1024**3):.1f}GB each")
        
        return {
            "deployment_type": "multi_gpu",
            "gpu_indices": available_gpus[:min_gpus_needed],
            "memory_per_gpu": total_memory_needed // min_gpus_needed,
            "total_memory": total_memory_needed,
            "model_sharding": True,
            "tensor_parallel": min_gpus_needed,
            "pipeline_parallel": 1
        }
    
    def allocate_resources(self, deployment_id: str, plan: Dict[str, Any]) -> bool:
        """
        Allocate GPU resources for an LLM deployment.
        
        Args:
            deployment_id: Unique identifier for the deployment
            plan: Deployment plan from plan_llm_deployment
        
        Returns:
            True if allocation successful, False otherwise
        """
        gpu_indices = plan["gpu_indices"]
        memory_per_gpu = plan["memory_per_gpu"]
        
        # Check if GPUs are still available
        current_gpu_info = self.gpu_monitor.get_all_gpu_info()
        gpu_memory_map = {gpu.index: gpu.memory_free for gpu in current_gpu_info}
        
        for gpu_idx in gpu_indices:
            if gpu_idx not in gpu_memory_map or gpu_memory_map[gpu_idx] < memory_per_gpu:
                logger.error(f"GPU {gpu_idx} no longer has sufficient memory")
                return False
        
        # Record allocation
        self.resource_allocations[deployment_id] = {
            "gpu_indices": gpu_indices,
            "memory_allocated": memory_per_gpu * len(gpu_indices),
            "timestamp": datetime.now(),
            "plan": plan
        }
        
        logger.info(f"Allocated resources for deployment {deployment_id}: GPUs {gpu_indices}")
        return True
    
    def deallocate_resources(self, deployment_id: str) -> bool:
        """
        Deallocate GPU resources for an LLM deployment.
        
        Args:
            deployment_id: Unique identifier for the deployment
        
        Returns:
            True if deallocation successful, False otherwise
        """
        if deployment_id in self.resource_allocations:
            allocation = self.resource_allocations.pop(deployment_id)
            logger.info(f"Deallocated resources for deployment {deployment_id}: GPUs {allocation['gpu_indices']}")
            return True
        return False
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource allocation status."""
        gpu_summary = self.gpu_monitor.get_gpu_summary()
        
        return {
            "gpu_summary": gpu_summary,
            "active_allocations": len(self.resource_allocations),
            "allocations": {
                deployment_id: {
                    "gpu_indices": alloc["gpu_indices"],
                    "memory_allocated_gb": alloc["memory_allocated"] / (1024**3),
                    "deployment_type": alloc["plan"]["deployment_type"],
                    "timestamp": alloc["timestamp"].isoformat()
                }
                for deployment_id, alloc in self.resource_allocations.items()
            }
        }

