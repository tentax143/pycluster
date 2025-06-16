# PyCluster v2.0 - Enhanced Documentation

## Overview

PyCluster v2.0 is a comprehensive Windows-based Python clustering package designed for distributed computing with advanced LLM (Large Language Model) support, NVIDIA GPU monitoring, and a beautiful modern dashboard. This enhanced version builds upon the original PyCluster with significant improvements for AI/ML workloads.

## Key Features

### 🚀 **Enhanced Core Features**
- **LLM-First Design**: Built specifically for deploying and managing Large Language Models
- **NVIDIA GPU Support**: Full GPU monitoring, resource allocation, and utilization tracking
- **Modern Dashboard**: Beautiful, responsive web interface with real-time monitoring
- **Windows Optimized**: Excellent Windows compatibility with native utilities
- **Distributed Architecture**: Head node/worker architecture similar to Ray but more accessible

### 🎯 **New in v2.0**
- **LLM Deployment Engine**: Deploy models like DeepSeek, Code Llama, and custom models
- **GPU Resource Manager**: Intelligent GPU allocation and memory management
- **Advanced Monitoring**: Real-time GPU metrics, temperature, power usage
- **Model Sharding**: Support for tensor parallel and pipeline parallel strategies
- **REST API**: Complete API for programmatic cluster and LLM management
- **Enhanced Dashboard**: Modern UI with charts, alerts, and performance metrics

## Quick Start

### Installation

```bash
# Install PyCluster with all dependencies
pip install pycluster[gpu]

# Or install basic version
pip install pycluster
```

### Basic Cluster Setup

```python
from pycluster import HeadNode, GPUMonitor

# Start monitoring GPUs
gpu_monitor = GPUMonitor()
gpu_monitor.start_monitoring()

# Create and start cluster
with HeadNode("my-cluster") as head:
    result = head.start(n_local_workers=4)
    print(f"Dashboard: {result['dashboard_address']}")
    
    # Your distributed computing code here
    cluster = head.cluster_manager
    future = cluster.submit_task(lambda x: x**2, 10)
    print(f"Result: {future.result()}")
```

### LLM Deployment

```python
from pycluster import HeadNode, LLMClusterManager, LLMResourceManager

with HeadNode("llm-cluster") as head:
    head.start(n_local_workers=2)
    
    # Initialize LLM management
    llm_manager = LLMClusterManager(head.cluster_manager)
    
    # Deploy DeepSeek model
    deployment_id = llm_manager.deploy_model(
        model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        model_size="7b",
        precision="fp16",
        gpu_per_replica=2
    )
    
    # Perform inference
    response = llm_manager.inference(
        deployment_id=deployment_id,
        prompt="Write a Python function to sort a list:",
        max_tokens=200,
        temperature=0.2
    )
    
    print(f"Generated code: {response['text']}")
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     PyCluster v2.0                         │
├─────────────────────────────────────────────────────────────┤
│  Dashboard (React)     │  API Server (Flask)               │
│  - Real-time monitoring │  - REST endpoints                │
│  - GPU statistics      │  - CORS enabled                   │
│  - LLM management      │  - WebSocket support              │
├─────────────────────────────────────────────────────────────┤
│                    Core Framework                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │
│  │   Head Node     │ │  Worker Nodes   │ │ GPU Monitor   │ │
│  │  - Scheduler    │ │  - Task exec    │ │ - NVML        │ │
│  │  - Dashboard    │ │  - GPU access   │ │ - Metrics     │ │
│  │  - Coordination │ │  - LLM serving  │ │ - Allocation  │ │
│  └─────────────────┘ └─────────────────┘ └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   LLM Management Layer                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │
│  │ LLM Manager     │ │ Resource Mgr    │ │ Model Registry│ │
│  │ - Deployment    │ │ - GPU alloc     │ │ - HF models   │ │
│  │ - Inference     │ │ - Memory mgmt   │ │ - Custom      │ │
│  │ - Scaling       │ │ - Sharding      │ │ - Versioning  │ │
│  └─────────────────┘ └─────────────────┘ └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Dask Foundation                         │
│  - Distributed computing  - Task scheduling                │
│  - Worker management     - Data serialization              │
│  - Network communication - Fault tolerance                 │
└─────────────────────────────────────────────────────────────┘
```

### GPU Resource Management

PyCluster v2.0 includes sophisticated GPU resource management:

1. **GPU Discovery**: Automatic detection of NVIDIA GPUs using NVML
2. **Memory Tracking**: Real-time monitoring of GPU memory usage
3. **Temperature Monitoring**: Thermal management and alerts
4. **Power Management**: Power usage tracking and limits
5. **Resource Allocation**: Intelligent assignment of GPUs to LLM deployments

### LLM Deployment Strategies

#### Single GPU Deployment
```python
# Deploy small model on single GPU
deployment_id = llm_manager.deploy_model(
    model_name="microsoft/DialoGPT-small",
    model_size="117m",
    gpu_per_replica=1
)
```

#### Multi-GPU Tensor Parallel
```python
# Deploy large model across multiple GPUs
deployment_id = llm_manager.deploy_model(
    model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    model_size="7b",
    gpu_per_replica=2,
    sharding_strategy="tensor_parallel"
)
```

#### Pipeline Parallel
```python
# Deploy very large model with pipeline parallelism
deployment_id = llm_manager.deploy_model(
    model_name="custom/large-model-70b",
    model_size="70b",
    gpu_per_replica=8,
    sharding_strategy="pipeline_parallel"
)
```

## Dashboard Features

### Overview Tab
- **Cluster Status**: Active workers, GPU utilization, memory usage
- **Real-time Charts**: GPU utilization, system resources over time
- **Quick Stats**: Key metrics at a glance

### GPU Monitoring Tab
- **Per-GPU Details**: Memory usage, utilization, temperature, power
- **Historical Data**: GPU performance trends
- **Alerts**: Temperature and power warnings

### LLM Models Tab
- **Model Management**: Deploy, monitor, and manage LLM deployments
- **Performance Metrics**: Requests/sec, latency, tokens/sec
- **Resource Usage**: GPU allocation per model
- **Control Actions**: Start, stop, restart, scale models

### Workers Tab
- **Worker Status**: Health, resource usage, GPU assignment
- **Network Information**: IP addresses, connection status
- **Resource Allocation**: CPU, memory, GPU assignment per worker

### Monitoring Tab
- **System Alerts**: Recent warnings and notifications
- **Network I/O**: Cluster communication metrics
- **Resource Distribution**: Memory and GPU usage across cluster

## API Reference

### Cluster Management

#### Start Head Node
```http
POST /api/cluster/start
Content-Type: application/json

{
    "cluster_name": "my-cluster",
    "n_workers": 4,
    "worker_resources": {"GPU": 1}
}
```

#### Get Cluster Status
```http
GET /api/cluster/status
```

### GPU Monitoring

#### Get GPU Status
```http
GET /api/llm/gpu/status
```

Response:
```json
{
    "summary": {
        "available": true,
        "count": 4,
        "gpus": [
            {
                "id": 0,
                "name": "NVIDIA RTX 4090",
                "memory_total": 24.0,
                "memory_used": 18.2,
                "utilization": 87,
                "temperature": 72,
                "power_usage": 380
            }
        ]
    },
    "recent_metrics": [...]
}
```

### LLM Management

#### Deploy Model
```http
POST /api/llm/models/deploy
Content-Type: application/json

{
    "model_name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "model_size": "7b",
    "precision": "fp16",
    "gpu_per_replica": 2,
    "sharding_strategy": "tensor_parallel"
}
```

#### Perform Inference
```http
POST /api/llm/models/{deployment_id}/inference
Content-Type: application/json

{
    "prompt": "Write a Python function to calculate fibonacci:",
    "max_tokens": 200,
    "temperature": 0.2,
    "stop_sequences": ["```"]
}
```

## Configuration

### Environment Variables

```bash
# GPU Configuration
PYCLUSTER_GPU_MEMORY_FRACTION=0.9
PYCLUSTER_GPU_ALLOW_GROWTH=true

# Cluster Configuration
PYCLUSTER_SCHEDULER_PORT=8786
PYCLUSTER_DASHBOARD_PORT=8787
PYCLUSTER_API_PORT=5000

# LLM Configuration
PYCLUSTER_MODEL_CACHE_DIR=/path/to/models
PYCLUSTER_MAX_MODEL_SIZE=70b
PYCLUSTER_DEFAULT_PRECISION=fp16
```

### Configuration File

Create `pycluster.yaml`:

```yaml
cluster:
  name: "production-cluster"
  scheduler_port: 8786
  dashboard_port: 8787
  
gpu:
  memory_fraction: 0.9
  allow_growth: true
  temperature_threshold: 85
  power_limit_threshold: 0.95

llm:
  model_cache_dir: "/models"
  default_precision: "fp16"
  max_concurrent_requests: 100
  request_timeout: 300

monitoring:
  gpu_interval: 1.0
  metrics_retention: 3600
  alert_thresholds:
    gpu_temperature: 80
    gpu_memory: 0.95
    cluster_memory: 0.90
```

## Windows-Specific Features

### Windows Service Integration
```python
from pycluster.windows_utils import WindowsClusterManager

# Install as Windows service
service_manager = WindowsClusterManager()
service_manager.install_service(
    service_name="PyCluster",
    cluster_config="pycluster.yaml"
)
```

### Firewall Configuration
```python
from pycluster.windows_utils import configure_firewall

# Automatically configure Windows Firewall
configure_firewall(
    ports=[8786, 8787, 5000],
    cluster_name="PyCluster"
)
```

### Network Discovery
```python
from pycluster.network_utils import NetworkDiscovery

# Discover cluster nodes on network
discovery = NetworkDiscovery()
nodes = discovery.discover_cluster_nodes()
print(f"Found {len(nodes)} cluster nodes")
```

## Performance Optimization

### GPU Memory Optimization

1. **Use Mixed Precision**: Always use `fp16` or `bf16` for inference
2. **Model Sharding**: Split large models across multiple GPUs
3. **Memory Pooling**: Enable GPU memory pooling for better utilization
4. **Batch Processing**: Process multiple requests in batches

### Cluster Optimization

1. **Worker Placement**: Place workers close to data sources
2. **Network Optimization**: Use high-bandwidth connections between nodes
3. **Resource Allocation**: Match worker resources to workload requirements
4. **Load Balancing**: Distribute work evenly across workers

### Example: Optimized DeepSeek Deployment

```python
from pycluster import HeadNode, LLMClusterManager, LLMResourceManager

# Optimized configuration for DeepSeek 7B
config = {
    "model_name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "model_size": "7b",
    "precision": "fp16",  # Reduce memory usage
    "gpu_per_replica": 2,  # Use 2 GPUs for tensor parallelism
    "max_memory_per_gpu": "12GB",  # Leave memory for other processes
    "sharding_strategy": "tensor_parallel",
    "batch_size": 8,  # Process multiple requests together
    "max_sequence_length": 2048,  # Reasonable context length
}

with HeadNode("optimized-cluster") as head:
    head.start(n_local_workers=2, worker_resources={"GPU": 2})
    
    llm_manager = LLMClusterManager(head.cluster_manager)
    deployment_id = llm_manager.deploy_model(**config)
    
    # Warm up the model
    llm_manager.inference(
        deployment_id=deployment_id,
        prompt="def hello():",
        max_tokens=10
    )
    
    print("✅ Optimized DeepSeek deployment ready!")
```

## Troubleshooting

### Common Issues

#### GPU Not Detected
```
Error: NVML Shared Library Not Found
```

**Solution**: Install NVIDIA drivers and CUDA toolkit
```bash
# Windows: Download from NVIDIA website
# Linux: 
sudo apt update
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
```

#### Insufficient GPU Memory
```
Error: CUDA out of memory
```

**Solutions**:
1. Use smaller precision: `fp16` instead of `fp32`
2. Reduce batch size
3. Enable model sharding across multiple GPUs
4. Use gradient checkpointing

#### Model Loading Timeout
```
Error: Model deployment timeout
```

**Solutions**:
1. Increase timeout in configuration
2. Use faster storage (SSD) for model cache
3. Pre-download models to local cache
4. Check network connectivity to Hugging Face

#### Worker Connection Issues
```
Error: Worker failed to connect to scheduler
```

**Solutions**:
1. Check firewall settings
2. Verify network connectivity
3. Ensure ports are not blocked
4. Check Windows Defender settings

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from pycluster import HeadNode
# Debug information will be printed
```

### Performance Profiling

```python
from pycluster.profiling import ClusterProfiler

profiler = ClusterProfiler()
profiler.start()

# Your cluster operations here

report = profiler.stop()
print(report.summary())
```

## Examples

### Example 1: Code Generation Cluster

```python
"""
Deploy a code generation cluster with DeepSeek Coder
"""
from pycluster import HeadNode, LLMClusterManager

def setup_code_generation_cluster():
    with HeadNode("code-cluster", host="0.0.0.0") as head:
        # Start cluster with GPU workers
        result = head.start(n_local_workers=2, worker_resources={"GPU": 1})
        print(f"Code cluster started: {result['dashboard_address']}")
        
        # Deploy DeepSeek Coder
        llm_manager = LLMClusterManager(head.cluster_manager)
        deployment_id = llm_manager.deploy_model(
            model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            model_size="7b",
            precision="fp16",
            gpu_per_replica=1
        )
        
        # Test code generation
        coding_tasks = [
            "Write a Python function to implement quicksort:",
            "Create a REST API endpoint using Flask:",
            "Implement a binary search tree class:",
        ]
        
        for task in coding_tasks:
            response = llm_manager.inference(
                deployment_id=deployment_id,
                prompt=task,
                max_tokens=300,
                temperature=0.1  # Low temperature for consistent code
            )
            print(f"\nTask: {task}")
            print(f"Generated code:\n{response['text']}")

if __name__ == "__main__":
    setup_code_generation_cluster()
```

### Example 2: Multi-Model Serving

```python
"""
Deploy multiple models for different tasks
"""
from pycluster import HeadNode, LLMClusterManager

def setup_multi_model_cluster():
    with HeadNode("multi-model-cluster") as head:
        head.start(n_local_workers=4, worker_resources={"GPU": 1})
        
        llm_manager = LLMClusterManager(head.cluster_manager)
        
        # Deploy different models for different tasks
        models = {
            "code": llm_manager.deploy_model(
                model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                model_size="7b",
                gpu_per_replica=2
            ),
            "chat": llm_manager.deploy_model(
                model_name="microsoft/DialoGPT-medium",
                model_size="345m",
                gpu_per_replica=1
            )
        }
        
        # Route requests to appropriate models
        def generate_code(prompt):
            return llm_manager.inference(
                deployment_id=models["code"],
                prompt=prompt,
                temperature=0.1
            )
        
        def chat_response(prompt):
            return llm_manager.inference(
                deployment_id=models["chat"],
                prompt=prompt,
                temperature=0.7
            )
        
        # Example usage
        code = generate_code("def fibonacci(n):")
        chat = chat_response("Hello, how are you?")
        
        print(f"Generated code: {code['text']}")
        print(f"Chat response: {chat['text']}")

if __name__ == "__main__":
    setup_multi_model_cluster()
```

## Contributing

We welcome contributions to PyCluster! Here's how to get started:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/pycluster/pycluster.git
cd pycluster

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .[dev,gpu]

# Run tests
pytest tests/ -v
```

### Code Style

We use Black for code formatting:
```bash
black pycluster/
```

### Testing

Run the full test suite:
```bash
pytest tests/ -v --cov=pycluster
```

## License

PyCluster is released under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [https://pycluster.readthedocs.io](https://pycluster.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/pycluster/pycluster/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pycluster/pycluster/discussions)
- **Email**: support@pycluster.org

## Changelog

### v2.0.0 (Current)
- ✨ **New**: LLM deployment and serving capabilities
- ✨ **New**: NVIDIA GPU monitoring and resource management
- ✨ **New**: Modern React dashboard with real-time monitoring
- ✨ **New**: REST API for programmatic control
- ✨ **New**: Model sharding and distributed inference
- ✨ **New**: Windows service integration
- 🔧 **Improved**: Better error handling and logging
- 🔧 **Improved**: Enhanced documentation and examples
- 🐛 **Fixed**: Various stability and performance issues

### v1.0.0
- 🎉 **Initial**: Basic clustering functionality
- 🎉 **Initial**: Head node/worker architecture
- 🎉 **Initial**: Simple dashboard
- 🎉 **Initial**: Windows compatibility

---

**PyCluster v2.0** - Bringing enterprise-grade LLM deployment to everyone! 🚀

