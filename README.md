# PyCluster Codebase Index

## Overview
PyCluster is a Windows-first distributed computing framework built on Dask, designed to simplify the creation and management of Python clusters. It features a head node/worker architecture, comprehensive GPU monitoring, and powerful capabilities for deploying and serving Large Language Models (LLMs) across your cluster.

**Version**: 0.3.0  
**License**: MIT  
**Python**: 3.8+  
**Platform**: Windows (primary), Linux/macOS (experimental)

## Project Structure

```
pycluster_working_copy/
├── pycluster/                          # Main PyCluster package
│   ├── pycluster/                      # Core Python package
│   │   ├── __init__.py                 # Package initialization & exports
│   │   ├── cluster.py                  # Core Dask cluster management
│   │   ├── node.py                     # HeadNode & WorkerNode classes
│   │   ├── cli.py                      # Original CLI (legacy)
│   │   ├── cli_enhanced.py             # Enhanced CLI with Windows fixes
│   │   ├── dashboard.py                # Dashboard integration
│   │   ├── gpu_monitor.py              # NVIDIA GPU monitoring
│   │   ├── llm_serving.py              # LLM deployment & serving
│   │   ├── network_utils.py            # Network utilities
│   │   ├── windows_fixes.py            # Windows-specific bug fixes
│   │   ├── windows_utils.py            # Windows utility functions
│   │   └── worker_discovery.py         # Auto-discovery & easy join
│   ├── examples/                       # Usage examples
│   │   ├── basic_example.py            # Basic cluster setup
│   │   ├── llm_example.py              # LLM deployment example
│   │   ├── deepseek_example.py         # DeepSeek model example
│   │   └── multi_machine_example.py    # Multi-machine setup
│   ├── tests/                          # Test suite
│   │   ├── test_pycluster.py           # Core functionality tests
│   │   └── test_enhanced_features.py   # Enhanced features tests
│   ├── join_worker.py                  # Standalone worker join script
│   ├── setup.py                        # Package setup
│   ├── pyproject.toml                  # Project configuration
│   └── README.md                       # Main documentation
├── pycluster-api/                      # Flask REST API backend
│   ├── src/
│   │   ├── main.py                     # Flask app entry point
│   │   ├── routes/
│   │   │   ├── cluster.py              # Cluster management API
│   │   │   ├── llm.py                  # LLM management API
│   │   │   └── user.py                 # User management API
│   │   ├── models/
│   │   │   └── user.py                 # User model
│   │   └── database/
│   │       └── app.db                  # SQLite database
│   └── requirements.txt                # API dependencies
└── pycluster-dashboard/                # React web dashboard frontend
```

## Core Components

### 1. Cluster Management (`cluster.py`)
**Purpose**: Core Dask cluster management functionality

**Key Classes**:
- `ClusterManager`: Main cluster management class
  - `start_head_node()`: Start scheduler and dashboard
  - `add_worker()`: Add worker nodes to cluster
  - `submit_task()`: Submit individual tasks
  - `map_tasks()`: Map tasks across workers
  - `get_cluster_info()`: Get cluster status

**Key Features**:
- Head node/worker architecture
- Task submission and execution
- Cluster monitoring and status reporting
- Windows compatibility fixes

### 2. Node Management (`node.py`)
**Purpose**: Simplified interfaces for head and worker nodes

**Key Classes**:
- `HeadNode`: Simplified head node interface
  - `start()`: Start head node with optional local workers
  - `get_connection_info()`: Get connection details
  - `get_cluster_status()`: Get cluster status
  - `shutdown()`: Graceful shutdown

- `WorkerNode`: Simplified worker node interface
  - `start()`: Connect to scheduler and start workers
  - `get_status()`: Get worker status
  - `shutdown()`: Graceful shutdown

### 3. LLM Serving (`llm_serving.py`)
**Purpose**: Large Language Model deployment and inference

**Key Classes**:
- `LLMWorker`: Worker for LLM inference on specific GPUs
  - `load_model()`: Load model onto assigned GPUs
  - `submit_request()`: Submit inference requests
  - `_process_request()`: Process inference requests

- `LLMClusterManager`: Cluster-wide LLM management
  - `deploy_model()`: Deploy model across cluster
  - `inference()`: Perform distributed inference
  - `get_deployment_status()`: Check deployment status
  - `undeploy_model()`: Remove model deployment

**Data Classes**:
- `LLMRequest`: Inference request structure
- `LLMResponse`: Inference response structure
- `LLMModelInfo`: Model information

### 4. GPU Monitoring (`gpu_monitor.py`)
**Purpose**: Comprehensive NVIDIA GPU monitoring and resource management

**Key Classes**:
- `GPUMonitor`: NVIDIA GPU monitoring using NVML
  - `get_gpu_info()`: Get detailed GPU information
  - `get_system_metrics()`: Get comprehensive system metrics
  - `start_monitoring()`: Start continuous monitoring
  - `get_gpu_summary()`: Get GPU summary statistics

- `LLMResourceManager`: LLM-specific resource management
  - `estimate_llm_memory_requirements()`: Estimate model memory needs
  - `find_suitable_gpus()`: Find GPUs with sufficient memory
  - `plan_llm_deployment()`: Plan optimal deployment strategy
  - `allocate_resources()`: Allocate GPU resources

**Data Classes**:
- `GPUInfo`: Detailed GPU information
- `SystemMetrics`: System-wide metrics

### 5. Worker Discovery (`worker_discovery.py`)
**Purpose**: Automatic cluster discovery and easy worker joining

**Key Classes**:
- `ClusterDiscovery`: Automatic cluster discovery
  - `start_broadcasting()`: Broadcast cluster information
  - `start_discovery()`: Listen for cluster broadcasts
  - `get_discovered_clusters()`: Get discovered clusters

- `EasyWorkerJoin`: Simplified worker joining
  - `discover_clusters()`: Discover available clusters
  - `join_cluster_interactive()`: Interactive cluster joining
  - `join_cluster_by_name()`: Join specific cluster by name
  - `test_connection()`: Test cluster connectivity

**Data Classes**:
- `ClusterInfo`: Cluster information for discovery

### 6. Windows Utilities (`windows_fixes.py`, `windows_utils.py`)
**Purpose**: Windows-specific optimizations and fixes

**Key Features**:
- Windows event loop fixes
- Firewall configuration assistance
- Port availability checking
- Performance optimizations
- Configuration management

### 7. Enhanced CLI (`cli_enhanced.py`)
**Purpose**: Command-line interface with Windows support

**Key Features**:
- Head node startup: `python -m pycluster.cli_enhanced --verbose`
- Worker node startup: `python -m pycluster.cli_enhanced worker --scheduler tcp://IP:8786`
- Windows diagnostics: `--diagnose` flag
- Verbose logging: `--verbose` flag
- Configuration file support: `--config` flag

## API Backend (`pycluster-api/`)

### Flask Application (`src/main.py`)
**Purpose**: REST API server for cluster management

**Key Features**:
- CORS support for cross-origin requests
- Static file serving for dashboard
- Database initialization
- LLM manager initialization

### API Routes

#### Cluster Management (`src/routes/cluster.py`)
**Endpoints**:
- `GET /api/cluster/status`: Get cluster status
- `POST /api/cluster/start-head`: Start head node
- `POST /api/cluster/start-worker`: Start worker node
- `POST /api/cluster/connect`: Connect to existing cluster
- `POST /api/cluster/shutdown`: Shutdown cluster
- `POST /api/cluster/submit-task`: Submit computation task
- `GET /api/cluster/dashboard-info`: Get dashboard information
- `GET /api/cluster/workers`: Get worker information
- `GET /api/cluster/metrics`: Get cluster metrics
- `GET /api/cluster/health`: Health check

#### LLM Management (`src/routes/llm.py`)
**Endpoints**:
- `GET /api/llm/health`: LLM service health check
- `GET /api/llm/gpu/status`: Get GPU status
- `GET /api/llm/models`: List deployed models
- `POST /api/llm/models/deploy`: Deploy new model
- `GET /api/llm/models/<id>/status`: Get model deployment status
- `POST /api/llm/models/<id>/inference`: Perform inference
- `DELETE /api/llm/models/<id>`: Undeploy model
- `GET /api/llm/resources/status`: Get resource status
- `GET /api/llm/models/available`: Get available models

#### User Management (`src/routes/user.py`)
**Endpoints**:
- User authentication and management
- Session handling

## Examples

### Basic Example (`examples/basic_example.py`)
Demonstrates:
- Head node startup with local workers
- Dashboard integration
- Task submission and execution
- Cluster monitoring

### LLM Example (`examples/llm_example.py`)
Demonstrates:
- GPU monitoring initialization
- LLM deployment planning
- Model deployment and inference
- Resource management
- Cleanup procedures

### DeepSeek Example (`examples/deepseek_example.py`)
Demonstrates:
- DeepSeek model deployment
- Advanced LLM configuration
- Distributed inference

### Multi-Machine Example (`examples/multi_machine_example.py`)
Demonstrates:
- Multi-machine cluster setup
- Worker node joining
- Distributed task execution

## Testing

### Test Suite (`tests/`)
**Files**:
- `test_pycluster.py`: Core functionality tests
- `test_enhanced_features.py`: Enhanced features tests

**Test Coverage**:
- Cluster manager functionality
- Head node and worker node operations
- Dashboard integration
- Windows utilities
- Network discovery
- Integration workflows

## Key Features

### 1. Distributed Computing
- Dask-based distributed task execution
- Head node/worker architecture
- Automatic load balancing
- Fault tolerance and recovery

### 2. GPU Support
- NVIDIA GPU monitoring via NVML
- Memory usage tracking
- Temperature and power monitoring
- Process-level GPU usage tracking
- Intelligent resource allocation

### 3. LLM Capabilities
- Model deployment across cluster
- Distributed inference
- Model sharding support
- Resource-aware deployment planning
- Multiple model support

### 4. Windows Optimization
- Windows-specific fixes and optimizations
- Firewall configuration assistance
- Event loop compatibility
- Performance tuning

### 5. Easy Worker Joining
- Automatic cluster discovery
- Interactive worker joining
- Network broadcast-based discovery
- Connection testing and validation

### 6. Web Dashboard
- Real-time cluster monitoring
- Resource utilization tracking
- GPU metrics visualization
- Task execution monitoring

### 7. REST API
- Programmatic cluster control
- LLM management endpoints
- Integration with external systems
- Cross-platform compatibility

## Dependencies

### Core Dependencies
- `dask[complete]>=2023.1.0`: Distributed computing framework
- `distributed>=2023.1.0`: Dask distributed scheduler
- `psutil>=5.8.0`: System monitoring
- `requests>=2.25.0`: HTTP requests
- `flask>=2.0.0`: Web framework
- `flask-cors>=3.0.0`: CORS support
- `pynvml>=11.4.1`: NVIDIA GPU monitoring
- `torch>=1.12.0`: PyTorch for LLM support
- `transformers>=4.20.0`: Hugging Face transformers
- `numpy>=1.21.0`: Numerical computing
- `pandas>=1.3.0`: Data manipulation

### Optional Dependencies
- `dask-cuda>=22.0.0`: CUDA support for Dask
- `cupy-cuda11x>=10.0.0`: GPU-accelerated computing
- `cudf-cu11>=22.0.0`: GPU-accelerated dataframes

### Development Dependencies
- `pytest>=6.0`: Testing framework
- `pytest-cov>=2.0`: Test coverage
- `black>=21.0`: Code formatting
- `flake8>=3.8`: Linting
- `mypy>=0.800`: Type checking

## Configuration

### Project Configuration (`pyproject.toml`)
- Package metadata and versioning
- Dependency specifications
- Build system configuration
- Development tools configuration

### Package Setup (`setup.py`)
- Package installation configuration
- Entry points definition
- Package data inclusion

## Usage Patterns

### 1. Quick Start
```bash
# Start head node
python -m pycluster.cli_enhanced --verbose

# Join worker (on another machine)
python join_worker.py
```

### 2. Programmatic Usage
```python
from pycluster import HeadNode, ClusterManager

# Start head node
with HeadNode("my-cluster") as head:
    head.start(n_local_workers=2)
    
    # Submit tasks
    cluster_manager = head.cluster_manager
    future = cluster_manager.submit_task(my_function, args)
    result = future.result()
```

### 3. LLM Deployment
```python
from pycluster import LLMClusterManager, GPUMonitor

# Initialize GPU monitoring
gpu_monitor = GPUMonitor()
gpu_monitor.start_monitoring()

# Deploy model
llm_manager = LLMClusterManager(cluster_manager)
deployment_id = llm_manager.deploy_model(
    model_name="microsoft/DialoGPT-small",
    model_size="1b",
    precision="fp16"
)

# Perform inference
response = llm_manager.inference(
    deployment_id=deployment_id,
    prompt="Hello, how are you?",
    max_tokens=50
)
```

### 4. API Usage
```python
import requests

# Start head node via API
response = requests.post('http://localhost:5000/api/cluster/start-head', json={
    'cluster_name': 'my-cluster',
    'local_workers': 2
})

# Deploy LLM model
response = requests.post('http://localhost:5000/api/llm/models/deploy', json={
    'model_name': 'microsoft/DialoGPT-small',
    'model_size': '1b',
    'precision': 'fp16'
})
```

## Architecture Patterns

### 1. Head Node/Worker Architecture
- Single head node with scheduler and dashboard
- Multiple worker nodes for task execution
- Automatic load balancing and fault tolerance

### 2. Resource Management
- GPU-aware resource allocation
- Memory usage monitoring and optimization
- Intelligent deployment planning

### 3. Service Discovery
- Network broadcast-based cluster discovery
- Automatic worker joining
- Connection health monitoring

### 4. Modular Design
- Separated concerns (cluster, LLM, GPU, network)
- Pluggable components
- Extensible architecture

## Performance Considerations

### 1. Windows Optimization
- Event loop compatibility fixes
- Firewall configuration assistance
- Performance tuning for Windows environments

### 2. GPU Utilization
- Memory-aware model loading
- Multi-GPU support
- Efficient resource allocation

### 3. Network Efficiency
- Optimized task distribution
- Connection pooling
- Minimal network overhead

## Security Features

### 1. Network Security
- Configurable firewall rules
- Secure cluster communication
- Authentication support (via API)

### 2. Resource Isolation
- Process-level isolation
- Memory limits per worker
- GPU resource allocation

## Future Enhancements

### 1. Planned Features
- Kubernetes integration
- Cloud deployment support
- Advanced monitoring and alerting
- Model versioning and management

### 2. Performance Improvements
- Enhanced GPU utilization
- Better memory management
- Optimized task scheduling

### 3. Developer Experience
- Enhanced CLI tools
- Better error handling
- Comprehensive documentation
- More examples and tutorials

This index provides a comprehensive overview of the PyCluster codebase, its components, architecture, and usage patterns. The framework is designed to be Windows-first while maintaining cross-platform compatibility, with a focus on ease of use and powerful distributed computing capabilities.
