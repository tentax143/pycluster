# PyCluster Codebase Index

## Overview

PyCluster is a comprehensive Windows-first distributed computing framework built on Dask that provides enterprise-grade capabilities for distributed computing, LLM deployment, GPU management, and cluster orchestration. The project consists of three main components:

1. **pycluster** - Core distributed computing framework
2. **pycluster-api** - REST API server for cluster management
3. **pycluster-dashboard** - Web-based dashboard for monitoring and management

## Project Structure

```
pycluster/
├── pycluster/                    # Main package directory
│   ├── __init__.py              # Package initialization and exports
│   ├── cluster.py               # Core cluster management
│   ├── node.py                  # Head and worker node classes
│   ├── llm_serving.py           # LLM deployment and inference
│   ├── gpu_monitor.py           # GPU monitoring and resource management
│   ├── dashboard.py             # Dashboard management utilities
│   ├── cli_enhanced.py          # Enhanced command-line interface
│   ├── worker_discovery.py      # Auto-discovery and worker joining
│   ├── windows_utils.py         # Windows-specific utilities
│   ├── windows_fixes.py         # Windows compatibility fixes
│   ├── network_utils.py         # Network discovery and utilities
│   └── typing_fix.py            # Type compatibility fixes
├── examples/                     # Usage examples
│   ├── basic_example.py         # Basic cluster setup
│   ├── llm_example.py           # LLM deployment example
│   ├── multi_machine_example.py # Multi-machine cluster setup
│   └── deepseek_example.py      # DeepSeek model example
├── tests/                        # Test suite
│   ├── test_pycluster.py        # Main test suite
│   └── test_enhanced_features.py # Enhanced features tests
├── setup.py                     # Package setup configuration
├── pyproject.toml               # Modern Python packaging
└── README.md                    # Comprehensive documentation

pycluster-api/                    # REST API server
├── src/
│   ├── main.py                  # Flask application entry point
│   ├── models/
│   │   └── user.py              # User model for authentication
│   └── routes/
│       ├── cluster.py           # Cluster management API endpoints
│       ├── llm.py               # LLM management API endpoints
│       └── user.py              # User management API endpoints
├── requirements.txt             # API server dependencies
└── database/                    # SQLite database storage

pycluster-dashboard/              # Web dashboard
└── dist/                        # Built dashboard files
    ├── index.html               # Main dashboard page
    └── assets/                  # Static assets (JS, CSS, images)
```

## Core Components

### 1. Cluster Management (`cluster.py`)

**Purpose**: Core cluster management functionality using Dask as the underlying framework.

**Key Classes**:
- `ClusterManager`: Main cluster management class providing high-level interface for creating and managing distributed computing clusters

**Key Features**:
- Head node/worker architecture
- Task submission and mapping
- Cluster information and status
- Resource management
- Automatic cleanup and shutdown

**Key Methods**:
- `start_head_node()`: Start the head node (scheduler) of the cluster
- `add_worker()`: Add worker nodes to the cluster
- `submit_task()`: Submit individual tasks to the cluster
- `map_tasks()`: Map a function over data using the cluster
- `get_cluster_info()`: Get comprehensive cluster information
- `shutdown()`: Clean shutdown of cluster resources

### 2. Node Management (`node.py`)

**Purpose**: Simplified interfaces for starting and managing head and worker nodes.

**Key Classes**:
- `HeadNode`: Simplified interface for starting and managing a head node
- `WorkerNode`: Simplified interface for starting and managing worker nodes

**Key Features**:
- Easy cluster startup with context managers
- Automatic discovery broadcasting
- Connection information management
- Status monitoring
- Graceful shutdown handling

**Key Methods**:
- `start()`: Start the node with specified configuration
- `get_connection_info()`: Get connection information for workers
- `get_cluster_status()`: Get current cluster status
- `shutdown()`: Graceful shutdown of the node

### 3. LLM Serving (`llm_serving.py`)

**Purpose**: LLM deployment and distributed inference capabilities.

**Key Classes**:
- `LLMWorker`: Worker class for running LLM inference on specific GPUs
- `LLMClusterManager`: Manage LLM serving across a Dask cluster
- `LLMRequest`: Request structure for LLM inference
- `LLMResponse`: Response structure from LLM inference
- `LLMModelInfo`: Information about loaded LLM models

**Key Features**:
- Multi-model support with model sharding
- GPU resource allocation and management
- Distributed inference across multiple workers
- Request routing and load balancing
- Model deployment and lifecycle management

**Key Methods**:
- `deploy_model()`: Deploy an LLM model across the cluster
- `inference()`: Perform inference using a deployed model
- `load_model()`: Load an LLM model onto assigned GPUs
- `get_deployment_status()`: Get status of model deployments
- `undeploy_model()`: Remove a model deployment

### 4. GPU Monitoring (`gpu_monitor.py`)

**Purpose**: Enhanced GPU monitoring and resource management for LLM workloads.

**Key Classes**:
- `GPUMonitor`: Monitor NVIDIA GPUs using NVML (NVIDIA Management Library)
- `LLMResourceManager`: Manage resources for LLM workloads across the cluster

**Key Features**:
- Real-time GPU metrics collection (memory, utilization, temperature, power)
- System-wide metrics monitoring
- LLM memory requirement estimation
- Resource allocation and planning
- Historical metrics tracking

**Key Methods**:
- `get_gpu_info()`: Get detailed information about specific GPUs
- `get_system_metrics()`: Get comprehensive system metrics
- `start_monitoring()`: Start continuous monitoring in background
- `estimate_llm_memory_requirements()`: Estimate memory needs for LLM models
- `plan_llm_deployment()`: Plan LLM deployment across available GPUs

### 5. Dashboard Management (`dashboard.py`)

**Purpose**: Manager for accessing and interacting with the Dask dashboard.

**Key Classes**:
- `DashboardManager`: Manager for accessing and interacting with the Dask dashboard

**Key Features**:
- Dashboard accessibility checking
- Browser integration
- Cluster status retrieval
- Worker information gathering
- Task stream monitoring
- Comprehensive reporting

**Key Methods**:
- `is_accessible()`: Check if dashboard is accessible
- `open_in_browser()`: Open dashboard in default web browser
- `get_cluster_status()`: Get cluster status from dashboard API
- `get_worker_info()`: Get worker information from dashboard
- `generate_dashboard_report()`: Generate comprehensive dashboard report

### 6. Worker Discovery (`worker_discovery.py`)

**Purpose**: Enhanced worker discovery and auto-join functionality.

**Key Classes**:
- `ClusterDiscovery`: Handles automatic discovery of PyCluster head nodes
- `EasyWorkerJoin`: Simplified worker joining with multiple discovery methods

**Key Features**:
- UDP broadcasting for cluster announcement
- Network scanning for cluster discovery
- Interactive cluster selection
- Connection testing and validation
- Automatic worker joining scripts

**Key Methods**:
- `start_broadcasting()`: Start broadcasting cluster information
- `start_discovery()`: Start listening for cluster broadcasts
- `discover_clusters()`: Discover available clusters on network
- `join_cluster_interactive()`: Interactive cluster joining
- `test_connection()`: Test connectivity to scheduler

### 7. Enhanced CLI (`cli_enhanced.py`)

**Purpose**: Enhanced command-line interface with Windows support and diagnostics.

**Key Features**:
- Comprehensive argument parsing
- Windows-specific optimizations
- Built-in diagnostics
- Head node and worker node management
- Verbose logging and error handling

**Key Commands**:
- `head`: Start head node
- `worker`: Start worker node
- `--diagnose`: Run Windows diagnostics
- `--verbose`: Enable verbose logging

## API Server (pycluster-api)

### Structure
- **Flask-based REST API** with CORS support
- **Modular route organization** with blueprints
- **SQLite database** for user management
- **Static file serving** for dashboard integration

### Key Routes

#### Cluster Management (`/api/cluster/`)
- `GET /status`: Get current cluster status
- `POST /start-head`: Start a head node
- `POST /start-worker`: Start worker node(s)
- `POST /connect`: Connect to existing cluster
- `POST /shutdown`: Shutdown the cluster
- `POST /submit-task`: Submit tasks to cluster
- `GET /dashboard-info`: Get dashboard information
- `GET /workers`: Get detailed worker information
- `GET /metrics`: Get cluster performance metrics
- `GET /health`: Health check endpoint

#### LLM Management (`/api/llm/`)
- `GET /health`: Check LLM service health
- `GET /gpu/status`: Get detailed GPU status
- `GET /models`: List deployed LLM models
- `POST /models/deploy`: Deploy a new LLM model
- `GET /models/<id>/status`: Get model deployment status
- `POST /models/<id>/inference`: Perform model inference
- `DELETE /models/<id>`: Undeploy a model
- `GET /resources/status`: Get resource allocation status
- `GET /models/available`: Get list of available models

#### User Management (`/api/users/`)
- `GET /users`: Get all users
- `POST /users`: Create new user
- `GET /users/<id>`: Get specific user
- `PUT /users/<id>`: Update user
- `DELETE /users/<id>`: Delete user

## Dashboard (pycluster-dashboard)

### Structure
- **React-based web application** (built with Vite)
- **Static file distribution** through Flask API server
- **Real-time monitoring** of cluster status
- **Interactive management** of cluster resources

### Features
- Cluster status monitoring
- Worker node management
- Task execution tracking
- GPU resource monitoring
- LLM model deployment interface
- Performance metrics visualization

## Examples and Usage Patterns

### 1. Basic Cluster Setup (`basic_example.py`)
```python
from pycluster import HeadNode, DashboardManager

with HeadNode(cluster_name="example-cluster") as head:
    result = head.start(n_local_workers=2)
    
    if result["status"] == "success":
        # Submit tasks
        cluster = head.cluster_manager
        future = cluster.submit_task(lambda x: x**2, 10)
        print(f"Result: {future.result()}")
```

### 2. LLM Deployment (`llm_example.py`)
```python
from pycluster import HeadNode, LLMClusterManager, GPUMonitor

# Initialize GPU monitoring
gpu_monitor = GPUMonitor()
resource_manager = LLMResourceManager(gpu_monitor)

with HeadNode(cluster_name="llm-cluster") as head:
    head.start(n_local_workers=2)
    
    # Deploy LLM model
    llm_manager = LLMClusterManager(head.cluster_manager)
    deployment_id = llm_manager.deploy_model("deepseek-ai/deepseek-coder-7b-instruct-v1.5")
    
    # Perform inference
    response = llm_manager.inference(deployment_id, "Write Python code for sorting")
```

### 3. Multi-Machine Setup (`multi_machine_example.py`)
```python
# Head node
head = HeadNode(cluster_name="multi-machine-cluster", host="0.0.0.0")
result = head.start(n_local_workers=1)

# Worker node
worker = WorkerNode(scheduler_address="tcp://192.168.1.100:8786")
worker.start(n_workers=2, threads_per_worker=2)
```

## Testing

### Test Structure (`tests/`)
- **Unit tests** for individual components
- **Integration tests** for complete workflows
- **Mock testing** for GPU and network components
- **Windows-specific tests** for compatibility

### Key Test Classes
- `TestClusterManager`: Core cluster functionality
- `TestHeadNode`: Head node management
- `TestWorkerNode`: Worker node management
- `TestDashboardManager`: Dashboard integration
- `TestWindowsClusterManager`: Windows utilities
- `TestNetworkDiscovery`: Network discovery
- `TestIntegration`: End-to-end workflows

## Dependencies

### Core Dependencies
- **dask[complete]>=2023.1.0**: Distributed computing framework
- **distributed>=2023.1.0**: Dask distributed components
- **psutil>=5.8.0**: System monitoring
- **requests>=2.25.0**: HTTP client
- **flask>=2.0.0**: Web framework
- **flask-cors>=3.0.0**: CORS support

### Optional Dependencies
- **GPU Support**: `pynvml>=11.0.0`, `nvidia-ml-py>=11.0.0`
- **LLM Support**: `torch>=1.9.0`, `transformers>=4.20.0`, `accelerate>=0.20.0`
- **Development**: `pytest>=6.0.0`, `black>=22.0.0`, `flake8>=4.0.0`

## Key Features and Capabilities

### 1. Distributed Computing
- **Head Node/Worker Architecture**: Single scheduler with multiple worker nodes
- **Dask Integration**: Built on proven distributed computing framework
- **Automatic Load Balancing**: Intelligent task distribution
- **Fault Tolerance**: Automatic recovery from worker failures
- **Scalable Design**: Add/remove workers dynamically

### 2. LLM Integration
- **Multi-Model Support**: Deploy multiple LLM models simultaneously
- **Model Sharding**: Tensor parallelism across multiple GPUs
- **Precision Control**: FP16, FP32, and mixed precision support
- **Resource Planning**: Intelligent GPU memory allocation
- **Distributed Inference**: Load balancing across model replicas

### 3. GPU Management
- **NVIDIA GPU Support**: Direct NVML integration
- **Real-time Monitoring**: Live GPU metrics collection
- **Memory Tracking**: VRAM usage and allocation monitoring
- **Resource Allocation**: Intelligent GPU assignment for models
- **Multi-GPU Support**: Distributed models across multiple GPUs

### 4. Windows Optimization
- **Windows-First Design**: Built specifically for Windows
- **Event Loop Compatibility**: Fixed asyncio event loop issues
- **Firewall Management**: Automatic Windows Firewall configuration
- **Service Creation**: Windows service script generation
- **Diagnostics**: Comprehensive Windows troubleshooting

### 5. Network & Discovery
- **UDP Broadcasting**: Automatic cluster announcement
- **Network Scanning**: Active network scanning for clusters
- **Service Discovery**: Zero-configuration cluster discovery
- **Connection Testing**: Built-in connectivity validation
- **Firewall Integration**: Automatic firewall rule creation

### 6. Monitoring & Dashboard
- **Real-time Monitoring**: Live cluster health monitoring
- **Web Dashboard**: Full Dask dashboard access with custom metrics
- **Interactive Charts**: Real-time data visualization
- **Multi-page Interface**: Status, workers, tasks, system pages
- **Performance Analytics**: Comprehensive performance metrics

### 7. REST API
- **Cluster Management API**: Start/stop/restart cluster operations
- **LLM Management API**: Deploy models and perform inference via HTTP
- **User Management API**: User account and session management
- **Health Checks**: Comprehensive health monitoring endpoints

## Installation and Setup

### Basic Installation
```bash
pip install dask[complete] distributed psutil requests
cd pycluster
pip install -e .
```

### GPU Support
```bash
pip install pycluster[gpu]
```

### Development Installation
```bash
pip install -e .[dev]
```

### API Server Setup
```bash
cd pycluster-api
pip install -r requirements.txt
python src/main.py
```

## Command Line Usage

### Start Head Node
```bash
pycluster --cluster-name my-cluster --local-workers 2 --verbose
```

### Start Worker Node
```bash
pycluster worker --scheduler tcp://192.168.1.100:8786 --workers 2
```

### Run Diagnostics
```bash
pycluster --diagnose
```

## Configuration

### Cluster Configuration
- **JSON-based configuration** files
- **Environment variables** support
- **Command-line options** for flexibility
- **Default settings** with sensible defaults
- **Configuration validation** and error handling

### Windows Configuration
- **Automatic configuration** directory creation
- **Firewall rule management**
- **Service configuration** for Windows services
- **Network interface detection**
- **Port conflict resolution**

## Performance and Scalability

### Scalability Features
- **Horizontal Scaling**: Add unlimited worker nodes
- **Vertical Scaling**: Scale individual node resources
- **Auto-scaling**: Automatic resource scaling
- **Load Distribution**: Intelligent load balancing
- **Resource Pooling**: Shared resource management

### Performance Optimization
- **Memory Optimization**: Efficient memory usage patterns
- **CPU Optimization**: Multi-core CPU utilization
- **Network Optimization**: Optimized network communication
- **GPU Optimization**: Efficient GPU resource utilization
- **Caching Systems**: Intelligent caching for performance

## Security and Compliance

### Security Features
- **Authentication**: User authentication and authorization
- **Encryption**: Data encryption in transit and at rest
- **Access Control**: Role-based access control
- **Audit Logging**: Comprehensive audit trails
- **Secure Communication**: Encrypted cluster communication

### Compliance Features
- **Data Privacy**: GDPR-compliant data handling
- **Audit Trails**: Complete operation audit trails
- **Configuration Management**: Secure configuration handling
- **Backup & Recovery**: Secure backup and recovery procedures

## Future Roadmap

### Planned Features
- **Kubernetes Integration**: Native Kubernetes support
- **Cloud Deployment**: Multi-cloud deployment support
- **Advanced Monitoring**: Enhanced monitoring and alerting
- **Model Versioning**: Comprehensive model versioning
- **AutoML Integration**: Automated machine learning integration

### Performance Enhancements
- **Enhanced GPU Utilization**: Improved GPU resource utilization
- **Better Memory Management**: Advanced memory management
- **Optimized Scheduling**: Improved task scheduling algorithms
- **Network Optimization**: Enhanced network performance
- **Caching Improvements**: Advanced caching mechanisms

## Support and Documentation

### Documentation
- **Comprehensive README**: Complete capabilities overview
- **API Documentation**: Detailed API reference
- **Examples**: Multiple usage examples
- **Troubleshooting**: Common issues and solutions
- **Windows Guide**: Windows-specific setup and configuration

### Community and Support
- **GitHub Repository**: Source code and issue tracking
- **Issue Tracker**: Bug reports and feature requests
- **Documentation**: Extensive documentation and examples
- **Examples**: Multiple working examples and tutorials

---

This codebase index provides a comprehensive overview of the PyCluster project structure, components, and capabilities. The framework represents a complete solution for distributed computing with LLM support, optimized for Windows environments while maintaining cross-platform compatibility.
