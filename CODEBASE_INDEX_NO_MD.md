# PyCluster Codebase Index (Excluding Markdown Files)

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
│   ├── cli.py                   # Basic command-line interface
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
├── install_windows.py           # Windows installation script
├── join_worker.py               # Worker joining script
├── join_worker_simple.py        # Simplified worker joining
├── run_worker.py                # Worker execution script
└── stress_test.py               # Performance testing

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
    └── app.db                   # Application database

pycluster-dashboard/              # Web dashboard
└── dist/                        # Built dashboard files
    ├── index.html               # Main dashboard page
    ├── vite.svg                 # Vite logo
    └── assets/                  # Static assets (JS, CSS, images)
        ├── index-BiXM3Eo2.js    # Main JavaScript bundle
        ├── index-S0NQcHQ-.css   # Main CSS bundle
        └── 1747929076903.jpeg   # Dashboard image
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

**Purpose**: Head and worker node classes for cluster architecture.

**Key Classes**:
- `HeadNode`: Manages the cluster scheduler and coordination
- `WorkerNode`: Handles task execution and resource management

**Key Features**:
- Context manager support for automatic cleanup
- Resource monitoring and management
- Network configuration and discovery
- Windows-specific optimizations

### 3. LLM Serving (`llm_serving.py`)

**Purpose**: Large Language Model deployment and inference capabilities.

**Key Classes**:
- `LLMClusterManager`: Manages LLM deployments across the cluster
- `LLMWorker`: Handles model loading and inference on worker nodes
- `LLMRequest`: Request structure for LLM inference
- `LLMResponse`: Response structure for LLM outputs

**Key Features**:
- Multi-GPU model deployment
- Distributed inference
- Model caching and optimization
- Request/response streaming
- Support for popular models (DeepSeek, DialoGPT, etc.)

**Key Methods**:
- `deploy_model()`: Deploy a model across cluster workers
- `inference()`: Perform inference on deployed models
- `list_deployments()`: List active model deployments
- `scale_deployment()`: Scale model deployment across workers

### 4. GPU Monitoring (`gpu_monitor.py`)

**Purpose**: GPU resource monitoring and management.

**Key Classes**:
- `GPUMonitor`: Real-time GPU monitoring
- `LLMResourceManager`: GPU resource allocation for LLM workloads

**Key Features**:
- NVIDIA GPU monitoring via pynvml
- Memory usage tracking
- Temperature and utilization monitoring
- Resource allocation for LLM models
- Multi-GPU support

### 5. Dashboard Management (`dashboard.py`)

**Purpose**: Web dashboard for cluster monitoring and management.

**Key Classes**:
- `DashboardManager`: Manages dashboard lifecycle and configuration

**Key Features**:
- Real-time cluster monitoring
- Task execution visualization
- Resource usage graphs
- Worker status display
- Performance metrics

### 6. Command Line Interface

#### Enhanced CLI (`cli_enhanced.py`)
**Purpose**: Advanced command-line interface with comprehensive features.

**Key Features**:
- Cluster creation and management
- Worker joining and discovery
- LLM deployment commands
- GPU monitoring commands
- Interactive mode support

#### Basic CLI (`cli.py`)
**Purpose**: Simple command-line interface for basic operations.

**Key Features**:
- Head node startup
- Worker node joining
- Basic cluster operations

### 7. Windows Utilities

#### Windows Utils (`windows_utils.py`)
**Purpose**: Windows-specific cluster management utilities.

**Key Classes**:
- `WindowsClusterManager`: Windows-optimized cluster management

**Key Features**:
- Windows firewall configuration
- Service management
- Network adapter configuration
- Windows-specific optimizations

#### Windows Fixes (`windows_fixes.py`)
**Purpose**: Windows compatibility fixes and diagnostics.

**Key Features**:
- Common Windows issues resolution
- Network configuration fixes
- Firewall rule management
- Diagnostic tools

### 8. Network Utilities (`network_utils.py`)

**Purpose**: Network discovery and configuration utilities.

**Key Classes**:
- `NetworkDiscovery`: Automatic cluster node discovery

**Key Features**:
- UDP-based node discovery
- Network interface enumeration
- Port availability checking
- Cluster topology detection

### 9. Worker Discovery (`worker_discovery.py`)

**Purpose**: Automatic worker discovery and joining mechanisms.

**Key Classes**:
- `EasyWorkerJoin`: Simplified worker joining process
- `ClusterDiscovery`: Automatic cluster discovery

**Key Features**:
- Automatic cluster discovery
- One-command worker joining
- Network-based cluster detection
- Simplified setup process

### 10. Type Compatibility (`typing_fix.py`)

**Purpose**: Type compatibility fixes for different Python versions.

**Key Features**:
- Python version compatibility
- Type hint fixes
- Import compatibility
- Runtime type checking

## API Server Components

### 1. Main Application (`main.py`)

**Purpose**: Flask application entry point for the REST API server.

**Key Features**:
- Flask application setup
- Database initialization
- Route registration
- CORS configuration
- Error handling

### 2. User Management (`models/user.py`, `routes/user.py`)

**Purpose**: User authentication and management.

**Key Features**:
- User model definition
- Authentication endpoints
- User registration and login
- Session management

### 3. Cluster API (`routes/cluster.py`)

**Purpose**: REST API endpoints for cluster management.

**Key Features**:
- Cluster creation and deletion
- Worker management
- Status monitoring
- Resource allocation

### 4. LLM API (`routes/llm.py`)

**Purpose**: REST API endpoints for LLM management.

**Key Features**:
- Model deployment
- Inference requests
- Deployment scaling
- Model management

## Configuration Files

### 1. Package Configuration

#### `pyproject.toml`
- Modern Python packaging configuration
- Dependencies and optional dependencies
- Build system configuration
- Project metadata
- Entry points for CLI tools

#### `setup.py`
- Traditional Python packaging setup
- Package discovery and installation
- Entry points configuration
- Metadata and classifiers

### 2. Dependencies

#### `requirements.txt` (API Server)
- Flask and related dependencies
- Database dependencies
- CORS support
- Authentication libraries

## Example Scripts

### 1. Basic Examples

#### `basic_example.py`
- Simple cluster setup
- Basic task submission
- Worker management
- Cleanup procedures

#### `llm_example.py`
- LLM model deployment
- Inference examples
- Multi-model management
- Performance testing

#### `multi_machine_example.py`
- Multi-machine cluster setup
- Network configuration
- Cross-machine communication
- Distributed task execution

#### `deepseek_example.py`
- DeepSeek model specific examples
- Model configuration
- Inference patterns
- Performance optimization

### 2. Utility Scripts

#### `install_windows.py`
- Windows-specific installation
- Dependency management
- System configuration
- Service setup

#### `join_worker.py`
- Worker joining script
- Network discovery
- Configuration management
- Error handling

#### `join_worker_simple.py`
- Simplified worker joining
- Minimal configuration
- Quick setup
- Basic error handling

#### `run_worker.py`
- Worker execution script
- Process management
- Resource monitoring
- Logging configuration

#### `stress_test.py`
- Performance testing
- Load testing
- Benchmarking
- Resource utilization testing

## Test Suite

### 1. Core Tests (`test_pycluster.py`)
- Cluster management tests
- Node functionality tests
- Task execution tests
- Resource management tests

### 2. Enhanced Features Tests (`test_enhanced_features.py`)
- LLM serving tests
- GPU monitoring tests
- Dashboard functionality tests
- Windows-specific tests

## Build and Distribution

### 1. Build Artifacts
- `dist/`: Distribution packages (wheels and source distributions)
- `build/`: Build intermediate files
- `*.egg-info/`: Package metadata

### 2. Version Information
- Current version: 0.2.0
- Python compatibility: 3.8+
- Platform support: Windows, Linux
- Dependencies: Dask, PyTorch, Transformers, Flask

## Key Features Summary

### Distributed Computing
- Head node/worker architecture
- Task distribution and execution
- Resource management and monitoring
- Automatic scaling and load balancing

### LLM Support
- Multi-GPU model deployment
- Distributed inference
- Model caching and optimization
- Popular model support (DeepSeek, DialoGPT)

### Windows Optimization
- Native Windows support
- Firewall configuration
- Service management
- Network optimization

### Monitoring and Management
- Real-time dashboard
- REST API interface
- GPU monitoring
- Performance metrics

### Developer Experience
- Simple Python API
- Command-line tools
- Comprehensive examples
- Extensive documentation

## File Count Summary

- **Python Files**: 50+ core Python modules
- **Configuration Files**: 5+ (pyproject.toml, setup.py, requirements.txt)
- **Example Scripts**: 8+ example and utility scripts
- **Test Files**: 2+ test suites
- **API Files**: 5+ Flask application files
- **Build Artifacts**: Multiple distribution packages
- **Database Files**: SQLite database for API server
- **Static Assets**: JavaScript, CSS, and image files for dashboard

This codebase represents a comprehensive distributed computing framework with modern LLM capabilities, optimized for Windows environments while maintaining cross-platform compatibility.
