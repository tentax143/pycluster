# PyCluster: Distributed Python Computing for Windows with LLM & GPU Support

![PyCluster Dashboard Screenshot]([https://github.com/pycluster/pycluster/blob/main/docs/dashboard_screenshot.png?raw=true](https://github.com/tentax143/pycluster/blob/main/pycluster-dashboard/dist/assets/1747929076903.jpeg?raw=true))

PyCluster is a robust, Windows-first distributed computing framework built on Dask, designed to simplify the creation and management of Python clusters. It features a head node/worker architecture, comprehensive GPU monitoring, and powerful capabilities for deploying and serving Large Language Models (LLMs) across your cluster. With its intuitive web dashboard and new easy worker joining features, PyCluster makes distributed computing accessible and efficient.

## ✨ Key Features

-   **Head Node / Worker Architecture**: Easily designate one system as the head node and others as workers for distributed task execution.
-   **Windows Optimized**: Designed with Windows compatibility in mind, including specific fixes, performance optimizations, and firewall considerations.
-   **Enhanced Web Dashboard**: A modern, intuitive React-based dashboard for real-time monitoring of cluster health, worker status, resource utilization (CPU, Memory, Network, Disk), and GPU metrics.
-   **NVIDIA GPU Integration**: Comprehensive monitoring of NVIDIA GPUs (temperature, memory usage, utilization) and intelligent resource allocation for LLM workloads.
-   **Large Language Model (LLM) Support**: Built-in capabilities for deploying and serving LLMs (e.g., DeepSeek, Code Llama) across your cluster, supporting distributed inference and model sharding.
-   **Easy Worker Joining**: New auto-discovery and interactive tools simplify the process of adding worker nodes to your cluster, eliminating the need for manual IP address configuration.
-   **REST API**: A Flask-based API for programmatic control and integration with other systems.
-   **Comprehensive Testing**: Robust test suite ensuring reliability and performance.
-   **Detailed Documentation**: Extensive guides for setup, usage, and troubleshooting.

## 🚀 Quick Start

### 1. Installation

**Prerequisites:**
-   Python 3.8+ (Anaconda/Miniconda recommended for environment management)
-   Windows Operating System (Linux/macOS support is experimental)
-   (Optional for GPU support) NVIDIA GPU with CUDA drivers installed

**Clone the repository (or extract the provided zip):**

```bash
git clone https://github.com/pycluster/pycluster.git
cd pycluster
```

**Create and activate a Python environment:**

```bash
conda create -n pycluster_env python=3.9
conda activate pycluster_env
```

**Install PyCluster and its dependencies:**

```bash
pip install .
# For GPU support (requires NVIDIA drivers and CUDA toolkit):
pip install .[gpu]
```

### 2. Start the Head Node

Open a command prompt (preferably as Administrator to avoid firewall issues) and run:

```bash
python -m pycluster.cli_enhanced --verbose
```

This will start the Dask scheduler, a local Dask worker (by default 2 workers), and the web dashboard. You will see output similar to this:

```
✓ Head node started successfully!
  Cluster: pycluster
  Scheduler: tcp://YOUR_HEAD_NODE_IP:8786
  Dashboard: http://YOUR_HEAD_NODE_IP:8787
  Workers: 2

Windows users:
  - Dashboard may take a moment to load
  - If connection fails, check Windows Firewall
  - Run as Administrator if needed

Press Ctrl+C to stop the cluster
```

Access the dashboard in your web browser at `http://localhost:8787` (or `http://YOUR_HEAD_NODE_IP:8787`).

### 3. Join Worker Nodes (Easy Way!)

On any machine you want to add as a worker (ensure PyCluster is installed on it), open a new command prompt, activate your Python environment, navigate to the `pycluster` directory, and run:

```bash
python join_worker.py
```

This script will automatically discover available PyCluster head nodes on your network and guide you through an interactive selection process. You can also use other options:

-   **Auto-join first available**: `python join_worker.py --auto`
-   **Join by cluster name**: `python join_worker.py --cluster-name "my-cluster"`
-   **List available clusters**: `python join_worker.py --list`
-   **Manual join (if auto-discovery fails)**: `python join_worker.py --scheduler tcp://YOUR_HEAD_NODE_IP:8786`

Once connected, the worker will appear in your dashboard.

### 4. Run a Distributed Task (Example)

Create a Python file (e.g., `my_task.py`):

```python
# my_task.py

import time
from pycluster import ClusterManager

def square(x):
    time.sleep(1) # Simulate work
    return x * x

if __name__ == "__main__":
    # Connect to the running cluster
    # If running on the same machine as head node, use localhost
    # Otherwise, use the head node's actual IP
    cluster_manager = ClusterManager(scheduler_address="tcp://localhost:8786")
    
    print("Submitting tasks to the cluster...")
    futures = cluster_manager.submit_tasks([square for _ in range(10)], range(10))
    
    results = [f.result() for f in futures]
    print(f"Results: {results}")
    
    # You can also use the LLMClusterManager for LLM-specific tasks
    # from pycluster import LLMClusterManager
    # llm_manager = LLMClusterManager(scheduler_address="tcp://localhost:8786")
    # response = llm_manager.generate_text("Tell me a joke.")
    # print(response)
```

Run the script:

```bash
python my_task.py
```

Observe the tasks being processed by your workers in the PyCluster dashboard!

## ⚙️ Project Structure

```
pycluster/
├── pycluster/                 # Core PyCluster Python package
│   ├── __init__.py            # Package initialization, version (v0.3.1)
│   ├── cli.py                 # Original CLI (legacy)
│   ├── cli_enhanced.py        # Enhanced CLI with Windows fixes & diagnostics
│   ├── cluster.py             # Core Dask cluster management
│   ├── dashboard.py           # Dashboard integration
│   ├── gpu_monitor.py         # NVIDIA GPU monitoring
│   ├── llm_serving.py         # LLM deployment and serving logic
│   ├── network_utils.py       # Network utilities for discovery
│   ├── node.py                # HeadNode and WorkerNode classes
│   ├── windows_fixes.py       # Windows-specific bug fixes and optimizations
│   ├── windows_utils.py       # Windows utility functions
│   └── worker_discovery.py    # Auto-discovery and easy join logic
├── examples/                  # Example usage scripts
├── tests/                     # Pytest test suite
├── pycluster-api/             # Flask REST API backend
│   ├── src/                   # API source code
│   └── requirements.txt       # API dependencies
├── pycluster-dashboard/       # React web dashboard frontend
│   ├── public/                # Static assets
│   ├── src/                   # React source code
│   └── package.json           # Node.js dependencies
├── join_worker.py             # Standalone script for easy worker joining
├── setup.py                   # Python package setup script
├── pyproject.toml             # Project configuration
├── README.md                  # This file
└── DOCUMENTATION.md           # Comprehensive project documentation
```

## ⚠️ Troubleshooting

### 1. `Failed to start head node: Timed out trying to connect...`

This usually indicates a port blocking issue or another service using the same ports. 

-   **Solution**: 
    -   **Run as Administrator**: Open your command prompt/PowerShell as an Administrator and try again.
    -   **Windows Firewall**: Ensure inbound rules are created for TCP ports `8786` (scheduler), `8787` (dashboard), and `5000` (Flask API). You can use the `--diagnose` flag for help.
    -   **Port Conflict**: Check if another application is already using these ports.

### 2. `TypeError: unhashable type: 'list'`

This often points to a version incompatibility between Python's `typing` module and `dask.distributed`.

-   **Solution**: 
    -   Update `typing_extensions`: `pip install --upgrade typing_extensions`
    -   Update Dask and Distributed: `pip install --upgrade dask distributed`

### 3. `WorkerNode.start() got an unexpected keyword argument 'scheduler_address'` or `AttributeError: 'WorkerNode' object has no attribute 'connect_to_cluster'`

These errors indicate that your PyCluster installation is outdated or corrupted.

-   **Solution**: 
    -   **Clean Reinstallation**: 
        1.  Navigate to the root `pycluster` directory.
        2.  Run `pip uninstall pycluster` (confirm with `y`).
        3.  Delete any `__pycache__` folders in the `pycluster` directory.
        4.  Run `pip install .` (or `pip install .[gpu]`) to reinstall the latest version.

### 4. Dashboard Not Accessible

-   **Solution**: 
    -   **Firewall**: Most common cause. Ensure ports `8787` and `5000` are open in your Windows Firewall.
    -   **Run as Administrator**: Try starting the head node as an administrator.
    -   **Correct URL**: Use `http://localhost:8787` if running on the same machine.
    -   **Flask API**: Ensure the Flask API is running. You can test its health at `http://localhost:5000/api/cluster/health`.

### 5. Workers Not Joining / No Clusters Found

-   **Solution**: 
    -   **Head Node Running**: Ensure your head node is actively running and broadcasting.
    -   **Network Connectivity**: Verify that the worker machine can reach the head node machine (e.g., by `ping`ing the head node's IP).
    -   **Windows Firewall (Worker)**: Ensure the worker's firewall allows outbound connections and inbound connections on UDP port `8788` for discovery.
    -   **Run as Administrator**: Try running the `join_worker.py` script as an administrator.

For more detailed troubleshooting and advanced configurations, please refer to the `DOCUMENTATION.md` file.

## 🤝 Contributing

We welcome contributions to PyCluster! If you have suggestions, bug reports, or want to contribute code, please open an issue or pull request on the GitHub repository.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

**Happy Distributed Computing with PyCluster!** 🎉


