# PyCluster

A Windows-based Python clustering package with dashboard support that provides a simple interface for creating distributed computing clusters using Dask as the underlying framework.

## Features

- **Head Node/Worker Architecture**: Similar to Ray but with better Windows compatibility
- **Built-in Dashboard**: Real-time monitoring and visualization of cluster activity
- **Easy Setup**: Simple Python API for cluster management
- **Windows Compatible**: Designed specifically to work well on Windows systems
- **Scalable**: From single machine to multi-machine clusters
- **Python Native**: Integrates seamlessly with existing Python workflows

## Installation

```bash
pip install dask[complete] distributed psutil requests
```

Then install PyCluster:

```bash
cd pycluster
pip install -e .
```

## Quick Start

### Basic Single-Machine Cluster

```python
from pycluster import HeadNode, DashboardManager

# Start a head node with local workers
with HeadNode(cluster_name="my-cluster") as head:
    result = head.start(n_local_workers=4)
    
    if result["status"] == "success":
        print(f"Cluster started: {result['dashboard_address']}")
        
        # Submit tasks
        cluster = head.cluster_manager
        future = cluster.submit_task(lambda x: x**2, 10)
        print(f"Result: {future.result()}")  # Output: 100
```

### Multi-Machine Cluster

**On the head node machine:**

```python
from pycluster import HeadNode

head = HeadNode(host="0.0.0.0")  # Listen on all interfaces
result = head.start()

conn_info = head.get_connection_info()
print(f"Workers should connect to: {conn_info['scheduler_address']}")
print(f"Dashboard available at: {conn_info['dashboard_url']}")
```

**On worker machines:**

```python
from pycluster import WorkerNode

worker = WorkerNode(scheduler_address="tcp://192.168.1.100:8786")
worker.start(n_workers=2)
```

### Command Line Interface

Start a head node:
```bash
pycluster-head --cluster-name my-cluster --local-workers 2 --open-dashboard
```

Start workers:
```bash
pycluster-worker --scheduler tcp://192.168.1.100:8786 --workers 2
```

## Dashboard

The dashboard provides real-time monitoring of your cluster:

- **Task Stream**: Visualize task execution over time
- **Worker Status**: Monitor CPU, memory, and network usage
- **Progress Tracking**: See computation progress
- **Resource Utilization**: Track cluster resource usage

Access the dashboard at `http://<head-node-ip>:8787/status`

## Examples

See the `examples/` directory for complete examples:

- `basic_example.py`: Single-machine cluster with task submission
- `multi_machine_example.py`: Multi-machine cluster setup

## API Reference

### HeadNode

```python
head = HeadNode(
    cluster_name="my-cluster",
    host="0.0.0.0",
    scheduler_port=8786,
    dashboard_port=8787
)

# Start the head node
result = head.start(n_local_workers=2)

# Get connection information
conn_info = head.get_connection_info()

# Get cluster status
status = head.get_cluster_status()
```

### WorkerNode

```python
worker = WorkerNode(
    scheduler_address="tcp://192.168.1.100:8786",
    worker_name="my-worker"
)

# Start workers
result = worker.start(
    n_workers=2,
    threads_per_worker=4,
    memory_limit="4GB"
)

# Get worker status
status = worker.get_status()
```

### ClusterManager

```python
# Submit individual tasks
future = cluster.submit_task(my_function, arg1, arg2)
result = future.result()

# Map function over data
futures = cluster.map_tasks(my_function, data_list)
results = [f.result() for f in futures]
```

### DashboardManager

```python
dashboard = DashboardManager("http://192.168.1.100:8787")

# Check if accessible
if dashboard.is_accessible():
    # Open in browser
    dashboard.open_in_browser()
    
    # Get cluster information
    status = dashboard.get_cluster_status()
    workers = dashboard.get_worker_info()
```

## Requirements

- Python 3.8+
- Dask 2023.1.0+
- Windows, Linux, or macOS
- Network connectivity between machines (for multi-machine clusters)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

