# 🧠 PyCluster Codebase Index

> **Version**: 0.3.0  
> **License**: MIT  
> **Python**: 3.8+  
> **Primary Platform**: Windows (Linux/macOS support is experimental)

---

## 📦 Project Structure

```
pycluster_working_copy/
├── pycluster/                          # Main PyCluster package
│   ├── pycluster/                      # Core Python package
│   │   ├── __init__.py                 # Package init
│   │   ├── cluster.py                  # Dask cluster management
│   │   ├── node.py                     # Head & Worker classes
│   │   ├── llm_serving.py              # LLM deployment/inference
│   │   ├── gpu_monitor.py              # GPU monitoring via NVML
│   │   ├── worker_discovery.py         # Cluster auto-discovery
│   │   ├── cli_enhanced.py             # Improved CLI
│   │   ├── dashboard.py                # Dashboard frontend integration
│   │   ├── windows_utils.py            # Windows utility helpers
│   │   ├── windows_fixes.py            # Event loop and firewall fixes
│   │   └── network_utils.py            # Network-related utilities
│   ├── examples/                       # Example scripts
│   ├── tests/                          # Test cases
│   ├── join_worker.py                  # Worker join script
│   ├── setup.py                        # Python setup file
│   ├── pyproject.toml                  # Build system config
│   └── README.md                       # Documentation
├── pycluster-api/                      # REST API backend
│   ├── src/
│   │   ├── main.py                     # Flask app entry point
│   │   ├── routes/
│   │   │   ├── cluster.py              # Cluster API
│   │   │   ├── llm.py                  # LLM API
│   │   │   └── user.py                 # User management API
│   │   ├── models/
│   │   │   └── user.py                 # User model
│   │   └── database/
│   │       └── app.db                  # SQLite database
│   └── requirements.txt                # Dependencies
└── pycluster-dashboard/                # React-based dashboard frontend
```

---

## ⚙️ Core Modules

### `cluster.py`
- `ClusterManager`: Starts/stops scheduler, adds workers, submits tasks, monitors cluster

### `node.py`
- `HeadNode`: Starts local/remote cluster
- `WorkerNode`: Connects to head and executes tasks

### `llm_serving.py`
- `LLMWorker`: GPU-bound LLM executor
- `LLMClusterManager`: Deploys & manages models across cluster

### `gpu_monitor.py`
- `GPUMonitor`: NVML-based GPU metrics
- `LLMResourceManager`: GPU memory allocator for LLMs

### `worker_discovery.py`
- `ClusterDiscovery`: Broadcast-based cluster detection
- `EasyWorkerJoin`: Interactive and auto joining support

---

## 🌐 REST API (Flask Backend)

### Key Endpoints

#### `/api/cluster`
- `GET /status`: Cluster health and worker info
- `POST /start-head`: Start head node
- `POST /start-worker`: Start a worker
- `POST /submit-task`: Submit Dask task
- `GET /metrics`: Get resource usage
- `POST /shutdown`: Shut down cluster

#### `/api/llm`
- `POST /models/deploy`: Deploy a new model
- `GET /models`: List models
- `POST /models/<id>/inference`: Run inference
- `DELETE /models/<id>`: Remove model
- `GET /gpu/status`: Current GPU state
- `GET /resources/status`: Current resource usage

---

## 🚀 Usage Examples

### Start Head Node (CLI)
```bash
python -m pycluster.cli_enhanced --verbose
```

### Join Worker Node (CLI)
```bash
python join_worker.py
```

### Python API (Head Node Programmatic)
```python
from pycluster import HeadNode

with HeadNode(cluster_name=\"demo\") as head:
    head.start(n_local_workers=2)
```

### Submit Task (Python API)
```python
future = head.cluster_manager.submit_task(lambda x: x * 2, 10)
print(future.result())  # Output: 20
```

### Deploy & Run LLM (Python API)
```python
from pycluster import LLMClusterManager, GPUMonitor

monitor = GPUMonitor()
monitor.start_monitoring()

llm_mgr = LLMClusterManager(head.cluster_manager)
deployment_id = llm_mgr.deploy_model(\"microsoft/DialoGPT-small\", \"1b\", precision=\"fp16\")

response = llm_mgr.inference(deployment_id, prompt=\"Hi there!\", max_tokens=50)
print(response.text)
```

---

## 📊 Dashboard Features

- Real-time worker stats
- GPU and CPU usage graphs
- Task queue monitoring
- LLM deployment status

---

## 🛠 Dev & Test

### Example Scripts:
- `examples/basic_example.py`
- `examples/llm_example.py`
- `examples/deepseek_example.py`
- `examples/multi_machine_example.py`

### Testing:
```bash
pytest tests/
```

---

## 🔮 Planned Enhancements

- [ ] Kubernetes & Docker orchestration
- [ ] Cloud integration (Azure, GCP, AWS)
- [ ] Node authentication and secure RPC
- [ ] Model versioning and registry
- [ ] Auto-scaling workers

---

## 📥 Dependencies

### Core
- `dask[complete]`, `distributed`
- `torch`, `transformers`, `pynvml`, `flask`, `requests`, `psutil`

### Optional
- `dask-cuda`, `cupy`, `cudf` (for GPU dataframe acceleration)

### Dev
- `pytest`, `black`, `flake8`, `mypy`

---

## ✅ Summary

PyCluster is a modern, Windows-optimized Dask-based cluster framework featuring:

- Head/worker node architecture
- Automatic cluster discovery
- GPU-aware LLM deployment
- Interactive dashboard
- Full REST API
- Plug-and-play examples

> Built for developers who want distributed AI compute across Windows (or hybrid) clusters with minimal friction.
