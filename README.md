````markdown
# 🧠 PyCluster Codebase Index

> **Version**: 0.3.0  
> **License**: MIT  
> **Python**: 3.8+  
> **Primary Platform**: Windows (Linux/macOS support is experimental)

---

## 🚀 What is PyCluster?

**PyCluster** is a lightweight, Windows-first distributed computing framework built on **Dask**, designed to simplify the creation and management of high-performance Python compute clusters. Its core focus is:

- Windows-native execution
- Real-time GPU resource monitoring
- Automatic cluster node discovery
- Seamless deployment and inference of **Large Language Models (LLMs)**
- REST API and web dashboard support

Whether you're running a school computer lab or setting up a local LLM inference farm, PyCluster makes multi-machine orchestration intuitive and efficient — without needing Docker, Kubernetes, or Linux.

---

## 📆 Project Structure

```
pycluster_working_copy/
├── pycluster/                          # Main PyCluster package
│   ├── pycluster/                      # Core Python logic
│   │   ├── __init__.py                 # Initialization
│   │   ├── cluster.py                  # Dask-based cluster management
│   │   ├── node.py                     # HeadNode & WorkerNode APIs
│   │   ├── llm_serving.py              # LLM deployment and inference
│   │   ├── gpu_monitor.py              # GPU metrics via NVML
│   │   ├── worker_discovery.py         # Auto-discovery protocol
│   │   ├── cli_enhanced.py             # CLI entry point (Windows compatible)
│   │   ├── dashboard.py                # Dashboard utilities
│   │   ├── windows_utils.py            # Windows system functions
│   │   ├── windows_fixes.py            # Windows-specific patches
│   │   └── network_utils.py            # IP and port utilities
│   ├── examples/                       # Sample scripts
│   ├── tests/                          # Unit and integration tests
│   ├── join_worker.py                  # Script to connect workers
│   ├── setup.py                        # Packaging config
│   ├── pyproject.toml                  # Build config
│   └── README.md                       # Main documentation
├── pycluster-api/                      # Flask REST backend
│   ├── src/
│   │   ├── main.py                     # API app
│   │   ├── routes/                     # Endpoints
│   │   │   ├── cluster.py              # Cluster operations
│   │   │   ├── llm.py                  # Model-related endpoints
│   │   │   └── user.py                 # Auth/User handling
│   │   ├── models/
│   │   │   └── user.py                 # User model
│   │   └── database/
│   │       └── app.db                  # SQLite instance
│   └── requirements.txt                # Dependencies
└── pycluster-dashboard/                # React dashboard frontend
```

---

## ⚙️ Core Modules and Responsibilities

### 1. `cluster.py` — ClusterManager
- Starts scheduler
- Adds/removes workers
- Maps or submits tasks
- Tracks cluster status

### 2. `node.py` — Node Management
- `HeadNode`: Bootstraps scheduler, exposes dashboard
- `WorkerNode`: Registers with scheduler, executes tasks

### 3. `llm_serving.py` — LLM Deployment & Inference
- `LLMClusterManager`: Deploys models to workers
- `LLMWorker`: Loads and serves models
- Uses HuggingFace Transformers and PyTorch under the hood

### 4. `gpu_monitor.py` — GPU Resource Management
- Uses NVIDIA NVML via `pynvml`
- Collects per-GPU memory, usage, temp, power, and process metrics
- `LLMResourceManager` estimates memory and deploys models accordingly

### 5. `worker_discovery.py` — Auto Node Discovery
- UDP broadcast-based LAN discovery
- Interactive or silent join via hostname/IP
- Validates connections before registration

### 6. `cli_enhanced.py` — Enhanced CLI
- Cross-platform (esp. Windows-friendly)
- Example: `python -m pycluster.cli_enhanced`
- Can start scheduler or worker with flags

### 7. `dashboard.py`
- Serves external React dashboard
- Can also launch embedded Dask dashboard

---

## 🌐 Flask API (`pycluster-api/`)

### Main App (`main.py`)
- Flask app with CORS enabled
- Initializes SQLite, ClusterManager, GPU monitor
- Static hosting support for dashboard

### Routes

#### `/api/cluster`
- `GET /status`: Get cluster info
- `POST /start-head`: Start scheduler
- `POST /start-worker`: Register worker
- `POST /submit-task`: Submit function or workload
- `POST /shutdown`: Stop head or cluster
- `GET /metrics`: Live system and GPU metrics

#### `/api/llm`
- `POST /models/deploy`: Deploy HuggingFace model
- `GET /models`: List current deployments
- `POST /models/<id>/inference`: Run prompt
- `DELETE /models/<id>`: Remove model
- `GET /gpu/status`: GPU state

---

## 🧪 Example Workflows

### Start a Cluster via CLI
```bash
python -m pycluster.cli_enhanced --verbose
```

### Add a Worker
```bash
python join_worker.py
```

### Python Cluster Start
```python
from pycluster import HeadNode

with HeadNode("demo-cluster") as head:
    head.start(n_local_workers=2)
```

### Deploy an LLM
```python
from pycluster import LLMClusterManager, GPUMonitor

monitor = GPUMonitor()
monitor.start_monitoring()

llm_mgr = LLMClusterManager(head.cluster_manager)
deployment_id = llm_mgr.deploy_model(
    model_name="microsoft/DialoGPT-small",
    model_size="1b",
    precision="fp16"
)
```

### Inference Example
```python
response = llm_mgr.inference(
    deployment_id=deployment_id,
    prompt="Hello!",
    max_tokens=50
)
print(response.text)
```

---

## 📊 Dashboard Features
- Worker and head node resource tracking
- GPU usage visualization
- LLM model deployment status
- Task queue overview

---

## 🥪 Test & Develop
- Unit tests: `tests/test_pycluster.py`
- Feature tests: `tests/test_enhanced_features.py`
- Run with:
```bash
pytest tests/
```

---

## 💥 Dependencies

### Core
- `dask[complete]`, `distributed`
- `torch`, `transformers`, `psutil`, `requests`
- `flask`, `flask-cors`, `pynvml`

### Optional
- `dask-cuda`, `cupy`, `cudf` (for full GPU optimization)

### Dev
- `pytest`, `flake8`, `black`, `mypy`

---

## 🔭 Roadmap & Future Enhancements

### Infrastructure
- Kubernetes and Docker deployment support
- Cloud connectors (AWS, GCP, Azure)

### LLM & GPU Enhancements
- Multi-GPU model sharding
- GPU pinning & scheduling
- Memory-based eviction/reloading

### Monitoring & Security
- Prometheus + Grafana integration
- Role-based auth for API
- Secure communication between nodes

---

## ✅ Summary

PyCluster is a full-stack, Windows-friendly framework for distributed computing and LLM deployment. It combines:

- 📡 Head/worker node architecture
- ⚙️ Dask-based task execution
- 🧠 GPU-aware LLM deployment
- 🌐 Web dashboard + API control
- 🔍 Zero-config LAN worker discovery

> Perfect for local LLM farms, AI classrooms, or GPU-powered research setups — all without touching a container or VM.
````
