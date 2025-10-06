# PyCluster

A Windows-first distributed computing framework with built-in LLM support, GPU management, and cluster orchestration capabilities.

## Features

- **Distributed Computing**: Head node/worker architecture built on Dask
- **LLM Deployment**: Deploy and serve large language models across multiple GPUs
- **GPU Management**: Real-time monitoring and resource allocation for NVIDIA GPUs
- **Windows Optimized**: Native Windows support with automatic firewall configuration
- **Auto-Discovery**: UDP-based cluster discovery for easy worker joining
- **Web Dashboard**: Real-time monitoring and management interface
- **REST API**: Complete programmatic control via HTTP endpoints

## Quick Start

### Installation

```bash
pip install pycluster[gpu]  # With GPU support
# or
pip install pycluster       # CPU only
```

### Basic Usage

```python
from pycluster import HeadNode, LLMClusterManager

# Start cluster
with HeadNode("my-cluster") as head:
    head.start(n_local_workers=2)
    
    # Deploy LLM model
    llm_manager = LLMClusterManager(head.cluster_manager)
    deployment_id = llm_manager.deploy_model("deepseek-ai/deepseek-coder-7b-instruct-v1.5")
    
    # Perform inference
    response = llm_manager.inference(deployment_id, "Write Python code for sorting")
    print(response.text)
```

### Command Line

```bash
# Start head node
pycluster --cluster-name my-cluster --local-workers 2

# Join worker (on another machine)
pycluster worker --scheduler tcp://192.168.1.100:8786
```

## Architecture

- **Head Node**: Central scheduler managing the cluster
- **Worker Nodes**: Execute tasks and host LLM models
- **GPU Monitoring**: Real-time GPU metrics and resource allocation
- **Web Dashboard**: Monitor cluster status and performance
- **REST API**: Programmatic cluster management

## Supported Models

- DeepSeek Coder (`deepseek-ai/deepseek-coder-7b-instruct-v1.5`)
- DialoGPT (`microsoft/DialoGPT-small`)
- Code Llama (`codellama/CodeLlama-7b-Instruct-hf`)
- Any Hugging Face compatible model

## Requirements

- Python 3.8+
- Windows 10/11 (primary), Linux, macOS
- NVIDIA GPU (optional, for LLM acceleration)
- Network connectivity (for multi-machine clusters)

## Documentation

- [Complete Documentation](CODEBASE_INDEX.md)
- [Examples](examples/)
- [API Reference](pycluster-api/)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
