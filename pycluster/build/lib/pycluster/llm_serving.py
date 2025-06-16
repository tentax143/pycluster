"""
LLM serving and distributed inference capabilities for PyCluster
"""

import os
import sys
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import Future
import queue
import uuid

logger = logging.getLogger(__name__)

# Try to import LLM-related libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. LLM serving will be limited.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. LLM serving will be limited.")


@dataclass
class LLMRequest:
    """Request for LLM inference."""
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = None
    stream: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class LLMResponse:
    """Response from LLM inference."""
    request_id: str
    text: str
    tokens_generated: int
    total_tokens: int
    finish_reason: str
    generation_time: float
    metadata: Dict[str, Any] = None


@dataclass
class LLMModelInfo:
    """Information about a loaded LLM model."""
    model_id: str
    model_name: str
    model_size: str
    precision: str
    gpu_indices: List[int]
    memory_usage: int
    max_sequence_length: int
    loaded_at: datetime
    status: str  # "loading", "ready", "error"


class LLMWorker:
    """
    Worker class for running LLM inference on specific GPUs.
    """
    
    def __init__(self, worker_id: str, gpu_indices: List[int]):
        self.worker_id = worker_id
        self.gpu_indices = gpu_indices
        self.model = None
        self.tokenizer = None
        self.model_info = None
        self.request_queue = queue.Queue()
        self.response_callbacks = {}
        self.running = False
        self.worker_thread = None
        self.stats = {
            "requests_processed": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0.0,
            "average_tokens_per_second": 0.0
        }
    
    def load_model(self, model_name: str, model_size: str = "7b", precision: str = "fp16") -> bool:
        """
        Load an LLM model onto the assigned GPUs.
        
        Args:
            model_name: Name/path of the model
            model_size: Size of the model
            precision: Model precision
        
        Returns:
            True if model loaded successfully
        """
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            logger.error("PyTorch and Transformers are required for LLM serving")
            return False
        
        try:
            logger.info(f"Loading model {model_name} on GPUs {self.gpu_indices}")
            
            # Set CUDA devices
            if self.gpu_indices:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_indices))
                device = f"cuda:{self.gpu_indices[0]}" if len(self.gpu_indices) == 1 else "cuda"
            else:
                device = "cpu"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model configuration
            config = AutoConfig.from_pretrained(model_name)
            
            # Configure model loading based on precision and GPU count
            model_kwargs = {
                "torch_dtype": torch.float16 if precision == "fp16" else torch.float32,
                "device_map": "auto" if len(self.gpu_indices) > 1 else device,
                "trust_remote_code": True
            }
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            if len(self.gpu_indices) == 1:
                self.model = self.model.to(device)
            
            # Create model info
            self.model_info = LLMModelInfo(
                model_id=str(uuid.uuid4()),
                model_name=model_name,
                model_size=model_size,
                precision=precision,
                gpu_indices=self.gpu_indices,
                memory_usage=self._estimate_memory_usage(),
                max_sequence_length=getattr(config, 'max_position_embeddings', 2048),
                loaded_at=datetime.now(),
                status="ready"
            )
            
            logger.info(f"Model {model_name} loaded successfully on worker {self.worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.model_info = LLMModelInfo(
                model_id=str(uuid.uuid4()),
                model_name=model_name,
                model_size=model_size,
                precision=precision,
                gpu_indices=self.gpu_indices,
                memory_usage=0,
                max_sequence_length=0,
                loaded_at=datetime.now(),
                status="error"
            )
            return False
    
    def _estimate_memory_usage(self) -> int:
        """Estimate current GPU memory usage by the model."""
        if not TORCH_AVAILABLE or not self.gpu_indices:
            return 0
        
        try:
            total_memory = 0
            for gpu_idx in self.gpu_indices:
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(gpu_idx)
                    total_memory += memory_allocated
            return total_memory
        except Exception as e:
            logger.warning(f"Could not estimate memory usage: {e}")
            return 0
    
    def start_worker(self):
        """Start the worker thread for processing requests."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info(f"Started LLM worker {self.worker_id}")
    
    def stop_worker(self):
        """Stop the worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=10)
        logger.info(f"Stopped LLM worker {self.worker_id}")
    
    def _worker_loop(self):
        """Main worker loop for processing inference requests."""
        while self.running:
            try:
                # Get request from queue with timeout
                request, callback = self.request_queue.get(timeout=1.0)
                
                # Process the request
                start_time = time.time()
                response = self._process_request(request)
                end_time = time.time()
                
                response.generation_time = end_time - start_time
                
                # Update statistics
                self.stats["requests_processed"] += 1
                self.stats["total_tokens_generated"] += response.tokens_generated
                self.stats["total_inference_time"] += response.generation_time
                
                if self.stats["total_inference_time"] > 0:
                    self.stats["average_tokens_per_second"] = (
                        self.stats["total_tokens_generated"] / self.stats["total_inference_time"]
                    )
                
                # Send response via callback
                if callback:
                    callback(response)
                
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
    
    def _process_request(self, request: LLMRequest) -> LLMResponse:
        """Process a single inference request."""
        if not self.model or not self.tokenizer:
            return LLMResponse(
                request_id=request.request_id,
                text="",
                tokens_generated=0,
                total_tokens=0,
                finish_reason="error",
                generation_time=0.0,
                metadata={"error": "Model not loaded"}
            )
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_info.max_sequence_length - request.max_tokens
            )
            
            # Move to appropriate device
            if self.gpu_indices and torch.cuda.is_available():
                device = f"cuda:{self.gpu_indices[0]}"
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            input_length = inputs["input_ids"].shape[1]
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stop_strings=request.stop_sequences or []
                )
            
            # Decode response
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Determine finish reason
            finish_reason = "length" if len(generated_tokens) >= request.max_tokens else "stop"
            
            return LLMResponse(
                request_id=request.request_id,
                text=generated_text,
                tokens_generated=len(generated_tokens),
                total_tokens=len(outputs[0]),
                finish_reason=finish_reason,
                generation_time=0.0,  # Will be set by caller
                metadata={"input_tokens": input_length}
            )
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            return LLMResponse(
                request_id=request.request_id,
                text="",
                tokens_generated=0,
                total_tokens=0,
                finish_reason="error",
                generation_time=0.0,
                metadata={"error": str(e)}
            )
    
    def submit_request(self, request: LLMRequest, callback: Callable[[LLMResponse], None] = None):
        """Submit an inference request to the worker."""
        self.request_queue.put((request, callback))
    
    def get_status(self) -> Dict[str, Any]:
        """Get worker status and statistics."""
        return {
            "worker_id": self.worker_id,
            "gpu_indices": self.gpu_indices,
            "model_info": asdict(self.model_info) if self.model_info else None,
            "running": self.running,
            "queue_size": self.request_queue.qsize(),
            "stats": self.stats.copy(),
            "memory_usage": self._estimate_memory_usage()
        }


class LLMClusterManager:
    """
    Manage LLM serving across a Dask cluster.
    """
    
    def __init__(self, cluster_manager):
        self.cluster_manager = cluster_manager
        self.workers = {}
        self.models = {}
        self.request_router = LLMRequestRouter()
        self.load_balancer = LLMLoadBalancer()
    
    def deploy_model(self, 
                    model_name: str, 
                    model_size: str = "7b",
                    precision: str = "fp16",
                    replicas: int = 1,
                    gpu_per_replica: int = 1) -> str:
        """
        Deploy an LLM model across the cluster.
        
        Args:
            model_name: Name/path of the model
            model_size: Size of the model
            precision: Model precision
            replicas: Number of model replicas
            gpu_per_replica: GPUs per replica
        
        Returns:
            Deployment ID
        """
        deployment_id = str(uuid.uuid4())
        
        try:
            # Get cluster information
            cluster_info = self.cluster_manager.get_cluster_info()
            workers = cluster_info.get("workers", [])
            
            if len(workers) < replicas:
                raise ValueError(f"Not enough workers. Need {replicas}, have {len(workers)}")
            
            # Deploy to workers using Dask
            deployment_futures = []
            
            for i in range(replicas):
                worker_address = workers[i]["address"]
                
                # Submit model loading task to specific worker
                future = self.cluster_manager.submit_task(
                    self._load_model_on_worker,
                    model_name,
                    model_size,
                    precision,
                    list(range(gpu_per_replica)),  # GPU indices for this replica
                    worker=worker_address
                )
                deployment_futures.append(future)
            
            # Wait for all deployments to complete
            results = [f.result(timeout=300) for f in deployment_futures]  # 5 minute timeout
            
            # Store deployment information
            self.models[deployment_id] = {
                "model_name": model_name,
                "model_size": model_size,
                "precision": precision,
                "replicas": replicas,
                "gpu_per_replica": gpu_per_replica,
                "worker_results": results,
                "deployed_at": datetime.now(),
                "status": "ready"
            }
            
            logger.info(f"Successfully deployed model {model_name} with deployment ID {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}: {e}")
            self.models[deployment_id] = {
                "model_name": model_name,
                "status": "error",
                "error": str(e),
                "deployed_at": datetime.now()
            }
            raise
    
    def _load_model_on_worker(self, model_name: str, model_size: str, precision: str, gpu_indices: List[int]) -> Dict[str, Any]:
        """Load model on a specific worker (executed remotely)."""
        worker_id = f"worker_{os.getpid()}_{threading.get_ident()}"
        
        # Create and start LLM worker
        llm_worker = LLMWorker(worker_id, gpu_indices)
        success = llm_worker.load_model(model_name, model_size, precision)
        
        if success:
            llm_worker.start_worker()
            
            # Store worker reference (in practice, this would be managed differently)
            return {
                "worker_id": worker_id,
                "status": "ready",
                "model_info": asdict(llm_worker.model_info),
                "gpu_indices": gpu_indices
            }
        else:
            return {
                "worker_id": worker_id,
                "status": "error",
                "error": "Failed to load model"
            }
    
    def inference(self, deployment_id: str, prompt: str, **kwargs) -> LLMResponse:
        """
        Perform inference using a deployed model.
        
        Args:
            deployment_id: ID of the deployed model
            prompt: Input prompt
            **kwargs: Additional inference parameters
        
        Returns:
            LLM response
        """
        if deployment_id not in self.models:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.models[deployment_id]
        if deployment["status"] != "ready":
            raise ValueError(f"Deployment {deployment_id} is not ready")
        
        # Create request
        request = LLMRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 100),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stop_sequences=kwargs.get("stop_sequences"),
            stream=kwargs.get("stream", False)
        )
        
        # Submit inference task to cluster
        future = self.cluster_manager.submit_task(
            self._process_inference_request,
            deployment_id,
            request
        )
        
        # Wait for result
        response_data = future.result(timeout=60)
        return LLMResponse(**response_data)
    
    def _process_inference_request(self, deployment_id: str, request: LLMRequest) -> Dict[str, Any]:
        """Process inference request on worker (executed remotely)."""
        # This would be implemented to route to the appropriate LLM worker
        # For now, return a mock response
        return {
            "request_id": request.request_id,
            "text": f"Mock response for: {request.prompt[:50]}...",
            "tokens_generated": 20,
            "total_tokens": 50,
            "finish_reason": "stop",
            "generation_time": 0.5,
            "metadata": {"deployment_id": deployment_id}
        }
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of a model deployment."""
        if deployment_id not in self.models:
            return {"error": "Deployment not found"}
        
        return self.models[deployment_id].copy()
    
    def list_deployments(self) -> Dict[str, Dict[str, Any]]:
        """List all model deployments."""
        return {dep_id: self.get_deployment_status(dep_id) for dep_id in self.models}
    
    def undeploy_model(self, deployment_id: str) -> bool:
        """Remove a model deployment."""
        if deployment_id not in self.models:
            return False
        
        try:
            # Submit cleanup tasks to workers
            deployment = self.models[deployment_id]
            
            # In a real implementation, this would send cleanup tasks to workers
            # For now, just remove from local registry
            del self.models[deployment_id]
            
            logger.info(f"Successfully undeployed model {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to undeploy model {deployment_id}: {e}")
            return False


class LLMRequestRouter:
    """Route LLM requests to appropriate workers."""
    
    def __init__(self):
        self.routing_strategy = "round_robin"
        self.worker_index = 0
    
    def route_request(self, request: LLMRequest, available_workers: List[str]) -> str:
        """Route a request to an appropriate worker."""
        if not available_workers:
            raise ValueError("No available workers")
        
        if self.routing_strategy == "round_robin":
            worker = available_workers[self.worker_index % len(available_workers)]
            self.worker_index += 1
            return worker
        
        # Add more routing strategies as needed
        return available_workers[0]


class LLMLoadBalancer:
    """Load balance requests across LLM workers."""
    
    def __init__(self):
        self.worker_loads = {}
        self.worker_capacities = {}
    
    def update_worker_load(self, worker_id: str, current_load: int, capacity: int):
        """Update worker load information."""
        self.worker_loads[worker_id] = current_load
        self.worker_capacities[worker_id] = capacity
    
    def get_least_loaded_worker(self, available_workers: List[str]) -> Optional[str]:
        """Get the worker with the least load."""
        if not available_workers:
            return None
        
        min_load = float('inf')
        best_worker = None
        
        for worker_id in available_workers:
            load = self.worker_loads.get(worker_id, 0)
            capacity = self.worker_capacities.get(worker_id, 1)
            load_ratio = load / capacity if capacity > 0 else float('inf')
            
            if load_ratio < min_load:
                min_load = load_ratio
                best_worker = worker_id
        
        return best_worker

