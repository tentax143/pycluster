"""
Enhanced Flask API routes for LLM management
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin

# Add the pycluster package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pycluster'))

try:
    from pycluster import GPUMonitor, LLMClusterManager
    from pycluster.llm_serving import LLMClusterManager as LLMClusterManagerCore
    PYCLUSTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import pycluster modules: {e}")
    PYCLUSTER_AVAILABLE = False
    # Create fallback classes for when PyCluster is not available
    class GPUMonitor:
        def get_gpu_summary(self): return {"available": False, "count": 0, "gpus": []}
        def get_recent_metrics(self, count=10): return []
        def start_monitoring(self, interval=1.0): pass
    
    class LLMClusterManager:
        def __init__(self, cluster_manager): 
            self.cluster_manager = cluster_manager
            self.models = {}
        def get_deployed_models(self): return []
        def deploy_model(self, **kwargs): return "mock-deployment-id"
        def get_deployment_status(self, deployment_id): return {"status": "mock"}
        def inference(self, **kwargs): return {"text": "Mock response - PyCluster not available"}
        def undeploy_model(self, deployment_id): return True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
llm_bp = Blueprint('llm', __name__, url_prefix='/api/llm')

# Global variables for managers (will be initialized when cluster starts)
gpu_monitor = None
resource_manager = None
llm_manager = None

def initialize_llm_managers(cluster_manager=None):
    """Initialize LLM management components"""
    global gpu_monitor, llm_manager
    
    try:
        if not PYCLUSTER_AVAILABLE:
            logger.warning("PyCluster not available, using fallback implementations")
            gpu_monitor = GPUMonitor()
            if cluster_manager:
                llm_manager = LLMClusterManager(cluster_manager)
            return True
        
        # Initialize real GPU monitor
        gpu_monitor = GPUMonitor()
        gpu_monitor.start_monitoring(interval=2.0)
        
        # Initialize real LLM cluster manager
        if cluster_manager:
            llm_manager = LLMClusterManagerCore(cluster_manager)
            logger.info("LLM cluster manager initialized with real PyCluster")
        else:
            logger.warning("No cluster manager provided, LLM functionality will be limited")
        
        logger.info("LLM managers initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LLM managers: {e}")
        # Fallback to mock implementations
        gpu_monitor = GPUMonitor()
        if cluster_manager:
            llm_manager = LLMClusterManager(cluster_manager)
        return False

@llm_bp.route('/health', methods=['GET'])
@cross_origin()
def llm_health():
    """Check LLM service health"""
    try:
        gpu_status = gpu_monitor.get_gpu_summary() if gpu_monitor else {"available": False}
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "gpu_available": gpu_status.get("available", False),
            "gpu_count": gpu_status.get("count", 0),
            "managers_initialized": all([gpu_monitor, resource_manager, llm_manager])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@llm_bp.route('/gpu/status', methods=['GET'])
@cross_origin()
def gpu_status():
    """Get detailed GPU status"""
    try:
        if not gpu_monitor:
            return jsonify({"error": "GPU monitor not initialized"}), 503
        
        gpu_summary = gpu_monitor.get_gpu_summary()
        recent_metrics = gpu_monitor.get_recent_metrics(count=10)
        
        return jsonify({
            "summary": gpu_summary,
            "recent_metrics": [
                {
                    "timestamp": metric.timestamp.isoformat(),
                    "cpu_usage": metric.cpu_usage,
                    "memory_usage": metric.memory_usage,
                    "gpu_metrics": [
                        {
                            "gpu_id": gpu.gpu_id,
                            "utilization": gpu.utilization,
                            "memory_used": gpu.memory_used,
                            "memory_total": gpu.memory_total,
                            "temperature": gpu.temperature,
                            "power_usage": gpu.power_usage
                        } for gpu in metric.gpu_metrics
                    ]
                } for metric in recent_metrics
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@llm_bp.route('/models', methods=['GET'])
@cross_origin()
def list_models():
    """List deployed LLM models"""
    try:
        if not llm_manager:
            return jsonify({"error": "LLM manager not initialized"}), 503
        
        deployed_models = llm_manager.get_deployed_models()
        
        return jsonify({
            "models": deployed_models,
            "count": len(deployed_models)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@llm_bp.route('/models/deploy', methods=['POST'])
@cross_origin()
def deploy_model():
    """Deploy a new LLM model"""
    try:
        if not llm_manager:
            return jsonify({"error": "LLM manager not initialized"}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ['model_name', 'model_size']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Deploy model using real LLM cluster manager
        model_deployment_id = llm_manager.deploy_model(
            model_name=data['model_name'],
            model_size=data['model_size'],
            precision=data.get('precision', 'fp16'),
            replicas=data.get('replicas', 1),
            gpu_per_replica=data.get('gpu_per_replica', 1)
        )
        
        return jsonify({
            "deployment_id": model_deployment_id,
            "model_name": data['model_name'],
            "model_size": data['model_size'],
            "precision": data.get('precision', 'fp16'),
            "replicas": data.get('replicas', 1),
            "gpu_per_replica": data.get('gpu_per_replica', 1),
            "status": "deploying"
        })
        
    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        return jsonify({"error": str(e)}), 500

@llm_bp.route('/models/<deployment_id>/status', methods=['GET'])
@cross_origin()
def model_status(deployment_id):
    """Get status of a deployed model"""
    try:
        if not llm_manager:
            return jsonify({"error": "LLM manager not initialized"}), 503
        
        status = llm_manager.get_deployment_status(deployment_id)
        return jsonify(status)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@llm_bp.route('/models/<deployment_id>/inference', methods=['POST'])
@cross_origin()
def model_inference(deployment_id):
    """Perform inference with a deployed model"""
    try:
        if not llm_manager:
            return jsonify({"error": "LLM manager not initialized"}), 503
        
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
        
        # Perform inference using real LLM cluster manager
        response = llm_manager.inference(
            deployment_id=deployment_id,
            prompt=data['prompt'],
            max_tokens=data.get('max_tokens', 100),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            stop_sequences=data.get('stop_sequences', [])
        )
        
        # Convert response to dict if it's an LLMResponse object
        if hasattr(response, '__dict__'):
            response_dict = {
                "request_id": response.request_id,
                "text": response.text,
                "tokens_generated": response.tokens_generated,
                "total_tokens": response.total_tokens,
                "finish_reason": response.finish_reason,
                "generation_time": response.generation_time,
                "metadata": response.metadata or {}
            }
        else:
            response_dict = response
        
        return jsonify(response_dict)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return jsonify({"error": str(e)}), 500

@llm_bp.route('/models/<deployment_id>', methods=['DELETE'])
@cross_origin()
def undeploy_model(deployment_id):
    """Undeploy a model"""
    try:
        if not llm_manager:
            return jsonify({"error": "LLM manager not initialized"}), 503
        
        # Undeploy model using real LLM cluster manager
        success = llm_manager.undeploy_model(deployment_id)
        
        if success:
            return jsonify({"status": "undeployed", "deployment_id": deployment_id})
        else:
            return jsonify({"error": "Failed to undeploy model"}), 500
        
    except Exception as e:
        logger.error(f"Model undeployment failed: {e}")
        return jsonify({"error": str(e)}), 500

@llm_bp.route('/resources/status', methods=['GET'])
@cross_origin()
def resource_status():
    """Get current resource allocation status"""
    try:
        if not gpu_monitor:
            return jsonify({"error": "GPU monitor not initialized"}), 503
        
        # Get GPU status as resource status
        gpu_summary = gpu_monitor.get_gpu_summary()
        status = {
            "gpu_available": gpu_summary.get("available", False),
            "gpu_count": gpu_summary.get("count", 0),
            "gpus": gpu_summary.get("gpus", []),
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(status)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@llm_bp.route('/models/available', methods=['GET'])
@cross_origin()
def available_models():
    """Get list of available models for deployment"""
    # This would typically come from a model registry or configuration
    available_models = [
        {
            "name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            "display_name": "DeepSeek Coder 7B",
            "size": "7b",
            "type": "code",
            "memory_requirement": "14GB",
            "description": "Specialized for code generation and programming tasks"
        },
        {
            "name": "microsoft/DialoGPT-small",
            "display_name": "DialoGPT Small",
            "size": "117m",
            "type": "chat",
            "memory_requirement": "1GB",
            "description": "Small conversational model for testing"
        },
        {
            "name": "microsoft/DialoGPT-medium",
            "display_name": "DialoGPT Medium",
            "size": "345m",
            "type": "chat",
            "memory_requirement": "2GB",
            "description": "Medium conversational model"
        },
        {
            "name": "codellama/CodeLlama-7b-Instruct-hf",
            "display_name": "Code Llama 7B",
            "size": "7b",
            "type": "code",
            "memory_requirement": "14GB",
            "description": "Meta's code generation model"
        }
    ]
    
    return jsonify({
        "models": available_models,
        "count": len(available_models)
    })

# Error handlers
@llm_bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@llm_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

