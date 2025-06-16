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
    from pycluster import GPUMonitor, LLMResourceManager, LLMClusterManager
except ImportError as e:
    print(f"Warning: Could not import pycluster modules: {e}")
    # Create mock classes for development
    class GPUMonitor:
        def get_gpu_summary(self): return {"available": False, "count": 0, "gpus": []}
        def get_recent_metrics(self, count=10): return []
        def start_monitoring(self, interval=1.0): pass
    
    class LLMResourceManager:
        def __init__(self, gpu_monitor): pass
        def get_resource_status(self): return {"status": "mock"}
        def plan_llm_deployment(self, **kwargs): return {"status": "mock"}
    
    class LLMClusterManager:
        def __init__(self, cluster_manager): pass
        def get_deployed_models(self): return []
        def deploy_model(self, **kwargs): return "mock-deployment-id"
        def get_deployment_status(self, deployment_id): return {"status": "mock"}

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
    global gpu_monitor, resource_manager, llm_manager
    
    try:
        gpu_monitor = GPUMonitor()
        gpu_monitor.start_monitoring(interval=2.0)
        
        resource_manager = LLMResourceManager(gpu_monitor)
        
        if cluster_manager:
            llm_manager = LLMClusterManager(cluster_manager)
        
        logger.info("LLM managers initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LLM managers: {e}")
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
        if not all([resource_manager, llm_manager]):
            return jsonify({"error": "LLM managers not initialized"}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ['model_name', 'model_size']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Plan deployment
        deployment_plan = resource_manager.plan_llm_deployment(
            model_name=data['model_name'],
            model_size=data['model_size'],
            precision=data.get('precision', 'fp16'),
            sharding_strategy=data.get('sharding_strategy', 'auto')
        )
        
        # Allocate resources
        deployment_id = f"deployment-{int(time.time())}"
        allocation_success = resource_manager.allocate_resources(deployment_id, deployment_plan)
        
        if not allocation_success:
            return jsonify({"error": "Failed to allocate resources"}), 409
        
        # Deploy model
        model_deployment_id = llm_manager.deploy_model(
            model_name=data['model_name'],
            model_size=data['model_size'],
            precision=data.get('precision', 'fp16'),
            replicas=data.get('replicas', 1),
            gpu_per_replica=data.get('gpu_per_replica', 1),
            sharding_strategy=deployment_plan.get('sharding_strategy', 'none'),
            max_memory_per_gpu=data.get('max_memory_per_gpu', '12GB')
        )
        
        return jsonify({
            "deployment_id": model_deployment_id,
            "resource_allocation_id": deployment_id,
            "deployment_plan": deployment_plan,
            "status": "deploying"
        })
        
    except Exception as e:
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
        
        # Perform inference
        response = llm_manager.inference(
            deployment_id=deployment_id,
            prompt=data['prompt'],
            max_tokens=data.get('max_tokens', 100),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            stop_sequences=data.get('stop_sequences', [])
        )
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@llm_bp.route('/models/<deployment_id>', methods=['DELETE'])
@cross_origin()
def undeploy_model(deployment_id):
    """Undeploy a model"""
    try:
        if not all([resource_manager, llm_manager]):
            return jsonify({"error": "LLM managers not initialized"}), 503
        
        # Undeploy model
        success = llm_manager.undeploy_model(deployment_id)
        
        if success:
            # Also deallocate resources if we have the resource allocation ID
            # In a real implementation, we'd track this mapping
            return jsonify({"status": "undeployed", "deployment_id": deployment_id})
        else:
            return jsonify({"error": "Failed to undeploy model"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@llm_bp.route('/resources/status', methods=['GET'])
@cross_origin()
def resource_status():
    """Get current resource allocation status"""
    try:
        if not resource_manager:
            return jsonify({"error": "Resource manager not initialized"}), 503
        
        status = resource_manager.get_resource_status()
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

