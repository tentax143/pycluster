"""
Comprehensive test suite for PyCluster enhanced features
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pycluster import (
    GPUMonitor, LLMResourceManager, LLMClusterManager, 
    LLMRequest, LLMResponse, HeadNode, WorkerNode
)

class TestGPUMonitor:
    """Test GPU monitoring functionality"""
    
    def test_gpu_monitor_initialization(self):
        """Test GPU monitor can be initialized"""
        monitor = GPUMonitor()
        assert monitor is not None
        assert hasattr(monitor, 'get_gpu_summary')
        assert hasattr(monitor, 'start_monitoring')
    
    @patch('pynvml.nvmlInit')
    @patch('pynvml.nvmlDeviceGetCount')
    def test_gpu_summary_with_mock_gpus(self, mock_count, mock_init):
        """Test GPU summary with mocked NVIDIA GPUs"""
        mock_count.return_value = 2
        
        monitor = GPUMonitor()
        summary = monitor.get_gpu_summary()
        
        assert isinstance(summary, dict)
        assert 'available' in summary
        assert 'count' in summary
        assert 'gpus' in summary
    
    def test_gpu_monitoring_thread(self):
        """Test GPU monitoring thread functionality"""
        monitor = GPUMonitor()
        monitor.start_monitoring(interval=0.1)
        
        # Let it run for a short time
        time.sleep(0.3)
        
        metrics = monitor.get_recent_metrics(count=2)
        assert isinstance(metrics, list)
        
        monitor.stop_monitoring()
    
    def test_gpu_metrics_structure(self):
        """Test GPU metrics data structure"""
        monitor = GPUMonitor()
        summary = monitor.get_gpu_summary()
        
        # Should have proper structure even without GPUs
        assert 'available' in summary
        assert 'count' in summary
        assert 'gpus' in summary
        assert isinstance(summary['gpus'], list)

class TestLLMResourceManager:
    """Test LLM resource management"""
    
    def setup_method(self):
        """Setup for each test"""
        self.gpu_monitor = Mock()
        self.gpu_monitor.get_gpu_summary.return_value = {
            'available': True,
            'count': 2,
            'gpus': [
                {'id': 0, 'memory_total': 24.0, 'memory_used': 2.0},
                {'id': 1, 'memory_total': 24.0, 'memory_used': 3.0}
            ]
        }
        self.resource_manager = LLMResourceManager(self.gpu_monitor)
    
    def test_resource_manager_initialization(self):
        """Test resource manager initialization"""
        assert self.resource_manager is not None
        assert hasattr(self.resource_manager, 'plan_llm_deployment')
        assert hasattr(self.resource_manager, 'allocate_resources')
    
    def test_llm_deployment_planning(self):
        """Test LLM deployment planning"""
        plan = self.resource_manager.plan_llm_deployment(
            model_name="test-model",
            model_size="7b",
            precision="fp16"
        )
        
        assert isinstance(plan, dict)
        assert 'model_name' in plan
        assert 'estimated_memory' in plan
        assert 'recommended_gpus' in plan
    
    def test_resource_allocation(self):
        """Test resource allocation"""
        plan = {
            'model_name': 'test-model',
            'estimated_memory': 14.0,
            'recommended_gpus': 1
        }
        
        success = self.resource_manager.allocate_resources("test-deployment", plan)
        assert isinstance(success, bool)
    
    def test_resource_status(self):
        """Test resource status retrieval"""
        status = self.resource_manager.get_resource_status()
        assert isinstance(status, dict)

class TestLLMClusterManager:
    """Test LLM cluster management"""
    
    def setup_method(self):
        """Setup for each test"""
        self.cluster_manager = Mock()
        self.llm_manager = LLMClusterManager(self.cluster_manager)
    
    def test_llm_manager_initialization(self):
        """Test LLM manager initialization"""
        assert self.llm_manager is not None
        assert hasattr(self.llm_manager, 'deploy_model')
        assert hasattr(self.llm_manager, 'inference')
    
    def test_model_deployment(self):
        """Test model deployment"""
        deployment_id = self.llm_manager.deploy_model(
            model_name="test-model",
            model_size="7b",
            precision="fp16",
            replicas=1,
            gpu_per_replica=1
        )
        
        assert isinstance(deployment_id, str)
        assert len(deployment_id) > 0
    
    def test_deployment_status(self):
        """Test deployment status checking"""
        status = self.llm_manager.get_deployment_status("test-deployment")
        assert isinstance(status, dict)
        assert 'status' in status
    
    def test_inference_request(self):
        """Test inference request"""
        response = self.llm_manager.inference(
            deployment_id="test-deployment",
            prompt="Hello, world!",
            max_tokens=50
        )
        
        assert isinstance(response, dict)
        assert 'text' in response or 'error' in response

class TestLLMRequestResponse:
    """Test LLM request/response objects"""
    
    def test_llm_request_creation(self):
        """Test LLM request object creation"""
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.7
        )
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
    
    def test_llm_response_creation(self):
        """Test LLM response object creation"""
        response = LLMResponse(
            text="Generated text",
            tokens_generated=50,
            generation_time=1.5
        )
        
        assert response.text == "Generated text"
        assert response.tokens_generated == 50
        assert response.generation_time == 1.5

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_cluster_with_gpu_monitoring(self):
        """Test cluster startup with GPU monitoring"""
        try:
            # This test may fail in environments without GPUs
            gpu_monitor = GPUMonitor()
            gpu_monitor.start_monitoring(interval=1.0)
            
            # Let it collect some data
            await asyncio.sleep(2)
            
            metrics = gpu_monitor.get_recent_metrics(count=1)
            assert isinstance(metrics, list)
            
            gpu_monitor.stop_monitoring()
        except Exception as e:
            # Expected in environments without NVIDIA GPUs
            assert "NVML" in str(e) or "CUDA" in str(e)
    
    def test_end_to_end_mock_deployment(self):
        """Test end-to-end mock LLM deployment"""
        # Mock GPU monitor
        gpu_monitor = Mock()
        gpu_monitor.get_gpu_summary.return_value = {
            'available': True,
            'count': 1,
            'gpus': [{'id': 0, 'memory_total': 24.0, 'memory_used': 2.0}]
        }
        
        # Mock cluster manager
        cluster_manager = Mock()
        
        # Create managers
        resource_manager = LLMResourceManager(gpu_monitor)
        llm_manager = LLMClusterManager(cluster_manager)
        
        # Plan deployment
        plan = resource_manager.plan_llm_deployment(
            model_name="test-model",
            model_size="7b",
            precision="fp16"
        )
        
        # Allocate resources
        allocation_success = resource_manager.allocate_resources("test-deployment", plan)
        assert allocation_success
        
        # Deploy model
        deployment_id = llm_manager.deploy_model(
            model_name="test-model",
            model_size="7b",
            precision="fp16",
            replicas=1,
            gpu_per_replica=1
        )
        
        assert isinstance(deployment_id, str)
        
        # Check status
        status = llm_manager.get_deployment_status(deployment_id)
        assert isinstance(status, dict)
        
        # Perform inference
        response = llm_manager.inference(
            deployment_id=deployment_id,
            prompt="Test prompt",
            max_tokens=50
        )
        
        assert isinstance(response, dict)

class TestPerformance:
    """Performance and benchmark tests"""
    
    def test_gpu_monitoring_performance(self):
        """Test GPU monitoring performance"""
        monitor = GPUMonitor()
        
        start_time = time.time()
        for _ in range(100):
            summary = monitor.get_gpu_summary()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.1  # Should be fast even without GPUs
    
    def test_resource_planning_performance(self):
        """Test resource planning performance"""
        gpu_monitor = Mock()
        gpu_monitor.get_gpu_summary.return_value = {
            'available': True,
            'count': 4,
            'gpus': [
                {'id': i, 'memory_total': 24.0, 'memory_used': 2.0}
                for i in range(4)
            ]
        }
        
        resource_manager = LLMResourceManager(gpu_monitor)
        
        start_time = time.time()
        for _ in range(50):
            plan = resource_manager.plan_llm_deployment(
                model_name="test-model",
                model_size="7b",
                precision="fp16"
            )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 50
        assert avg_time < 0.05  # Should be very fast
    
    def test_concurrent_monitoring(self):
        """Test concurrent GPU monitoring"""
        monitor = GPUMonitor()
        results = []
        
        def monitor_worker():
            for _ in range(10):
                summary = monitor.get_gpu_summary()
                results.append(summary)
                time.sleep(0.01)
        
        # Start multiple monitoring threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=monitor_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        assert len(results) == 30  # 3 threads * 10 calls each
        
        # All results should have the same structure
        for result in results:
            assert 'available' in result
            assert 'count' in result
            assert 'gpus' in result

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_gpu_monitor_without_nvidia(self):
        """Test GPU monitor behavior without NVIDIA drivers"""
        monitor = GPUMonitor()
        summary = monitor.get_gpu_summary()
        
        # Should gracefully handle missing NVIDIA drivers
        assert isinstance(summary, dict)
        assert 'available' in summary
    
    def test_resource_manager_insufficient_memory(self):
        """Test resource manager with insufficient GPU memory"""
        gpu_monitor = Mock()
        gpu_monitor.get_gpu_summary.return_value = {
            'available': True,
            'count': 1,
            'gpus': [{'id': 0, 'memory_total': 8.0, 'memory_used': 6.0}]  # Only 2GB free
        }
        
        resource_manager = LLMResourceManager(gpu_monitor)
        
        # Try to plan a large model deployment
        plan = resource_manager.plan_llm_deployment(
            model_name="large-model",
            model_size="70b",  # Very large model
            precision="fp16"
        )
        
        # Should handle insufficient memory gracefully
        assert isinstance(plan, dict)
        assert 'error' in plan or 'warning' in plan or 'estimated_memory' in plan
    
    def test_llm_manager_invalid_deployment(self):
        """Test LLM manager with invalid deployment ID"""
        cluster_manager = Mock()
        llm_manager = LLMClusterManager(cluster_manager)
        
        status = llm_manager.get_deployment_status("invalid-deployment-id")
        assert isinstance(status, dict)
        assert 'error' in status or 'status' in status

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])

