#!/usr/bin/env python3
"""
Test script for PyCluster API integration
"""

import requests
import json
import time
import sys

def test_api_health():
    """Test API health endpoints"""
    print("🏥 Testing API Health Endpoints...")
    
    base_url = "http://localhost:5000/api"
    
    try:
        # Test cluster health
        print("🔄 Testing cluster health...")
        response = requests.get(f"{base_url}/cluster/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Cluster health: {data.get('status', 'unknown')}")
            print(f"   PyCluster available: {data.get('pycluster_available', False)}")
        else:
            print(f"❌ Cluster health failed: {response.status_code}")
            return False
        
        # Test LLM health
        print("🔄 Testing LLM health...")
        response = requests.get(f"{base_url}/llm/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ LLM health: {data.get('status', 'unknown')}")
            print(f"   GPU available: {data.get('gpu_available', False)}")
            print(f"   GPU count: {data.get('gpu_count', 0)}")
        else:
            print(f"❌ LLM health failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("💡 Make sure the API server is running:")
        print("   cd pycluster-api && python src/main.py")
        return False
    except Exception as e:
        print(f"❌ API health test failed: {e}")
        return False

def test_cluster_operations():
    """Test cluster operations"""
    print("\n🔄 Testing Cluster Operations...")
    
    base_url = "http://localhost:5000/api"
    
    try:
        # Test cluster status
        print("🔄 Getting cluster status...")
        response = requests.get(f"{base_url}/cluster/cluster/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Cluster status: {data.get('status', 'unknown')}")
            if data.get('status') == 'disconnected':
                print("   (No cluster connected - this is expected if no head node is running)")
        else:
            print(f"❌ Cluster status failed: {response.status_code}")
        
        # Test workers endpoint
        print("🔄 Getting workers info...")
        response = requests.get(f"{base_url}/cluster/cluster/workers", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Workers: {data.get('total_workers', 0)} total")
        else:
            print(f"❌ Workers info failed: {response.status_code}")
        
        # Test metrics
        print("🔄 Getting cluster metrics...")
        response = requests.get(f"{base_url}/cluster/cluster/metrics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Metrics retrieved successfully")
        else:
            print(f"❌ Metrics failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cluster operations test failed: {e}")
        return False

def test_llm_operations():
    """Test LLM operations"""
    print("\n🤖 Testing LLM Operations...")
    
    base_url = "http://localhost:5000/api"
    
    try:
        # Test available models
        print("🔄 Getting available models...")
        response = requests.get(f"{base_url}/llm/models/available", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"✅ Available models: {len(models)}")
            for model in models[:3]:  # Show first 3
                print(f"   - {model.get('display_name', 'Unknown')} ({model.get('size', 'Unknown')})")
        else:
            print(f"❌ Available models failed: {response.status_code}")
        
        # Test deployed models
        print("🔄 Getting deployed models...")
        response = requests.get(f"{base_url}/llm/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"✅ Deployed models: {len(models)}")
        else:
            print(f"❌ Deployed models failed: {response.status_code}")
        
        # Test GPU status
        print("🔄 Getting GPU status...")
        response = requests.get(f"{base_url}/llm/gpu/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            print(f"✅ GPU status: {summary.get('count', 0)} GPUs available")
        else:
            print(f"❌ GPU status failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM operations test failed: {e}")
        return False

def test_model_deployment():
    """Test model deployment (if cluster is available)"""
    print("\n🚀 Testing Model Deployment...")
    
    base_url = "http://localhost:5000/api"
    
    try:
        # First check if we have a cluster
        response = requests.get(f"{base_url}/cluster/cluster/status", timeout=5)
        if response.status_code != 200:
            print("❌ Cannot test deployment - cluster status unavailable")
            return False
        
        cluster_data = response.json()
        if cluster_data.get('status') == 'disconnected':
            print("⚠️  No cluster connected - skipping deployment test")
            print("💡 Start a head node first to test model deployment")
            return True
        
        # Try to deploy a small model
        print("🔄 Deploying test model...")
        deployment_data = {
            "model_name": "microsoft/DialoGPT-small",
            "model_size": "117m",
            "precision": "fp16",
            "replicas": 1,
            "gpu_per_replica": 1
        }
        
        response = requests.post(
            f"{base_url}/llm/models/deploy",
            json=deployment_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            deployment_id = data.get('deployment_id')
            print(f"✅ Model deployed successfully!")
            print(f"   Deployment ID: {deployment_id}")
            print(f"   Model: {data.get('model_name')}")
            print(f"   Status: {data.get('status')}")
            
            # Test inference
            print("🔄 Testing inference...")
            inference_data = {
                "prompt": "Hello, how are you?",
                "max_tokens": 20,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{base_url}/llm/models/{deployment_id}/inference",
                json=inference_data,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Inference successful!")
                print(f"   Response: {data.get('text', 'No text')[:100]}...")
                print(f"   Tokens: {data.get('tokens_generated', 0)}")
                print(f"   Time: {data.get('generation_time', 0):.2f}s")
            else:
                print(f"❌ Inference failed: {response.status_code}")
                print(f"   Error: {response.text}")
            
        else:
            print(f"❌ Model deployment failed: {response.status_code}")
            print(f"   Error: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model deployment test failed: {e}")
        return False

if __name__ == "__main__":
    print("PyCluster API Integration Test")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        print("\n❌ API health tests failed - stopping")
        sys.exit(1)
    
    # Test cluster operations
    test_cluster_operations()
    
    # Test LLM operations
    test_llm_operations()
    
    # Test model deployment
    test_model_deployment()
    
    print("\n✅ API integration tests completed!")
    print("\n💡 Next steps:")
    print("   1. Start a head node: python test_head_node.py")
    print("   2. Start workers: python test_worker_join.py")
    print("   3. Test full deployment: python test_api.py")
