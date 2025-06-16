"""
Example: DeepSeek model deployment with PyCluster
"""

import time
import logging
from pycluster import HeadNode, GPUMonitor, LLMResourceManager, LLMClusterManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_deepseek_model():
    """
    Deploy and run DeepSeek Coder model using PyCluster.
    """
    print("=== DeepSeek Model Deployment with PyCluster ===")
    
    # Initialize GPU monitoring
    print("\n1. Initializing GPU monitoring...")
    gpu_monitor = GPUMonitor()
    gpu_monitor.start_monitoring(interval=1.0)
    
    # Check GPU availability and requirements
    gpu_summary = gpu_monitor.get_gpu_summary()
    print(f"GPU Status: {gpu_summary}")
    
    if not gpu_summary['available']:
        print("Error: DeepSeek models require GPU acceleration. No GPUs detected.")
        return False
    
    total_vram = sum(gpu['memory_total'] for gpu in gpu_summary['gpus'])
    print(f"Total VRAM available: {total_vram:.1f}GB")
    
    if total_vram < 14:  # DeepSeek 7B typically needs ~14GB
        print(f"Warning: DeepSeek 7B requires ~14GB VRAM. Available: {total_vram:.1f}GB")
        print("Consider using model sharding or smaller precision.")
    
    # Initialize resource manager
    resource_manager = LLMResourceManager(gpu_monitor)
    
    # Start head node with GPU workers
    print("\n2. Starting PyCluster with GPU workers...")
    with HeadNode(cluster_name="deepseek-cluster", host="0.0.0.0") as head:
        # Start with GPU-enabled workers
        result = head.start(
            n_local_workers=min(2, gpu_summary['count']),  # One worker per GPU pair
            worker_resources={'GPU': 1}  # Each worker gets access to GPUs
        )
        
        if result["status"] != "success":
            print(f"Failed to start cluster: {result}")
            return False
        
        print(f"Cluster started successfully!")
        print(f"Dashboard: {result['dashboard_address']}")
        print(f"Scheduler: {result['scheduler_address']}")
        
        # Initialize LLM cluster manager
        llm_manager = LLMClusterManager(head.cluster_manager)
        
        # Plan DeepSeek deployment
        print("\n3. Planning DeepSeek deployment...")
        try:
            deployment_plan = resource_manager.plan_llm_deployment(
                model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                model_size="7b",
                precision="fp16",  # Use fp16 to reduce memory usage
                sharding_strategy="tensor_parallel" if gpu_summary['count'] > 1 else "none"
            )
            print(f"Deployment plan: {deployment_plan}")
            
            # Allocate resources
            deployment_id = "deepseek-deployment"
            allocation_success = resource_manager.allocate_resources(deployment_id, deployment_plan)
            
            if allocation_success:
                print(f"Resources allocated for deployment {deployment_id}")
                
                # Deploy DeepSeek model
                print("\n4. Deploying DeepSeek Coder 7B...")
                try:
                    model_deployment_id = llm_manager.deploy_model(
                        model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                        model_size="7b",
                        precision="fp16",
                        replicas=1,
                        gpu_per_replica=min(2, gpu_summary['count']),  # Use up to 2 GPUs
                        sharding_strategy=deployment_plan.get('sharding_strategy', 'none'),
                        max_memory_per_gpu="12GB"  # Limit memory usage
                    )
                    print(f"DeepSeek model deployed with ID: {model_deployment_id}")
                    
                    # Wait for deployment to complete
                    print("Waiting for model to load...")
                    for i in range(12):  # Wait up to 60 seconds
                        time.sleep(5)
                        status = llm_manager.get_deployment_status(model_deployment_id)
                        print(f"  Status: {status['status']} ({i*5}s)")
                        if status['status'] == 'running':
                            break
                    
                    if status['status'] == 'running':
                        print("✅ DeepSeek model is ready for inference!")
                        
                        # Perform coding inference
                        print("\n5. Testing DeepSeek coding capabilities...")
                        coding_prompts = [
                            "Write a Python function to calculate fibonacci numbers:",
                            "Create a binary search algorithm in Python:",
                            "Implement a simple REST API using Flask:"
                        ]
                        
                        for i, prompt in enumerate(coding_prompts, 1):
                            print(f"\n--- Test {i}: {prompt} ---")
                            try:
                                response = llm_manager.inference(
                                    deployment_id=model_deployment_id,
                                    prompt=prompt,
                                    max_tokens=200,
                                    temperature=0.2,  # Lower temperature for code generation
                                    stop_sequences=["```", "\n\n\n"]
                                )
                                print(f"Response: {response['text']}")
                                print(f"Tokens generated: {response.get('tokens_generated', 'N/A')}")
                                print(f"Generation time: {response.get('generation_time', 'N/A')}s")
                            except Exception as e:
                                print(f"Inference failed: {e}")
                        
                        # Monitor performance
                        print("\n6. Performance monitoring...")
                        for i in range(6):
                            time.sleep(5)
                            metrics = gpu_monitor.get_recent_metrics(count=1)
                            if metrics:
                                latest = metrics[-1]
                                gpu_info = []
                                for gpu_id, gpu_metric in enumerate(latest.gpu_metrics):
                                    gpu_info.append(f"GPU{gpu_id}: {gpu_metric.utilization:.1f}% util, "
                                                  f"{gpu_metric.memory_used:.1f}GB/{gpu_metric.memory_total:.1f}GB")
                                print(f"  {' | '.join(gpu_info)}")
                        
                        # Benchmark inference speed
                        print("\n7. Benchmarking inference speed...")
                        start_time = time.time()
                        benchmark_prompt = "def quicksort(arr):"
                        
                        for i in range(5):
                            response = llm_manager.inference(
                                deployment_id=model_deployment_id,
                                prompt=benchmark_prompt,
                                max_tokens=100,
                                temperature=0.1
                            )
                        
                        total_time = time.time() - start_time
                        avg_time = total_time / 5
                        print(f"Average inference time: {avg_time:.2f}s per request")
                        
                    else:
                        print(f"❌ Model failed to start. Final status: {status}")
                    
                    # Cleanup
                    print("\n8. Cleaning up...")
                    llm_manager.undeploy_model(model_deployment_id)
                    resource_manager.deallocate_resources(deployment_id)
                    print("✅ Cleanup completed")
                    
                except Exception as e:
                    print(f"❌ Model deployment failed: {e}")
                    print("This may be due to:")
                    print("- Insufficient GPU memory")
                    print("- Missing model files or Hugging Face access")
                    print("- CUDA/PyTorch compatibility issues")
                    return False
            else:
                print("❌ Failed to allocate resources")
                return False
                
        except Exception as e:
            print(f"❌ Deployment planning failed: {e}")
            return False
        
        print("\n✅ DeepSeek deployment example completed!")
        return True

if __name__ == "__main__":
    success = deploy_deepseek_model()
    if success:
        print("\n🎉 DeepSeek model successfully deployed and tested!")
    else:
        print("\n⚠️  DeepSeek deployment encountered issues. Check logs above.")

