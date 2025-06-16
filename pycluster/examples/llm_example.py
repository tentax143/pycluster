"""
Example: LLM deployment and inference with PyCluster
"""

import time
import logging
from pycluster import HeadNode, GPUMonitor, LLMResourceManager, LLMClusterManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Demonstrate LLM deployment and inference using PyCluster.
    """
    print("=== PyCluster LLM Example ===")
    
    # Initialize GPU monitoring
    print("\n1. Initializing GPU monitoring...")
    gpu_monitor = GPUMonitor()
    gpu_monitor.start_monitoring(interval=2.0)
    
    # Check GPU availability
    gpu_summary = gpu_monitor.get_gpu_summary()
    print(f"GPU Status: {gpu_summary}")
    
    if not gpu_summary['available']:
        print("Warning: No GPUs detected. LLM deployment may be limited to CPU.")
    
    # Initialize resource manager
    resource_manager = LLMResourceManager(gpu_monitor)
    
    # Start head node
    print("\n2. Starting PyCluster head node...")
    with HeadNode(cluster_name="llm-cluster", host="0.0.0.0") as head:
        result = head.start(n_local_workers=2)
        
        if result["status"] != "success":
            print(f"Failed to start cluster: {result}")
            return
        
        print(f"Cluster started successfully!")
        print(f"Dashboard: {result['dashboard_address']}")
        print(f"Scheduler: {result['scheduler_address']}")
        
        # Initialize LLM cluster manager
        llm_manager = LLMClusterManager(head.cluster_manager)
        
        # Example 1: Plan LLM deployment
        print("\n3. Planning LLM deployment...")
        try:
            deployment_plan = resource_manager.plan_llm_deployment(
                model_name="microsoft/DialoGPT-small",  # Small model for testing
                model_size="1b",
                precision="fp16"
            )
            print(f"Deployment plan: {deployment_plan}")
            
            # Allocate resources
            deployment_id = "test-deployment-1"
            allocation_success = resource_manager.allocate_resources(deployment_id, deployment_plan)
            
            if allocation_success:
                print(f"Resources allocated for deployment {deployment_id}")
                
                # Example 2: Deploy model (mock deployment for demonstration)
                print("\n4. Deploying LLM model...")
                try:
                    model_deployment_id = llm_manager.deploy_model(
                        model_name="microsoft/DialoGPT-small",
                        model_size="1b",
                        precision="fp16",
                        replicas=1,
                        gpu_per_replica=1 if gpu_summary['available'] else 0
                    )
                    print(f"Model deployed with ID: {model_deployment_id}")
                    
                    # Wait for deployment to complete
                    time.sleep(5)
                    
                    # Check deployment status
                    status = llm_manager.get_deployment_status(model_deployment_id)
                    print(f"Deployment status: {status}")
                    
                    # Example 3: Perform inference
                    print("\n5. Performing inference...")
                    try:
                        response = llm_manager.inference(
                            deployment_id=model_deployment_id,
                            prompt="Hello, how are you?",
                            max_tokens=50,
                            temperature=0.7
                        )
                        print(f"Inference response: {response}")
                    except Exception as e:
                        print(f"Inference failed (expected in demo): {e}")
                    
                    # Example 4: Monitor resources
                    print("\n6. Monitoring resources...")
                    resource_status = resource_manager.get_resource_status()
                    print(f"Resource status: {resource_status}")
                    
                    # Example 5: Cleanup
                    print("\n7. Cleaning up...")
                    llm_manager.undeploy_model(model_deployment_id)
                    resource_manager.deallocate_resources(deployment_id)
                    print("Cleanup completed")
                    
                except Exception as e:
                    print(f"Model deployment failed (expected in demo): {e}")
                    print("This is normal in a demo environment without actual model files")
            else:
                print("Failed to allocate resources")
                
        except Exception as e:
            print(f"Deployment planning failed: {e}")
            print("This may be due to insufficient GPU memory or missing dependencies")
        
        # Example 6: Show cluster information
        print("\n8. Cluster information:")
        cluster_info = head.cluster_manager.get_cluster_info()
        print(f"Workers: {len(cluster_info.get('workers', []))}")
        print(f"Total cores: {cluster_info.get('total_cores', 0)}")
        
        # Keep cluster running for a bit to demonstrate monitoring
        print("\n9. Monitoring cluster for 30 seconds...")
        for i in range(6):
            time.sleep(5)
            metrics = gpu_monitor.get_recent_metrics(count=1)
            if metrics:
                latest = metrics[-1]
                print(f"  CPU: {latest.cpu_usage:.1f}%, Memory: {latest.memory_usage:.1f}%, GPUs: {len(latest.gpu_metrics)}")
        
        print("\nExample completed successfully!")

if __name__ == "__main__":
    main()

