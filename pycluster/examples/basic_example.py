"""
Example: Basic cluster setup and usage
"""

import time
import logging
from pycluster import HeadNode, WorkerNode, DashboardManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_computation(x):
    """Simple computation function for testing."""
    import time
    time.sleep(0.1)  # Simulate some work
    return x * x


def main():
    """
    Demonstrate basic cluster setup and usage.
    """
    print("PyCluster Basic Example")
    print("=" * 50)
    
    # Start head node
    print("\n1. Starting head node...")
    with HeadNode(cluster_name="example-cluster") as head:
        result = head.start(n_local_workers=2)
        
        if result["status"] == "success":
            print(f"✓ Head node started successfully")
            print(f"  Scheduler: {result['scheduler_address']}")
            print(f"  Dashboard: {result['dashboard_address']}")
            
            # Get connection info
            conn_info = head.get_connection_info()
            print(f"  Connection info: {conn_info}")
            
            # Initialize dashboard manager
            dashboard = DashboardManager(conn_info["dashboard_url"])
            
            # Wait for dashboard to be ready
            print("\n2. Waiting for dashboard...")
            if dashboard.wait_for_dashboard(timeout=10):
                print("✓ Dashboard is accessible")
                
                # Open dashboard in browser (optional)
                # dashboard.open_in_browser()
                
                # Get cluster status
                print("\n3. Cluster status:")
                status = head.get_cluster_status()
                print(f"  Total workers: {status.get('total_workers', 0)}")
                print(f"  Cluster name: {status.get('cluster_name')}")
                
                # Submit some tasks
                print("\n4. Submitting tasks...")
                cluster_manager = head.cluster_manager
                
                # Submit individual tasks
                futures = []
                for i in range(10):
                    future = cluster_manager.submit_task(example_computation, i)
                    futures.append(future)
                
                # Wait for results
                results = [future.result() for future in futures]
                print(f"✓ Task results: {results}")
                
                # Map tasks
                print("\n5. Mapping tasks...")
                data = list(range(20))
                map_futures = cluster_manager.map_tasks(example_computation, data)
                map_results = [future.result() for future in map_futures]
                print(f"✓ Map results: {map_results[:5]}... (showing first 5)")
                
                # Generate dashboard report
                print("\n6. Dashboard report:")
                report = dashboard.generate_dashboard_report()
                print(f"  Dashboard accessible: {report['accessible']}")
                if 'cluster_status' in report:
                    print(f"  Scheduler ID: {report['cluster_status'].get('id', 'N/A')}")
                
                print("\n7. Keeping cluster running for 10 seconds...")
                print("   You can access the dashboard at:", conn_info["dashboard_url"])
                time.sleep(10)
                
            else:
                print("✗ Dashboard not accessible")
        else:
            print(f"✗ Failed to start head node: {result.get('message')}")
    
    print("\n✓ Example completed - cluster shutdown")


if __name__ == "__main__":
    main()

