"""
Example: Multi-machine cluster setup
"""

import sys
import logging
from pycluster import HeadNode, WorkerNode, DashboardManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_head_node():
    """Start a head node."""
    print("Starting head node...")
    
    head = HeadNode(
        cluster_name="multi-machine-cluster",
        host="0.0.0.0",  # Listen on all interfaces
        scheduler_port=8786,
        dashboard_port=8787
    )
    
    result = head.start(n_local_workers=1)
    
    if result["status"] == "success":
        print(f"✓ Head node started successfully")
        print(f"  Scheduler: {result['scheduler_address']}")
        print(f"  Dashboard: {result['dashboard_address']}")
        
        # Get connection info for workers
        conn_info = head.get_connection_info()
        print(f"\nFor workers to connect, use:")
        print(f"  python multi_machine_example.py worker {conn_info['scheduler_address']}")
        print(f"\nDashboard available at: {conn_info['dashboard_url']}")
        
        # Keep running
        try:
            print("\nHead node running. Press Ctrl+C to stop...")
            while True:
                status = head.get_cluster_status()
                print(f"Workers connected: {status.get('total_workers', 0)}")
                import time
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nShutting down head node...")
            head.shutdown()
    else:
        print(f"✗ Failed to start head node: {result.get('message')}")


def start_worker(scheduler_address):
    """Start a worker node."""
    print(f"Starting worker node, connecting to {scheduler_address}...")
    
    worker = WorkerNode(
        scheduler_address=scheduler_address,
        worker_name=f"worker-{socket.gethostname()}"
    )
    
    result = worker.start(n_workers=2, threads_per_worker=2)
    
    if result["status"] == "success":
        print(f"✓ Worker node started successfully")
        print(f"  Workers added: {result['workers_added']}")
        
        # Keep running
        try:
            print("\nWorker node running. Press Ctrl+C to stop...")
            while True:
                status = worker.get_status()
                print(f"Worker status: {status.get('status')}, Workers: {status.get('worker_count', 0)}")
                import time
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nShutting down worker node...")
            worker.shutdown()
    else:
        print(f"✗ Failed to start worker node: {result.get('message')}")


def main():
    """
    Main function to handle command line arguments.
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python multi_machine_example.py head    # Start head node")
        print("  python multi_machine_example.py worker <scheduler_address>  # Start worker")
        print("\nExample:")
        print("  # On head node machine:")
        print("  python multi_machine_example.py head")
        print("  # On worker machines:")
        print("  python multi_machine_example.py worker tcp://192.168.1.100:8786")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "head":
        start_head_node()
    elif mode == "worker":
        if len(sys.argv) < 3:
            print("Error: Worker mode requires scheduler address")
            print("Usage: python multi_machine_example.py worker <scheduler_address>")
            return
        scheduler_address = sys.argv[2]
        start_worker(scheduler_address)
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'head' or 'worker'")


if __name__ == "__main__":
    import socket
    main()

