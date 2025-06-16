"""
Command line interface for PyCluster
"""

import argparse
import logging
import sys
import socket
from .node import HeadNode, WorkerNode
from .dashboard import DashboardManager

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def head_node_cli():
    """Command line interface for starting a head node."""
    parser = argparse.ArgumentParser(description="Start a PyCluster head node")
    parser.add_argument("--cluster-name", default="pycluster", 
                       help="Name of the cluster")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host address to bind to")
    parser.add_argument("--scheduler-port", type=int, default=8786,
                       help="Port for the scheduler")
    parser.add_argument("--dashboard-port", type=int, default=8787,
                       help="Port for the dashboard")
    parser.add_argument("--local-workers", type=int, default=0,
                       help="Number of local workers to start")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--open-dashboard", action="store_true",
                       help="Open dashboard in browser")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    print(f"Starting PyCluster head node: {args.cluster_name}")
    
    head = HeadNode(
        cluster_name=args.cluster_name,
        host=args.host,
        scheduler_port=args.scheduler_port,
        dashboard_port=args.dashboard_port
    )
    
    try:
        result = head.start(n_local_workers=args.local_workers)
        
        if result["status"] == "success":
            print(f"✓ Head node started successfully")
            print(f"  Scheduler: {result['scheduler_address']}")
            print(f"  Dashboard: {result['dashboard_address']}")
            
            conn_info = head.get_connection_info()
            print(f"\nConnection information:")
            print(f"  Scheduler address: {conn_info['scheduler_address']}")
            print(f"  Dashboard URL: {conn_info['dashboard_url']}")
            print(f"  Host IP: {conn_info['host_ip']}")
            
            if args.open_dashboard:
                dashboard = DashboardManager(conn_info['dashboard_url'])
                dashboard.open_in_browser()
            
            print(f"\nTo connect workers, run:")
            print(f"  pycluster-worker --scheduler {conn_info['scheduler_address']}")
            
            print(f"\nHead node running. Press Ctrl+C to stop...")
            
            # Keep running until interrupted
            while True:
                import time
                time.sleep(1)
                
        else:
            print(f"✗ Failed to start head node: {result.get('message')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nShutting down head node...")
        head.shutdown()
        print("✓ Head node stopped")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        sys.exit(1)


def worker_node_cli():
    """Command line interface for starting a worker node."""
    parser = argparse.ArgumentParser(description="Start a PyCluster worker node")
    parser.add_argument("--scheduler", required=True,
                       help="Scheduler address to connect to (e.g., tcp://192.168.1.100:8786)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of workers to start")
    parser.add_argument("--threads", type=int, default=None,
                       help="Number of threads per worker")
    parser.add_argument("--memory-limit", default="auto",
                       help="Memory limit per worker")
    parser.add_argument("--name", default=None,
                       help="Name for this worker node")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    worker_name = args.name or f"worker-{socket.gethostname()}"
    print(f"Starting PyCluster worker node: {worker_name}")
    print(f"Connecting to scheduler: {args.scheduler}")
    
    worker = WorkerNode(
        scheduler_address=args.scheduler,
        worker_name=worker_name
    )
    
    try:
        result = worker.start(
            n_workers=args.workers,
            threads_per_worker=args.threads,
            memory_limit=args.memory_limit
        )
        
        if result["status"] == "success":
            print(f"✓ Worker node started successfully")
            print(f"  Workers added: {result['workers_added']}")
            print(f"  Total workers: {result['total_workers']}")
            
            print(f"\nWorker node running. Press Ctrl+C to stop...")
            
            # Keep running until interrupted
            while True:
                import time
                time.sleep(10)
                status = worker.get_status()
                print(f"Status: {status.get('status')}, Workers: {status.get('worker_count', 0)}")
                
        else:
            print(f"✗ Failed to start worker node: {result.get('message')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nShutting down worker node...")
        worker.shutdown()
        print("✓ Worker node stopped")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_node_cli()
    else:
        head_node_cli()

