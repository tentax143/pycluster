#!/usr/bin/env python3
"""
PyCluster Easy Worker Join Script

This script makes it easy to join a PyCluster as a worker node.
It will automatically discover available clusters on your network.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pycluster.worker_discovery import EasyWorkerJoin
    from pycluster.cli_enhanced import start_worker_node
except ImportError:
    print("❌ PyCluster not found. Please install PyCluster first:")
    print("   pip install .")
    sys.exit(1)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Easy PyCluster Worker Join")
    parser.add_argument("--cluster-name", help="Specific cluster name to join")
    parser.add_argument("--scheduler", help="Direct scheduler address (e.g., tcp://192.168.1.100:8786)")
    parser.add_argument("--auto", action="store_true", help="Auto-join first available cluster")
    parser.add_argument("--list", action="store_true", help="List available clusters and exit")
    parser.add_argument("--timeout", type=float, default=10.0, help="Discovery timeout in seconds")
    parser.add_argument("--nthreads", type=int, help="Number of threads per worker")
    parser.add_argument("--memory-limit", default="auto", help="Memory limit per worker")
    
    args = parser.parse_args()
    
    print("🔍 PyCluster Easy Worker Join")
    print("=" * 40)
    
    easy_join = EasyWorkerJoin()
    
    if args.list:
        # Just list available clusters
        print("Searching for clusters...")
        clusters = easy_join.discover_clusters(timeout=args.timeout)
        if clusters:
            print(f"\n✅ Found {len(clusters)} cluster(s):")
            for cluster in clusters:
                print(f"  📊 {cluster.name}")
                print(f"     Scheduler: {cluster.scheduler_address}")
                print(f"     Dashboard: {cluster.dashboard_url}")
                print(f"     Workers: {cluster.workers_count}")
                print()
        else:
            print("❌ No clusters found")
        return
    
    scheduler_address = None
    
    if args.scheduler:
        # Direct scheduler address provided
        scheduler_address = args.scheduler
        print(f"📡 Using provided scheduler: {scheduler_address}")
        
    elif args.cluster_name:
        # Join specific cluster by name
        print(f"🔍 Searching for cluster: {args.cluster_name}")
        scheduler_address = easy_join.join_cluster_by_name(args.cluster_name, args.timeout)
        if not scheduler_address:
            print(f"❌ Cluster '{args.cluster_name}' not found")
            print("\nTry:")
            print("  python join_worker.py --list    # List available clusters")
            print("  python join_worker.py           # Interactive selection")
            sys.exit(1)
            
    elif args.auto:
        # Auto-join first available cluster
        print("🔍 Searching for clusters...")
        clusters = easy_join.discover_clusters(timeout=args.timeout)
        if clusters:
            scheduler_address = clusters[0].scheduler_address
            print(f"✅ Auto-joining cluster: {clusters[0].name}")
        else:
            print("❌ No clusters found for auto-join")
            print("\nTroubleshooting:")
            print("1. Ensure a PyCluster head node is running")
            print("2. Check Windows Firewall settings")
            print("3. Verify you're on the same network")
            sys.exit(1)
            
    else:
        # Interactive selection
        scheduler_address = easy_join.join_cluster_interactive()
        if not scheduler_address:
            print("❌ No cluster selected")
            sys.exit(1)
    
    # Test connection before starting worker
    print(f"🔗 Testing connection to: {scheduler_address}")
    if not easy_join.test_connection(scheduler_address):
        print(f"❌ Cannot connect to scheduler: {scheduler_address}")
        print("\nPlease check:")
        print("1. Head node is running")
        print("2. Network connectivity")
        print("3. Windows Firewall settings")
        print("4. Correct IP address and port")
        sys.exit(1)
    
    print(f"🚀 Starting worker, connecting to: {scheduler_address}")
    
    # Create args object for start_worker_node
    class WorkerArgs:
        def __init__(self, scheduler_addr, nthreads=None, memory_limit="auto"):
            self.scheduler = scheduler_addr
            self.nthreads = nthreads
            self.memory_limit = memory_limit
    
    worker_args = WorkerArgs(scheduler_address, args.nthreads, args.memory_limit)
    
    try:
        start_worker_node(worker_args)
    except KeyboardInterrupt:
        print("\n✅ Worker stopped by user")
    except Exception as e:
        print(f"❌ Worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

