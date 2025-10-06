#!/usr/bin/env python3
"""
Test script for PyCluster worker discovery and joining
"""

import sys
import os
import time

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

from pycluster.worker_discovery import EasyWorkerJoin

def test_worker_discovery():
    """Test worker discovery functionality"""
    print("🔍 Testing PyCluster Worker Discovery...")
    
    try:
        easy_join = EasyWorkerJoin()
        
        # Test cluster discovery
        print("🔄 Discovering clusters on network...")
        clusters = easy_join.discover_clusters(timeout=10.0)
        
        if clusters:
            print(f"✅ Found {len(clusters)} cluster(s):")
            for i, cluster in enumerate(clusters, 1):
                print(f"   {i}. {cluster.name}")
                print(f"      Scheduler: {cluster.scheduler_address}")
                print(f"      Dashboard: {cluster.dashboard_url}")
                print(f"      Workers: {cluster.workers_count}")
                print(f"      Status: {cluster.status}")
                print()
            
            # Test connection to first cluster
            first_cluster = clusters[0]
            print(f"🔗 Testing connection to {first_cluster.name}...")
            
            if easy_join.test_connection(first_cluster.scheduler_address):
                print("✅ Connection test successful!")
                
                # Get cluster info
                cluster_info = easy_join.get_cluster_info(first_cluster.scheduler_address)
                if cluster_info:
                    print("📋 Cluster Information:")
                    print(f"   ID: {cluster_info.get('id', 'unknown')}")
                    print(f"   Type: {cluster_info.get('type', 'unknown')}")
                
                return first_cluster.scheduler_address
            else:
                print("❌ Connection test failed!")
                return None
        else:
            print("❌ No clusters found on network")
            print("\n💡 Make sure:")
            print("   1. A head node is running")
            print("   2. You're on the same network")
            print("   3. Windows Firewall allows the connection")
            return None
            
    except Exception as e:
        print(f"❌ Error during discovery: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_worker_join(scheduler_address):
    """Test worker joining functionality"""
    print(f"\n🚀 Testing Worker Join to {scheduler_address}...")
    
    try:
        from pycluster.cli_enhanced import start_worker_node
        
        # Create args object for start_worker_node
        class WorkerArgs:
            def __init__(self, scheduler_addr):
                self.scheduler = scheduler_addr
                self.nthreads = None
                self.memory_limit = "auto"
        
        worker_args = WorkerArgs(scheduler_address)
        
        print("🔄 Starting worker node...")
        print("   (Press Ctrl+C to stop the worker)")
        
        # Start worker (this will run until interrupted)
        start_worker_node(worker_args)
        
    except KeyboardInterrupt:
        print("\n✅ Worker stopped by user")
    except Exception as e:
        print(f"❌ Error starting worker: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("PyCluster Worker Discovery & Join Test")
    print("=" * 50)
    
    # Test discovery
    scheduler_address = test_worker_discovery()
    
    if scheduler_address:
        # Test worker join
        test_worker_join(scheduler_address)
    else:
        print("\n❌ Cannot test worker join without a discovered cluster")
        print("\n💡 To test worker join manually:")
        print("   python pycluster/join_worker_easy.py --auto")
        print("   python pycluster/join_worker_easy.py --list")
