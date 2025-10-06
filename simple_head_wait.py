#!/usr/bin/env python3
"""
Simple head node that starts and waits for external workers
"""

import asyncio
import sys
import os
import time
import socket

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

from pycluster.node import HeadNode

async def main():
    """Start head node and wait for workers"""
    print("PyCluster Simple Head Node - Waiting for Workers")
    print("=" * 50)
    
    try:
        # Create head node
        head = HeadNode(
            cluster_name='simple-test-cluster',
            host='0.0.0.0',
            scheduler_port=8786,
            dashboard_port=8787
        )
        
        print("Starting head node...")
        result = await head.start(n_local_workers=0)  # No local workers
        
        if result['status'] != 'success':
            print(f"Failed to start head node: {result}")
            return
        
        print("✅ Head node started successfully!")
        
        # Get connection info
        conn_info = head.get_connection_info()
        print(f"\n📋 Connection Information:")
        print(f"   Cluster Name: {conn_info['cluster_name']}")
        print(f"   Scheduler Address: {conn_info['scheduler_address']}")
        print(f"   Dashboard URL: {conn_info['dashboard_url']}")
        print(f"   Host IP: {conn_info['host_ip']}")
        
        print(f"\n🔍 Waiting for external workers to join...")
        print(f"   Use this address to connect workers:")
        print(f"   {conn_info['scheduler_address']}")
        print(f"\n💡 To connect a worker, run in another terminal:")
        print(f"   python phase1_worker_test.py --scheduler {conn_info['scheduler_address']}")
        print(f"   or")
        print(f"   python pycluster/join_worker_easy.py --scheduler {conn_info['scheduler_address']}")
        
        # Wait for workers to join
        from dask.distributed import Client
        
        max_wait_time = 300  # 5 minutes
        check_interval = 5   # Check every 5 seconds
        waited_time = 0
        workers_joined = False
        
        while waited_time < max_wait_time:
            try:
                client = Client(conn_info['scheduler_address'], timeout=5)
                info = client.scheduler_info()
                worker_count = info.get('n_workers', 0)
                client.close()
                
                if worker_count > 0:
                    print(f"\n🎉 Worker(s) joined! Total workers: {worker_count}")
                    
                    # List worker details
                    workers = info.get('workers', {})
                    for worker_addr, worker_info in workers.items():
                        print(f"   Worker: {worker_info.get('name', 'unknown')} at {worker_addr}")
                    
                    workers_joined = True
                    break
                else:
                    print(f"⏳ No workers yet... waiting ({waited_time}s/{max_wait_time}s)")
                    
            except Exception as e:
                print(f"⚠️  Error checking workers: {e}")
            
            time.sleep(check_interval)
            waited_time += check_interval
        
        if workers_joined:
            print(f"\n✅ Workers successfully joined the cluster!")
            print(f"   You can now run tasks on the cluster.")
            print(f"   Press Ctrl+C to stop the head node.")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n🛑 Shutting down head node...")
                head.shutdown()
                print(f"✅ Head node shutdown complete")
        else:
            print(f"\n⏰ No workers joined within {max_wait_time} seconds")
            print(f"   This is expected if no worker test is running")
            print(f"   Press Ctrl+C to stop the head node")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n🛑 Shutting down head node...")
                head.shutdown()
                print(f"✅ Head node shutdown complete")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n✅ Head node stopped by user")
    except Exception as e:
        print(f"❌ Head node crashed: {e}")
        sys.exit(1)
