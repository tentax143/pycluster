#!/usr/bin/env python3
"""
Test head node and worker on the same machine
"""

import asyncio
import sys
import os
import time
import threading

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

from pycluster.node import HeadNode, WorkerNode

async def test_same_machine():
    """Test head node and worker on the same machine"""
    print("🔄 Testing Head Node and Worker on Same Machine...")
    
    try:
        # Start head node
        print("🚀 Starting head node...")
        head = HeadNode(
            cluster_name='same-machine-test',
            host='127.0.0.1',  # Use localhost
            scheduler_port=8786,
            dashboard_port=8787
        )
        
        result = await head.start(n_local_workers=0)  # No local workers initially
        
        if result['status'] != 'success':
            print(f"❌ Head node failed to start: {result}")
            return False
        
        print("✅ Head node started successfully!")
        conn_info = head.get_connection_info()
        print(f"   Scheduler: {conn_info['scheduler_address']}")
        print(f"   Dashboard: {conn_info['dashboard_url']}")
        
        # Wait a moment for head node to be ready
        await asyncio.sleep(2)
        
        # Start worker
        print("\n🔄 Starting worker...")
        worker = WorkerNode(
            scheduler_address=conn_info['scheduler_address'],
            worker_name="same-machine-worker"
        )
        
        worker_result = await worker.start(
            n_workers=1,
            threads_per_worker=2,
            memory_limit="1GB"
        )
        
        if worker_result['status'] == 'success':
            print("✅ Worker connected successfully!")
            
            # Get cluster status
            status = head.get_cluster_status()
            print(f"\n📊 Cluster Status:")
            print(f"   Workers: {len(status.get('workers', []))}")
            
            # Test a simple task
            print("\n🔄 Testing task execution...")
            try:
                from dask.distributed import Client
                client = Client(conn_info['scheduler_address'])
                
                # Submit a simple task
                future = client.submit(lambda x: x * 2, 42)
                result = future.result(timeout=10)
                print(f"✅ Task test: 42 * 2 = {result}")
                
                client.close()
                
            except Exception as e:
                print(f"❌ Task execution failed: {e}")
            
            print("\n🎉 Same-machine test successful!")
            print("   (Press Ctrl+C to stop)")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                worker.shutdown()
                head.shutdown()
                print("✅ Shutdown complete")
                
        else:
            print(f"❌ Worker failed to connect: {worker_result}")
            head.shutdown()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Same-machine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("PyCluster Same-Machine Test")
    print("=" * 50)
    
    success = asyncio.run(test_same_machine())
    
    if success:
        print("\n✅ Same-machine test passed!")
    else:
        print("\n❌ Same-machine test failed!")
        sys.exit(1)
