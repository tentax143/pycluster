#!/usr/bin/env python3
"""
Simple worker that connects to head node
"""

import asyncio
import sys
import os
import time
import argparse

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

from pycluster.node import WorkerNode

async def main():
    """Connect worker to head node"""
    parser = argparse.ArgumentParser(description="Simple PyCluster Worker")
    parser.add_argument("--scheduler", required=True, help="Scheduler address (e.g., tcp://192.168.1.100:8786)")
    args = parser.parse_args()
    
    print("PyCluster Simple Worker - Connecting to Head Node")
    print("=" * 50)
    
    try:
        print(f"Connecting to scheduler: {args.scheduler}")
        
        # Create worker node
        worker = WorkerNode(
            scheduler_address=args.scheduler,
            worker_name="simple-test-worker"
        )
        
        print("Starting worker...")
        result = await worker.start(
            n_workers=1,
            threads_per_worker=2,
            memory_limit="1GB"
        )
        
        if result['status'] != 'success':
            print(f"❌ Failed to connect worker: {result}")
            return
        
        print("✅ Worker connected successfully!")
        
        # Get worker status
        status = worker.get_status()
        print(f"\n📊 Worker Status:")
        print(f"   Status: {status.get('status', 'unknown')}")
        print(f"   Worker Name: {status.get('worker_name', 'unknown')}")
        print(f"   Worker Count: {status.get('worker_count', 0)}")
        print(f"   Scheduler: {status.get('scheduler_address', 'unknown')}")
        
        # Test task execution
        print(f"\n🧪 Testing task execution...")
        try:
            from dask.distributed import Client
            
            client = Client(args.scheduler, timeout=10)
            
            # Test simple tasks
            tasks = [
                (lambda x: x * 2, 21),
                (lambda x: x + 10, 5),
                (lambda x: x ** 2, 4)
            ]
            
            for func, arg in tasks:
                future = client.submit(func, arg)
                result = future.result(timeout=10)
                print(f"   Task {func.__name__}({arg}) = {result}")
            
            # Test parallel execution
            futures = [client.submit(lambda x: x * i, 10) for i in range(1, 4)]
            results = [f.result(timeout=10) for f in futures]
            print(f"   Parallel tasks: {results}")
            
            client.close()
            print(f"✅ Task execution successful!")
            
        except Exception as e:
            print(f"❌ Task execution failed: {e}")
        
        print(f"\n🎉 Worker is running and ready for tasks!")
        print(f"   Press Ctrl+C to stop the worker")
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Shutting down worker...")
            worker.shutdown()
            print(f"✅ Worker shutdown complete")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n✅ Worker stopped by user")
    except Exception as e:
        print(f"❌ Worker crashed: {e}")
        sys.exit(1)
