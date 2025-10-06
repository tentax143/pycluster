#!/usr/bin/env python3
"""
Simple worker connection test using Dask directly
"""

import sys
import os
import time

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

def test_simple_worker():
    """Test worker connection using Dask directly"""
    print("🔄 Testing Simple Worker Connection...")
    
    scheduler_address = "tcp://172.16.71.183:8786"
    
    try:
        from dask.distributed import Client, Worker
        
        print(f"🔄 Connecting to {scheduler_address}...")
        
        # Test client connection first
        client = Client(scheduler_address, timeout=10)
        print("✅ Client connected successfully!")
        
        # Get scheduler info
        info = client.scheduler_info()
        print(f"📊 Current workers: {info.get('n_workers', 0)}")
        
        # Test task execution
        future = client.submit(lambda x: x * 3, 21)
        result = future.result(timeout=10)
        print(f"✅ Task test: 21 * 3 = {result}")
        
        client.close()
        
        # Now test worker connection
        print("\n🔄 Testing worker connection...")
        
        # Create and start a worker
        worker = Worker(
            scheduler_address,
            nthreads=2,
            memory_limit="1GB",
            name="test-worker-direct"
        )
        
        print("✅ Worker created, starting...")
        
        # Start worker in a separate thread
        import threading
        worker_thread = threading.Thread(target=worker.start, daemon=True)
        worker_thread.start()
        
        # Wait a moment for worker to connect
        time.sleep(3)
        
        # Check if worker connected
        client = Client(scheduler_address, timeout=10)
        info = client.scheduler_info()
        print(f"📊 Workers after adding: {info.get('n_workers', 0)}")
        
        # List workers
        workers = info.get('workers', {})
        print("📋 Worker list:")
        for worker_addr, worker_info in workers.items():
            print(f"   - {worker_info.get('name', 'unknown')}: {worker_addr}")
        
        client.close()
        
        print("\n✅ Simple worker test successful!")
        print("   (Worker will continue running in background)")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple worker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("PyCluster Simple Worker Test")
    print("=" * 50)
    
    success = test_simple_worker()
    
    if success:
        print("\n✅ Simple worker test passed!")
    else:
        print("\n❌ Simple worker test failed!")
        sys.exit(1)
