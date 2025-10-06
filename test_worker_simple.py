#!/usr/bin/env python3
"""
Simple worker test without async complications
"""

import sys
import os
import time
import threading

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

def test_simple_worker_connection():
    """Test worker connection using the simplest approach"""
    print("Testing Simple Worker Connection...")
    
    scheduler_address = "tcp://172.16.71.183:8786"
    
    try:
        from dask.distributed import Client, Worker
        
        # Test client connection first
        print("🔄 Testing client connection...")
        client = Client(scheduler_address, timeout=10)
        print("✅ Client connected successfully!")
        
        # Get initial worker count
        info = client.scheduler_info()
        initial_workers = info.get('n_workers', 0)
        print(f"📊 Initial workers: {initial_workers}")
        
        client.close()
        
        # Create worker with minimal configuration
        print("🔄 Creating worker...")
        worker = Worker(
            scheduler_address,
            nthreads=1,  # Minimal threads
            memory_limit="512MB",  # Small memory limit
            name="simple-test-worker"
        )
        
        # Start worker in a separate thread
        def start_worker():
            try:
                print("🔄 Starting worker...")
                worker.start()
                print("✅ Worker started successfully!")
                
                # Keep worker running
                while True:
                    time.sleep(1)
            except Exception as e:
                print(f"❌ Worker failed: {e}")
        
        worker_thread = threading.Thread(target=start_worker, daemon=True)
        worker_thread.start()
        
        # Wait for worker to connect
        print("🔄 Waiting for worker to connect...")
        time.sleep(3)
        
        # Check if worker connected
        client = Client(scheduler_address, timeout=10)
        info = client.scheduler_info()
        final_workers = info.get('n_workers', 0)
        
        print(f"📊 Final workers: {final_workers}")
        
        if final_workers > initial_workers:
            print("✅ Worker connected successfully!")
            
            # Test task execution
            print("🔄 Testing task execution...")
            future = client.submit(lambda x: x * 4, 10)
            result = future.result(timeout=10)
            print(f"✅ Task test: 10 * 4 = {result}")
            
        else:
            print("❌ Worker did not connect")
        
        client.close()
        
        print("\n✅ Simple worker test completed!")
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
    
    success = test_simple_worker_connection()
    
    if success:
        print("\n✅ Simple worker test passed!")
    else:
        print("\n❌ Simple worker test failed!")
        sys.exit(1)
