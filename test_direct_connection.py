#!/usr/bin/env python3
"""
Test direct connection to head node without discovery
"""

import sys
import os
import asyncio

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

from pycluster.node import WorkerNode

async def test_direct_connection():
    """Test direct connection to head node"""
    print("🔗 Testing Direct Connection to Head Node...")
    
    # Use the scheduler address from your head node output
    scheduler_address = "tcp://172.16.71.183:8786"
    
    try:
        # Create worker node
        worker = WorkerNode(
            scheduler_address=scheduler_address,
            worker_name="test-worker"
        )
        
        print(f"🔄 Connecting to {scheduler_address}...")
        
        # Start worker
        result = await worker.start(
            n_workers=1,
            threads_per_worker=2,
            memory_limit="1GB"
        )
        
        if result['status'] == 'success':
            print("✅ Worker connected successfully!")
            print(f"   Status: {result['status']}")
            
            # Get worker status
            status = worker.get_status()
            print(f"\n📊 Worker Status:")
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Worker Name: {status.get('worker_name', 'unknown')}")
            print(f"   Worker Count: {status.get('worker_count', 0)}")
            
            print("\n🎉 Direct connection test successful!")
            print("   (Press Ctrl+C to stop the worker)")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down worker...")
                worker.shutdown()
                print("✅ Worker shutdown complete")
                
        else:
            print(f"❌ Worker connection failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing direct connection: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("PyCluster Direct Connection Test")
    print("=" * 50)
    
    success = asyncio.run(test_direct_connection())
    
    if success:
        print("\n✅ Direct connection test passed!")
    else:
        print("\n❌ Direct connection test failed!")
        sys.exit(1)
