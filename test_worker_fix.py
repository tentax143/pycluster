#!/usr/bin/env python3
"""
Test script to verify PyCluster worker startup fix
"""
import asyncio
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_worker_startup():
    """Test worker startup with the fixed code"""
    try:
        from pycluster.node import WorkerNode
        
        # Test with a dummy scheduler address (this will fail to connect but shouldn't crash)
        scheduler_address = "tcp://localhost:8786"
        
        logger.info(f"Testing worker startup with scheduler: {scheduler_address}")
        
        worker = WorkerNode(scheduler_address=scheduler_address)
        
        # This should not crash with the RuntimeWarning anymore
        result = await worker.start(
            n_workers=1,
            threads_per_worker=2,
            memory_limit="auto"
        )
        
        logger.info(f"Worker startup result: {result}")
        
        # Cleanup
        worker.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"Worker startup test failed: {e}")
        return False

def main():
    print("🧪 Testing PyCluster Worker Startup Fix")
    print("=" * 50)
    
    try:
        success = asyncio.run(test_worker_startup())
        
        if success:
            print("✅ Worker startup test completed (connection may fail, but no crashes)")
            print("\nThe RuntimeWarning about 'coroutine was never awaited' should be fixed.")
        else:
            print("❌ Worker startup test failed")
            
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()

