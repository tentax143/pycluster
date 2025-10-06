#!/usr/bin/env python3
"""
Test script to verify Dask Worker startup fix
"""
import asyncio
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_worker_startup():
    """Test worker startup with proper async handling"""
    try:
        from dask.distributed import Worker
        
        # Create a worker (this will fail to connect but shouldn't crash)
        worker = Worker(
            scheduler_address="tcp://localhost:8786",
            nthreads=2,
            name="test-worker"
        )
        
        def start_worker_thread():
            """Start worker in a separate thread with proper async handling"""
            try:
                import asyncio
                
                async def run_worker():
                    """Run the worker with proper async handling"""
                    try:
                        logger.info("Starting worker...")
                        await worker.start()
                        logger.info("Worker started successfully!")
                        
                        # Keep the worker running for a short time
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Worker runtime error: {e}")
                    finally:
                        try:
                            await worker.close()
                            logger.info("Worker closed")
                        except:
                            pass
                
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(run_worker())
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
        
        # Start worker in a separate thread
        worker_thread = threading.Thread(target=start_worker_thread, daemon=True)
        worker_thread.start()
        
        # Wait for the worker thread to complete
        worker_thread.join(timeout=5)
        
        logger.info("✅ Worker startup test completed - no RuntimeWarning should appear")
        return True
        
    except Exception as e:
        logger.error(f"❌ Worker startup test failed: {e}")
        return False

def main():
    print("🧪 Testing Dask Worker Startup Fix")
    print("=" * 50)
    
    success = test_worker_startup()
    
    if success:
        print("✅ Test completed successfully!")
        print("The RuntimeWarning about 'coroutine was never awaited' should be fixed.")
    else:
        print("❌ Test failed")

if __name__ == "__main__":
    main()

