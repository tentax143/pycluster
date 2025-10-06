#!/usr/bin/env python3
"""
Stable PyCluster Worker Startup Script
This script provides a more stable way to start PyCluster workers
"""

import asyncio
import logging
import sys
import os
import signal
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

class StableWorker:
    def __init__(self, scheduler_address, n_workers=1, threads_per_worker=4, memory_limit="2GB"):
        self.scheduler_address = scheduler_address
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.worker = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def start(self):
        """Start the worker with retry logic"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting worker (attempt {attempt + 1}/{max_retries})")
                logger.info(f"Scheduler: {self.scheduler_address}")
                logger.info(f"Workers: {self.n_workers}")
                logger.info(f"Threads per worker: {self.threads_per_worker}")
                logger.info(f"Memory limit: {self.memory_limit}")
                
                from pycluster.node import WorkerNode
                
                self.worker = WorkerNode(scheduler_address=self.scheduler_address)
                
                result = await self.worker.start(
                    n_workers=self.n_workers,
                    threads_per_worker=self.threads_per_worker,
                    memory_limit=self.memory_limit
                )
                
                logger.info("✅ Worker started successfully!")
                logger.info(f"Worker info: {result}")
                self.running = True
                
                # Keep the worker running
                await self._keep_alive()
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Worker startup failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("❌ All startup attempts failed")
                    return False
    
    async def _keep_alive(self):
        """Keep the worker running and monitor its health"""
        logger.info("🔄 Worker is running. Press Ctrl+C to stop.")
        
        while self.running:
            try:
                # Check if worker is still alive
                if self.worker and hasattr(self.worker, 'is_alive'):
                    if not self.worker.is_alive():
                        logger.error("❌ Worker died unexpectedly")
                        break
                
                # Sleep for a bit
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"Error in keep-alive loop: {e}")
                break
    
    async def stop(self):
        """Stop the worker gracefully"""
        logger.info("🛑 Stopping worker...")
        self.running = False
        
        if self.worker:
            try:
                self.worker.stop()
                logger.info("✅ Worker stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping worker: {e}")

async def main():
    """Main function"""
    # Configuration
    scheduler_address = "tcp://172.16.71.183:8786"
    n_workers = 1  # Start with just 1 worker
    threads_per_worker = 4  # Conservative thread count
    memory_limit = "2GB"  # Conservative memory limit
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        scheduler_address = sys.argv[1]
    if len(sys.argv) > 2:
        n_workers = int(sys.argv[2])
    if len(sys.argv) > 3:
        threads_per_worker = int(sys.argv[3])
    if len(sys.argv) > 4:
        memory_limit = sys.argv[4]
    
    logger.info("🚀 PyCluster Stable Worker")
    logger.info("=" * 40)
    
    worker = StableWorker(
        scheduler_address=scheduler_address,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    
    try:
        success = await worker.start()
        if not success:
            logger.error("❌ Failed to start worker")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await worker.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
