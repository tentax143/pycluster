#!/usr/bin/env python3
"""
Simple PyCluster Worker - Direct Dask Worker Implementation
This bypasses the PyCluster wrapper and uses Dask workers directly
"""

import asyncio
import logging
import sys
import os
import signal
import time
from dask.distributed import Worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleWorker:
    def __init__(self, scheduler_address, nthreads=4, memory_limit="2GB", name=None):
        self.scheduler_address = scheduler_address
        self.nthreads = nthreads
        self.memory_limit = memory_limit
        self.name = name or f"simple-worker-{os.getpid()}"
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
        """Start the worker"""
        try:
            logger.info(f"Starting simple worker: {self.name}")
            logger.info(f"Scheduler: {self.scheduler_address}")
            logger.info(f"Threads: {self.nthreads}")
            logger.info(f"Memory limit: {self.memory_limit}")
            
            # Create worker
            self.worker = Worker(
                self.scheduler_address,
                nthreads=self.nthreads,
                memory_limit=self.memory_limit,
                name=self.name
            )
            
            # Start worker
            await self.worker.start()
            
            logger.info("✅ Simple worker started successfully!")
            self.running = True
            
            # Keep the worker running
            await self._keep_alive()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Simple worker startup failed: {e}")
            return False
    
    async def _keep_alive(self):
        """Keep the worker running and monitor its health"""
        logger.info("🔄 Simple worker is running. Press Ctrl+C to stop.")
        
        while self.running:
            try:
                # Check if worker is still alive
                if self.worker and hasattr(self.worker, 'status'):
                    if self.worker.status == 'closed':
                        logger.error("❌ Worker died unexpectedly")
                        break
                
                # Sleep for a bit
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"Error in keep-alive loop: {e}")
                break
    
    async def stop(self):
        """Stop the worker gracefully"""
        logger.info("🛑 Stopping simple worker...")
        self.running = False
        
        if self.worker:
            try:
                await self.worker.close()
                logger.info("✅ Simple worker stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping worker: {e}")

async def main():
    """Main function"""
    # Configuration
    scheduler_address = "tcp://172.16.71.183:8786"
    nthreads = 4
    memory_limit = "2GB"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        scheduler_address = sys.argv[1]
    if len(sys.argv) > 2:
        nthreads = int(sys.argv[2])
    if len(sys.argv) > 3:
        memory_limit = sys.argv[3]
    
    logger.info("🚀 PyCluster Simple Worker")
    logger.info("=" * 40)
    
    worker = SimpleWorker(
        scheduler_address=scheduler_address,
        nthreads=nthreads,
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

