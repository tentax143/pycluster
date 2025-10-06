#!/usr/bin/env python3
"""
Synchronous PyCluster Worker - No Async, Just Pure Dask
This is the most reliable approach for worker startup
"""

import logging
import sys
import os
import signal
import time
import threading
from dask.distributed import Worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyncWorker:
    def __init__(self, scheduler_address, nthreads=4, memory_limit="2GB", name=None):
        self.scheduler_address = scheduler_address
        self.nthreads = nthreads
        self.memory_limit = memory_limit
        self.name = name or f"sync-worker-{os.getpid()}"
        self.worker = None
        self.running = False
        self.worker_thread = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.stop()
    
    def start(self):
        """Start the worker synchronously"""
        try:
            logger.info(f"Starting sync worker: {self.name}")
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
            
            # Start worker in a separate thread
            self.worker_thread = threading.Thread(
                target=self._run_worker,
                daemon=True
            )
            self.worker_thread.start()
            
            # Wait a moment for worker to start
            time.sleep(2)
            
            if self.worker.status == 'running':
                logger.info("✅ Sync worker started successfully!")
                self.running = True
                return True
            else:
                logger.error(f"❌ Worker failed to start, status: {self.worker.status}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Sync worker startup failed: {e}")
            return False
    
    def _run_worker(self):
        """Run the worker in a separate thread"""
        try:
            # Start the worker (this is synchronous)
            self.worker.start()
            
            # Keep running until stopped
            while self.running and self.worker.status == 'running':
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Worker thread error: {e}")
        finally:
            try:
                if self.worker:
                    self.worker.close()
            except:
                pass
    
    def stop(self):
        """Stop the worker gracefully"""
        logger.info("🛑 Stopping sync worker...")
        self.running = False
        
        if self.worker:
            try:
                self.worker.close()
                logger.info("✅ Sync worker stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping worker: {e}")
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
    
    def keep_alive(self):
        """Keep the main thread alive"""
        logger.info("🔄 Sync worker is running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.stop()

def main():
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
    
    logger.info("🚀 PyCluster Sync Worker")
    logger.info("=" * 40)
    
    worker = SyncWorker(
        scheduler_address=scheduler_address,
        nthreads=nthreads,
        memory_limit=memory_limit
    )
    
    try:
        success = worker.start()
        if not success:
            logger.error("❌ Failed to start worker")
            sys.exit(1)
        
        # Keep the main thread alive
        worker.keep_alive()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        worker.stop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

