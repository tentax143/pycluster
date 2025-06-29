#!/usr/bin/env python3
"""
PyCluster Simple Worker Join Script

This script makes it easy to join a PyCluster as a worker node.
It bypasses the typing compatibility issues by using direct Dask imports.
"""

import sys
import os
import argparse
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple PyCluster Worker Join")
    parser.add_argument("--scheduler", required=True, help="Scheduler address (e.g., tcp://192.168.1.100:8786)")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of Dask worker processes to start")
    parser.add_argument("--threads-per-worker", type=int, help="Number of threads per worker")
    parser.add_argument("--memory-limit", default="auto", help="Memory limit per worker")
    parser.add_argument("--name", help="Worker name")
    return parser.parse_args()

async def start_worker(args, worker_name, idx):
    from dask.distributed import Worker
    worker = Worker(
        args.scheduler,
        nthreads=args.threads_per_worker,
        memory_limit=args.memory_limit,
        name=f"{worker_name}-{idx}" if args.n_workers > 1 else worker_name
    )
    await worker.start()
    print(f"  ✓ Started worker {idx+1}/{args.n_workers}")
    await worker.finished()  # Keep the worker running

def main():
    args = parse_args()
    import socket
    worker_name = args.name or f"worker-{socket.gethostname()}"
    print("🔍 PyCluster Simple Worker Join")
    print("=" * 40)
    print(f"📡 Connecting to scheduler: {args.scheduler}")
    print(f"🚀 Starting {args.n_workers} worker(s) with name: {worker_name}")

    async def run_all_workers():
        await asyncio.gather(*[
            start_worker(args, worker_name, i)
            for i in range(args.n_workers)
        ])

    try:
        asyncio.run(run_all_workers())
    except KeyboardInterrupt:
        print("\n🛑 Stopping workers...")
        print("✅ Workers stopped")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages:")
        print("  pip install dask[complete] distributed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start workers: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure the head node is running")
        print("2. Check network connectivity")
        print("3. Verify the scheduler address is correct")
        print("4. Check Windows Firewall settings")
        sys.exit(1)

if __name__ == "__main__":
    main() 