#!/usr/bin/env python3
"""
Get the scheduler address from a running head node
"""

import sys
import os
import socket
import time

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

def get_scheduler_address():
    """Get scheduler address from discovery or return default"""
    try:
        from pycluster.worker_discovery import EasyWorkerJoin
        
        easy_join = EasyWorkerJoin()
        clusters = easy_join.discover_clusters(timeout=5.0)
        
        if clusters:
            scheduler_address = clusters[0].scheduler_address
            print(f"Found cluster: {clusters[0].name}")
            print(f"Scheduler address: {scheduler_address}")
            return scheduler_address
        else:
            print("No clusters found via discovery")
            return None
            
    except Exception as e:
        print(f"Discovery failed: {e}")
        return None

def test_connection(scheduler_address):
    """Test connection to scheduler"""
    try:
        from dask.distributed import Client
        
        print(f"Testing connection to {scheduler_address}...")
        client = Client(scheduler_address, timeout=5)
        
        info = client.scheduler_info()
        print(f"Connected successfully!")
        print(f"Workers: {info.get('n_workers', 0)}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Getting scheduler address...")
    
    # Try discovery first
    scheduler_address = get_scheduler_address()
    
    if not scheduler_address:
        # Try common addresses
        common_addresses = [
            "tcp://172.16.71.183:8786",
            "tcp://127.0.0.1:8786",
            "tcp://localhost:8786"
        ]
        
        for addr in common_addresses:
            print(f"Trying {addr}...")
            if test_connection(addr):
                scheduler_address = addr
                break
    
    if scheduler_address:
        print(f"\nScheduler address: {scheduler_address}")
        print(f"Use this address for worker tests:")
        print(f"python phase1_worker_test.py --scheduler {scheduler_address}")
    else:
        print("\nNo scheduler found. Make sure a head node is running.")
        sys.exit(1)
