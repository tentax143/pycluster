#!/usr/bin/env python3
"""
Test if head node is actually listening and accessible
"""

import socket
import sys
import os
import time

def test_port_connectivity(host, port, timeout=5):
    """Test if a port is open and accessible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Error testing {host}:{port}: {e}")
        return False

def test_dask_connection(scheduler_address):
    """Test Dask client connection"""
    try:
        from dask.distributed import Client
        
        print(f"🔄 Testing Dask client connection to {scheduler_address}...")
        client = Client(scheduler_address, timeout=10)
        
        # Test basic operations
        info = client.scheduler_info()
        print(f"✅ Connected to scheduler!")
        print(f"   Scheduler info: {info}")
        
        # Test submitting a simple task
        future = client.submit(lambda x: x * 2, 42)
        result = future.result(timeout=10)
        print(f"✅ Task execution test: 42 * 2 = {result}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Dask connection failed: {e}")
        return False

def main():
    print("🔍 Testing Head Node Connectivity")
    print("=" * 50)
    
    # Test basic port connectivity
    host = "172.16.71.183"
    scheduler_port = 8786
    dashboard_port = 8787
    
    print(f"🔄 Testing port connectivity to {host}...")
    
    # Test scheduler port
    if test_port_connectivity(host, scheduler_port):
        print(f"✅ Scheduler port {scheduler_port} is open")
    else:
        print(f"❌ Scheduler port {scheduler_port} is closed or filtered")
    
    # Test dashboard port
    if test_port_connectivity(host, dashboard_port):
        print(f"✅ Dashboard port {dashboard_port} is open")
    else:
        print(f"❌ Dashboard port {dashboard_port} is closed or filtered")
    
    # Test Dask connection
    scheduler_address = f"tcp://{host}:{scheduler_port}"
    test_dask_connection(scheduler_address)
    
    print("\n💡 Troubleshooting tips:")
    print("1. Make sure the head node is still running")
    print("2. Check Windows Firewall settings")
    print("3. Verify the IP address is correct")
    print("4. Try running both head and worker on the same machine")

if __name__ == "__main__":
    main()
