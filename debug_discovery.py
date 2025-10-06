#!/usr/bin/env python3
"""
Debug script for PyCluster discovery issues
"""

import sys
import os
import socket
import time
import json
import threading

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

def test_udp_broadcast():
    """Test UDP broadcasting functionality"""
    print("🔍 Testing UDP Broadcasting...")
    
    discovery_port = 8788
    
    try:
        # Test sending a broadcast
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Windows-specific socket options
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        test_message = {
            "type": "pycluster_announcement",
            "cluster_name": "test-cluster",
            "scheduler_address": "tcp://172.16.71.183:8786",
            "dashboard_url": "http://172.16.71.183:8787",
            "host_ip": "172.16.71.183",
            "timestamp": time.time(),
            "workers_count": 1,
            "status": "active"
        }
        
        message = json.dumps(test_message).encode('utf-8')
        
        print(f"🔄 Sending broadcast on port {discovery_port}...")
        
        # Try different broadcast addresses
        broadcast_addresses = [
            '<broadcast>',
            '255.255.255.255',
            '192.168.1.255',
            '192.168.0.255',
            '10.0.0.255',
            '172.16.255.255'  # For your network
        ]
        
        for addr in broadcast_addresses:
            try:
                sock.sendto(message, (addr, discovery_port))
                print(f"✅ Sent to {addr}:{discovery_port}")
            except Exception as e:
                print(f"❌ Failed to send to {addr}:{discovery_port} - {e}")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"❌ UDP broadcast test failed: {e}")
        return False

def test_udp_listen():
    """Test UDP listening functionality"""
    print("\n👂 Testing UDP Listening...")
    
    discovery_port = 8788
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        sock.bind(('', discovery_port))
        sock.settimeout(5.0)  # 5 second timeout
        
        print(f"🔄 Listening on port {discovery_port} for 5 seconds...")
        
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                data, addr = sock.recvfrom(1024)
                message = json.loads(data.decode('utf-8'))
                print(f"✅ Received from {addr}: {message.get('cluster_name', 'unknown')}")
                return True
            except socket.timeout:
                continue
            except Exception as e:
                print(f"❌ Error receiving: {e}")
                continue
        
        print("❌ No messages received")
        sock.close()
        return False
        
    except Exception as e:
        print(f"❌ UDP listen test failed: {e}")
        return False

def test_network_interfaces():
    """Test network interface detection"""
    print("\n🌐 Testing Network Interfaces...")
    
    try:
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            # Get network interfaces on Windows
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            print("Network interfaces:")
            print(result.stdout)
            
            # Try to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                print(f"Local IP: {local_ip}")
                
                # Calculate broadcast address
                ip_parts = local_ip.split('.')
                if len(ip_parts) == 4:
                    broadcast = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.255"
                    print(f"Calculated broadcast: {broadcast}")
        
        return True
        
    except Exception as e:
        print(f"❌ Network interface test failed: {e}")
        return False

def test_discovery_manually():
    """Test discovery manually"""
    print("\n🔧 Testing Discovery Manually...")
    
    try:
        from pycluster.worker_discovery import ClusterDiscovery, ClusterInfo
        
        # Create discovery instance
        discovery = ClusterDiscovery()
        
        # Create cluster info
        cluster_info = ClusterInfo(
            name="manual-test-cluster",
            scheduler_address="tcp://172.16.71.183:8786",
            dashboard_url="http://172.16.71.183:8787",
            host_ip="172.16.71.183",
            port=8788,
            timestamp=time.time(),
            workers_count=1,
            status="active"
        )
        
        print("🔄 Starting manual broadcasting...")
        discovery.start_broadcasting(cluster_info)
        
        print("✅ Broadcasting started, will run for 10 seconds...")
        time.sleep(10)
        
        discovery.stop()
        print("✅ Broadcasting stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("PyCluster Discovery Debug")
    print("=" * 50)
    
    # Test network interfaces
    test_network_interfaces()
    
    # Test UDP broadcasting
    test_udp_broadcast()
    
    # Test UDP listening
    test_udp_listen()
    
    # Test discovery manually
    test_discovery_manually()
    
    print("\n💡 If all tests pass, the issue might be:")
    print("   1. Windows Firewall blocking UDP broadcasts")
    print("   2. Network configuration preventing broadcasts")
    print("   3. Timing issues between head node and worker discovery")
