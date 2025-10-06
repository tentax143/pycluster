#!/usr/bin/env python3
"""
Quick fix for PyCluster worker startup issues
"""
import subprocess
import sys
import os

def fix_firewall_ports():
    """Fix Windows firewall for PyCluster ports"""
    if not sys.platform.startswith('win'):
        print("This script is for Windows only")
        return False
    
    try:
        # Add firewall rules for PyCluster ports
        ports = [8786, 8787, 5000]
        
        for port in ports:
            # Add inbound rule
            cmd = [
                'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                f'name=PyCluster Port {port}',
                'dir=in',
                'action=allow',
                'protocol=TCP',
                f'localport={port}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Added firewall rule for port {port}")
            else:
                print(f"⚠️  Could not add firewall rule for port {port}: {result.stderr}")
        
        return True
    except Exception as e:
        print(f"❌ Error configuring firewall: {e}")
        return False

def test_connection(scheduler_address):
    """Test connection to scheduler"""
    try:
        import socket
        from urllib.parse import urlparse
        
        # Parse the scheduler address
        parsed = urlparse(scheduler_address)
        host = parsed.hostname
        port = parsed.port
        
        print(f"Testing connection to {host}:{port}...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("✅ Connection successful!")
            return True
        else:
            print("❌ Connection failed!")
            return False
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

def main():
    print("🔧 PyCluster Worker Startup Fix")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        print("Usage: python fix_worker_startup.py <scheduler_address>")
        print("Example: python fix_worker_startup.py tcp://172.16.71.183:8786")
        sys.exit(1)
    
    scheduler_address = sys.argv[1]
    
    # Test connection first
    if not test_connection(scheduler_address):
        print("\n🔧 Attempting to fix firewall issues...")
        if fix_firewall_ports():
            print("\n🔄 Testing connection again...")
            if test_connection(scheduler_address):
                print("✅ Firewall fix successful!")
            else:
                print("❌ Connection still failing. Check if head node is running.")
        else:
            print("❌ Could not fix firewall. Try running as Administrator.")
    
    print(f"\n🚀 Now try starting the worker:")
    print(f"python -m pycluster.cli_enhanced worker --scheduler {scheduler_address}")

if __name__ == "__main__":
    main()

