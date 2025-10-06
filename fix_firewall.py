#!/usr/bin/env python3
"""
Quick firewall fix for PyCluster
"""
import subprocess
import sys
import os

def is_admin():
    """Check if running as administrator"""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def fix_firewall():
    """Fix Windows firewall for PyCluster ports"""
    if not sys.platform.startswith('win'):
        print("This script is for Windows only")
        return False
    
    if not is_admin():
        print("❌ This script must be run as Administrator")
        print("Right-click on Command Prompt and select 'Run as administrator'")
        return False
    
    try:
        # Add firewall rules for PyCluster ports
        ports = [8786, 8787, 5000]
        
        print("🔧 Adding Windows Firewall rules for PyCluster...")
        
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
                print(f"⚠️  Could not add firewall rule for port {port}")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()}")
        
        print("\n✅ Firewall configuration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error configuring firewall: {e}")
        return False

def main():
    print("🛡️  PyCluster Firewall Fix")
    print("=" * 30)
    
    if fix_firewall():
        print("\n🎉 Firewall is now configured for PyCluster!")
        print("\nYou can now try starting your worker again:")
        print("python -m pycluster.cli_enhanced worker --scheduler tcp://172.16.71.183:8786")
    else:
        print("\n❌ Could not configure firewall")
        print("Please run this script as Administrator")

if __name__ == "__main__":
    main()

