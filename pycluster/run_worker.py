#!/usr/bin/env python3
"""
PyCluster Worker Runner with Environment Fixes

This script sets environment variables to avoid Python 3.12+ typing issues
and then runs the worker join script.
"""

import os
import sys
import subprocess

def main():
    # Set environment variables to avoid typing issues
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Try to import the simple worker script
    try:
        from join_worker_simple import main as worker_main
        worker_main()
    except ImportError:
        print("❌ Could not import worker script")
        print("Trying alternative approach...")
        
        # Alternative: run the original script with environment fixes
        try:
            # Set environment variables for the subprocess
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = '0'
            env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
            
            # Run the original join_worker.py
            result = subprocess.run([
                sys.executable, 'join_worker.py'
            ] + sys.argv[1:], env=env)
            
            sys.exit(result.returncode)
            
        except Exception as e:
            print(f"❌ Failed to run worker: {e}")
            print("\nManual workaround:")
            print("1. Set environment variable: set PYTHONHASHSEED=0")
            print("2. Run: python join_worker_simple.py --scheduler tcp://172.16.71.183:8786 --n-workers 20")
            sys.exit(1)

if __name__ == "__main__":
    main() 