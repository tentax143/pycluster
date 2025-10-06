#!/usr/bin/env python3
"""
PyCluster Worker Diagnostic Tool
Helps identify and fix worker startup issues
"""

import asyncio
import logging
import sys
import os
import socket
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_port_availability(host, port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception as e:
        logger.error(f"Error checking port {host}:{port}: {e}")
        return False

def check_scheduler_connection(scheduler_address):
    """Check if scheduler is reachable"""
    try:
        # Parse scheduler address
        if scheduler_address.startswith("tcp://"):
            address = scheduler_address[6:]  # Remove tcp://
        else:
            address = scheduler_address
        
        host, port = address.split(":")
        port = int(port)
        
        logger.info(f"Checking scheduler connection to {host}:{port}")
        
        if check_port_availability(host, port):
            logger.info("✅ Scheduler is reachable")
            return True
        else:
            logger.error("❌ Scheduler is not reachable")
            return False
            
    except Exception as e:
        logger.error(f"Error checking scheduler connection: {e}")
        return False

def check_system_resources():
    """Check system resources"""
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        logger.info(f"Memory: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        logger.info(f"CPU cores: {cpu_count}")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        logger.info(f"Disk space: {disk.free / (1024**3):.2f} GB free")
        
        return True
        
    except ImportError:
        logger.warning("psutil not available, skipping system resource check")
        return False
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return False

def check_dask_installation():
    """Check Dask installation"""
    try:
        import dask
        import distributed
        logger.info(f"✅ Dask version: {dask.__version__}")
        logger.info(f"✅ Distributed version: {distributed.__version__}")
        return True
    except ImportError as e:
        logger.error(f"❌ Dask not properly installed: {e}")
        return False

def check_pycluster_installation():
    """Check PyCluster installation"""
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))
        
        from pycluster import WorkerNode, HeadNode
        logger.info("✅ PyCluster modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ PyCluster not properly installed: {e}")
        return False

def test_worker_startup(scheduler_address):
    """Test worker startup with minimal configuration"""
    try:
        logger.info("Testing worker startup...")
        
        # Add current directory to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))
        
        from pycluster.node import WorkerNode
        
        async def test_worker():
            try:
                worker = WorkerNode(scheduler_address=scheduler_address)
                
                # Test with minimal configuration
                result = await worker.start(
                    n_workers=1,
                    threads_per_worker=2,
                    memory_limit="1GB"
                )
                
                logger.info("✅ Worker started successfully")
                logger.info(f"Worker info: {result}")
                
                # Clean shutdown
                worker.stop()
                logger.info("✅ Worker stopped cleanly")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Worker startup failed: {e}")
                return False
        
        # Run the test
        return asyncio.run(test_worker())
        
    except Exception as e:
        logger.error(f"❌ Worker test failed: {e}")
        return False

def fix_common_issues():
    """Attempt to fix common issues"""
    logger.info("Attempting to fix common issues...")
    
    # Check if running as administrator on Windows
    if sys.platform.startswith('win'):
        try:
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            if not is_admin:
                logger.warning("⚠️  Not running as administrator. Some features may not work properly.")
                logger.info("💡 Try running as administrator for better compatibility")
        except:
            pass
    
    # Check firewall
    logger.info("💡 If you're having connection issues, check Windows Firewall settings")
    logger.info("💡 Make sure ports 8786, 8787, and worker ports are open")

def main():
    """Main diagnostic function"""
    logger.info("🔍 PyCluster Worker Diagnostic Tool")
    logger.info("=" * 50)
    
    # Get scheduler address from user or use default
    if len(sys.argv) > 1:
        scheduler_address = sys.argv[1]
    else:
        scheduler_address = "tcp://172.16.71.183:8786"
    
    logger.info(f"Target scheduler: {scheduler_address}")
    logger.info("")
    
    # Run diagnostics
    checks = [
        ("System Resources", check_system_resources),
        ("Dask Installation", check_dask_installation),
        ("PyCluster Installation", check_pycluster_installation),
        ("Scheduler Connection", lambda: check_scheduler_connection(scheduler_address)),
        ("Worker Startup Test", lambda: test_worker_startup(scheduler_address))
    ]
    
    results = {}
    for check_name, check_func in checks:
        logger.info(f"🔍 Running {check_name} check...")
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"❌ {check_name} check failed: {e}")
            results[check_name] = False
        logger.info("")
    
    # Summary
    logger.info("📊 Diagnostic Summary")
    logger.info("=" * 50)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{check_name}: {status}")
    
    # Recommendations
    logger.info("")
    logger.info("💡 Recommendations:")
    
    if not results.get("Scheduler Connection", False):
        logger.info("- Check if the head node is running on the scheduler address")
        logger.info("- Verify network connectivity and firewall settings")
    
    if not results.get("Worker Startup Test", False):
        logger.info("- Try reducing the number of workers or threads")
        logger.info("- Check available system resources")
        logger.info("- Ensure no other Dask workers are running on the same machine")
    
    if not results.get("PyCluster Installation", False):
        logger.info("- Reinstall PyCluster: pip install -e .")
        logger.info("- Check Python path and module imports")
    
    # Fix common issues
    fix_common_issues()

if __name__ == "__main__":
    main()
