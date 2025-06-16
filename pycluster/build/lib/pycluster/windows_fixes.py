"""
Windows-specific fixes and improvements for PyCluster
"""

import os
import sys
import time
import logging
import asyncio
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def fix_windows_event_loop():
    """
    Fix Windows event loop issues for asyncio operations.
    This is particularly important for Dask scheduler startup.
    """
    if sys.platform.startswith('win'):
        # On Windows, use ProactorEventLoop for better compatibility
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Set a reasonable timeout for operations
        if hasattr(asyncio, 'set_event_loop_policy'):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

def start_scheduler_safely(scheduler, timeout=60):
    """
    Safely start a Dask scheduler with proper async handling.
    
    Args:
        scheduler: Dask Scheduler instance
        timeout: Timeout in seconds
        
    Returns:
        Started scheduler instance
    """
    def run_scheduler():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Apply Windows fixes
            fix_windows_event_loop()
            
            async def start_async():
                await scheduler.start()
                return scheduler
            
            return loop.run_until_complete(start_async())
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise
        finally:
            try:
                loop.close()
            except:
                pass
    
    # Run in a separate thread to avoid event loop conflicts
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_scheduler)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.error(f"Scheduler startup timed out after {timeout} seconds")
            raise TimeoutError(f"Scheduler startup timed out after {timeout} seconds")

def check_windows_firewall(ports):
    """
    Check if Windows Firewall might be blocking PyCluster ports.
    
    Args:
        ports: List of ports to check
        
    Returns:
        Dict with firewall status and recommendations
    """
    if not sys.platform.startswith('win'):
        return {"status": "not_windows", "message": "Not running on Windows"}
    
    try:
        import subprocess
        
        blocked_ports = []
        for port in ports:
            try:
                # Check if port is listening
                result = subprocess.run(
                    ['netstat', '-an'], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                if f":{port}" not in result.stdout:
                    blocked_ports.append(port)
                    
            except Exception as e:
                logger.warning(f"Could not check port {port}: {e}")
        
        if blocked_ports:
            return {
                "status": "ports_blocked",
                "blocked_ports": blocked_ports,
                "message": f"Ports {blocked_ports} may be blocked by Windows Firewall",
                "recommendation": "Run 'python -m pycluster.windows_utils configure_firewall' as administrator"
            }
        else:
            return {
                "status": "ok",
                "message": "No obvious firewall issues detected"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not check firewall status: {e}"
        }

def get_windows_network_info():
    """
    Get Windows-specific network information for cluster setup.
    
    Returns:
        Dict with network configuration
    """
    if not sys.platform.startswith('win'):
        return {"status": "not_windows"}
    
    try:
        import socket
        import subprocess
        
        # Get hostname and IP
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        # Get network interfaces
        try:
            result = subprocess.run(
                ['ipconfig'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            network_info = result.stdout
        except:
            network_info = "Could not retrieve network configuration"
        
        return {
            "status": "ok",
            "hostname": hostname,
            "local_ip": local_ip,
            "network_config": network_info
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not get network info: {e}"
        }

def optimize_windows_performance():
    """
    Apply Windows-specific performance optimizations for PyCluster.
    """
    if not sys.platform.startswith('win'):
        return
    
    try:
        # Set process priority to high
        import psutil
        current_process = psutil.Process()
        current_process.nice(psutil.HIGH_PRIORITY_CLASS)
        logger.info("Set process priority to HIGH")
    except Exception as e:
        logger.warning(f"Could not set process priority: {e}")
    
    try:
        # Increase socket buffer sizes
        import socket
        socket.setdefaulttimeout(30)  # 30 second timeout
        logger.info("Set socket timeout to 30 seconds")
    except Exception as e:
        logger.warning(f"Could not configure socket settings: {e}")

def create_windows_startup_script(cluster_config_path: str, install_path: str):
    """
    Create a Windows batch script for easy PyCluster startup.
    
    Args:
        cluster_config_path: Path to cluster configuration file
        install_path: Path where PyCluster is installed
    """
    script_content = f"""@echo off
echo Starting PyCluster...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if PyCluster is installed
python -c "import pycluster" >nul 2>&1
if errorlevel 1 (
    echo ERROR: PyCluster is not installed
    echo Please install PyCluster with: pip install pycluster
    pause
    exit /b 1
)

REM Set environment variables
set PYTHONPATH={install_path}
set PYCLUSTER_CONFIG={cluster_config_path}

REM Start PyCluster
echo Starting PyCluster head node...
python -m pycluster.cli --config "%PYCLUSTER_CONFIG%"

if errorlevel 1 (
    echo.
    echo ERROR: PyCluster failed to start
    echo Check the error messages above
    echo.
    echo Common solutions:
    echo 1. Run as Administrator
    echo 2. Check Windows Firewall settings
    echo 3. Ensure no other services are using ports 8786, 8787
    echo.
    pause
)
"""
    
    script_path = os.path.join(install_path, "start_pycluster.bat")
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        logger.info(f"Created Windows startup script: {script_path}")
        return script_path
    except Exception as e:
        logger.error(f"Failed to create startup script: {e}")
        return None

def diagnose_windows_issues():
    """
    Comprehensive diagnosis of Windows-specific issues that might affect PyCluster.
    
    Returns:
        Dict with diagnosis results and recommendations
    """
    diagnosis = {
        "platform": sys.platform,
        "python_version": sys.version,
        "issues": [],
        "recommendations": []
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        diagnosis["issues"].append("Python version is too old (< 3.8)")
        diagnosis["recommendations"].append("Upgrade to Python 3.8 or newer")
    
    # Check if running as administrator
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            diagnosis["issues"].append("Not running as administrator")
            diagnosis["recommendations"].append("Run as administrator for full functionality")
    except:
        diagnosis["issues"].append("Could not check administrator status")
    
    # Check firewall status
    firewall_status = check_windows_firewall([8786, 8787, 5000])
    if firewall_status["status"] == "ports_blocked":
        diagnosis["issues"].append(f"Firewall may be blocking ports: {firewall_status['blocked_ports']}")
        diagnosis["recommendations"].append("Configure Windows Firewall to allow PyCluster ports")
    
    # Check network configuration
    network_info = get_windows_network_info()
    if network_info["status"] == "error":
        diagnosis["issues"].append("Network configuration issues detected")
        diagnosis["recommendations"].append("Check network connectivity and DNS resolution")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available < 2 * 1024 * 1024 * 1024:  # Less than 2GB
            diagnosis["issues"].append("Low available memory (< 2GB)")
            diagnosis["recommendations"].append("Close other applications to free memory")
    except:
        diagnosis["issues"].append("Could not check memory status")
    
    # Check disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free
        if free_space < 5 * 1024 * 1024 * 1024:  # Less than 5GB
            diagnosis["issues"].append("Low disk space (< 5GB)")
            diagnosis["recommendations"].append("Free up disk space")
    except:
        diagnosis["issues"].append("Could not check disk space")
    
    return diagnosis

if __name__ == "__main__":
    # Run diagnosis when script is executed directly
    print("PyCluster Windows Diagnosis")
    print("=" * 40)
    
    diagnosis = diagnose_windows_issues()
    
    print(f"Platform: {diagnosis['platform']}")
    print(f"Python: {diagnosis['python_version']}")
    print()
    
    if diagnosis["issues"]:
        print("Issues found:")
        for issue in diagnosis["issues"]:
            print(f"  ❌ {issue}")
        print()
        
        print("Recommendations:")
        for rec in diagnosis["recommendations"]:
            print(f"  💡 {rec}")
    else:
        print("✅ No issues detected!")
    
    print()
    print("Network Information:")
    network_info = get_windows_network_info()
    if network_info["status"] == "ok":
        print(f"  Hostname: {network_info['hostname']}")
        print(f"  Local IP: {network_info['local_ip']}")
    else:
        print(f"  Error: {network_info.get('message', 'Unknown error')}")

