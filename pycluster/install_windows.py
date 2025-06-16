"""
Windows installation script for PyCluster
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    return True


def check_pip():
    """Check if pip is available."""
    try:
        import pip
        return True
    except ImportError:
        print("Error: pip is not installed")
        return False


def install_dependencies():
    """Install PyCluster dependencies."""
    dependencies = [
        "dask[complete]>=2023.1.0",
        "distributed>=2023.1.0", 
        "psutil>=5.8.0",
        "requests>=2.25.0"
    ]
    
    print("Installing dependencies...")
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {dep}: {e}")
            return False
    
    return True


def install_pycluster():
    """Install PyCluster package."""
    print("Installing PyCluster...")
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    try:
        # Install in development mode
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", str(script_dir)
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install PyCluster: {e}")
        return False


def create_desktop_shortcuts():
    """Create desktop shortcuts for PyCluster tools."""
    if platform.system() != "Windows":
        return True
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        
        # Create shortcut for PyCluster Head Node
        path = os.path.join(desktop, "PyCluster Head Node.lnk")
        target = sys.executable
        arguments = "-m pycluster.cli"
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.Arguments = arguments
        shortcut.WorkingDirectory = os.path.dirname(target)
        shortcut.IconLocation = target
        shortcut.save()
        
        print("Desktop shortcuts created")
        return True
        
    except ImportError:
        print("Note: Could not create desktop shortcuts (winshell not available)")
        return True
    except Exception as e:
        print(f"Warning: Could not create desktop shortcuts: {e}")
        return True


def create_start_menu_entries():
    """Create Start Menu entries for PyCluster."""
    if platform.system() != "Windows":
        return True
    
    try:
        # Create batch files for easy access
        scripts_dir = Path.home() / "AppData" / "Local" / "PyCluster" / "Scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Head node script
        head_script = scripts_dir / "start_head_node.bat"
        with open(head_script, 'w') as f:
            f.write(f'''@echo off
title PyCluster Head Node
echo Starting PyCluster Head Node...
"{sys.executable}" -m pycluster.cli
pause
''')
        
        # Worker node script
        worker_script = scripts_dir / "start_worker_node.bat"
        with open(worker_script, 'w') as f:
            f.write(f'''@echo off
title PyCluster Worker Node
echo Starting PyCluster Worker Node...
set /p SCHEDULER="Enter scheduler address (e.g., tcp://192.168.1.100:8786): "
"{sys.executable}" -m pycluster.cli worker --scheduler %SCHEDULER%
pause
''')
        
        print(f"Start scripts created in {scripts_dir}")
        return True
        
    except Exception as e:
        print(f"Warning: Could not create start menu entries: {e}")
        return True


def setup_firewall_rules():
    """Setup Windows firewall rules for PyCluster."""
    if platform.system() != "Windows":
        return True
    
    print("Setting up Windows firewall rules...")
    
    ports = [8786, 8787, 8788]  # Scheduler, Dashboard, Discovery
    
    for port in ports:
        try:
            # Add inbound rule
            cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name=PyCluster Port {port}",
                "dir=in",
                "action=allow", 
                "protocol=TCP",
                f"localport={port}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Firewall rule added for port {port}")
            else:
                print(f"Note: Could not add firewall rule for port {port} (may require admin privileges)")
        
        except Exception as e:
            print(f"Warning: Firewall setup failed for port {port}: {e}")
    
    return True


def verify_installation():
    """Verify that PyCluster was installed correctly."""
    print("Verifying installation...")
    
    try:
        import pycluster
        print(f"✓ PyCluster {pycluster.__version__} installed successfully")
        
        # Test basic functionality
        from pycluster import ClusterManager, HeadNode, WorkerNode
        print("✓ Core components available")
        
        from pycluster.windows_utils import WindowsClusterManager
        from pycluster.network_utils import NetworkDiscovery
        print("✓ Windows utilities available")
        
        return True
        
    except ImportError as e:
        print(f"✗ Installation verification failed: {e}")
        return False


def main():
    """Main installation function."""
    print("PyCluster Windows Installation")
    print("=" * 40)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_pip():
        return False
    
    print(f"Python {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies")
        return False
    
    # Install PyCluster
    if not install_pycluster():
        print("Failed to install PyCluster")
        return False
    
    # Create shortcuts and menu entries
    create_desktop_shortcuts()
    create_start_menu_entries()
    
    # Setup firewall (optional)
    setup_firewall_rules()
    
    # Verify installation
    if not verify_installation():
        print("Installation verification failed")
        return False
    
    print()
    print("Installation completed successfully!")
    print()
    print("Quick Start:")
    print("1. To start a head node:")
    print("   python -m pycluster.cli")
    print()
    print("2. To start a worker node:")
    print("   python -m pycluster.cli worker --scheduler tcp://HEAD_NODE_IP:8786")
    print()
    print("3. Access the dashboard at:")
    print("   http://HEAD_NODE_IP:8787")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

