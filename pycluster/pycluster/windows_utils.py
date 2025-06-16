"""
Windows-specific utilities for PyCluster
"""

import os
import sys
import subprocess
import platform
import socket
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class WindowsClusterManager:
    """
    Windows-specific cluster management utilities.
    """
    
    def __init__(self):
        self.is_windows = platform.system() == 'Windows'
        self.config_dir = self._get_config_dir()
        self.ensure_config_dir()
    
    def _get_config_dir(self) -> Path:
        """Get the configuration directory for PyCluster."""
        if self.is_windows:
            # Use AppData on Windows
            appdata = os.environ.get('APPDATA', os.path.expanduser('~'))
            return Path(appdata) / 'PyCluster'
        else:
            # Use .config on Unix-like systems
            return Path.home() / '.config' / 'pycluster'
    
    def ensure_config_dir(self):
        """Ensure the configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_cluster_config(self, config: Dict[str, Any], name: str = 'default'):
        """Save cluster configuration to file."""
        config_file = self.config_dir / f'{name}.json'
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Cluster config saved to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def load_cluster_config(self, name: str = 'default') -> Optional[Dict[str, Any]]:
        """Load cluster configuration from file."""
        config_file = self.config_dir / f'{name}.json'
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
        return None
    
    def list_cluster_configs(self) -> List[str]:
        """List available cluster configurations."""
        try:
            configs = []
            for file in self.config_dir.glob('*.json'):
                configs.append(file.stem)
            return configs
        except Exception as e:
            logger.error(f"Failed to list configs: {e}")
            return []
    
    def get_network_interfaces(self) -> List[Dict[str, str]]:
        """Get available network interfaces."""
        interfaces = []
        
        try:
            if self.is_windows:
                # Use ipconfig on Windows
                result = subprocess.run(['ipconfig'], capture_output=True, text=True)
                # Parse ipconfig output (simplified)
                lines = result.stdout.split('\n')
                current_adapter = None
                
                for line in lines:
                    line = line.strip()
                    if 'adapter' in line.lower() and ':' in line:
                        current_adapter = line.split(':')[0].strip()
                    elif 'IPv4 Address' in line and current_adapter:
                        ip = line.split(':')[-1].strip()
                        interfaces.append({
                            'name': current_adapter,
                            'ip': ip,
                            'type': 'IPv4'
                        })
            else:
                # Use hostname -I on Unix-like systems
                result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
                ips = result.stdout.strip().split()
                for i, ip in enumerate(ips):
                    interfaces.append({
                        'name': f'Interface {i+1}',
                        'ip': ip,
                        'type': 'IPv4'
                    })
        
        except Exception as e:
            logger.warning(f"Failed to get network interfaces: {e}")
            # Fallback to socket method
            try:
                hostname = socket.gethostname()
                ip = socket.gethostbyname(hostname)
                interfaces.append({
                    'name': 'Default',
                    'ip': ip,
                    'type': 'IPv4'
                })
            except Exception as e2:
                logger.error(f"Fallback network detection failed: {e2}")
        
        return interfaces
    
    def check_port_availability(self, port: int, host: str = 'localhost') -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False
    
    def find_available_port(self, start_port: int = 8786, max_attempts: int = 100) -> Optional[int]:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            if self.check_port_availability(port):
                return port
        return None
    
    def create_windows_service_script(self, 
                                    service_name: str,
                                    python_path: str,
                                    script_path: str,
                                    args: List[str] = None) -> str:
        """Create a Windows service script for PyCluster components."""
        args = args or []
        args_str = ' '.join(args)
        
        script_content = f'''@echo off
REM PyCluster {service_name} Service Script
REM Generated automatically by PyCluster

echo Starting {service_name}...
cd /d "{os.path.dirname(script_path)}"
"{python_path}" "{script_path}" {args_str}

if errorlevel 1 (
    echo Error: {service_name} failed to start
    pause
    exit /b 1
)

echo {service_name} started successfully
'''
        
        script_file = self.config_dir / f'{service_name}.bat'
        try:
            with open(script_file, 'w') as f:
                f.write(script_content)
            logger.info(f"Service script created: {script_file}")
            return str(script_file)
        except Exception as e:
            logger.error(f"Failed to create service script: {e}")
            return ""
    
    def create_cluster_startup_script(self, config: Dict[str, Any]) -> str:
        """Create a startup script for the entire cluster."""
        script_content = f'''@echo off
REM PyCluster Startup Script
REM Configuration: {config.get('cluster_name', 'default')}

echo Starting PyCluster...
echo Configuration: {config.get('cluster_name', 'default')}
echo.

REM Start head node
if "{config.get('node_type', 'head')}"=="head" (
    echo Starting head node...
    python -m pycluster.cli --cluster-name "{config.get('cluster_name', 'pycluster')}" --host "{config.get('host', '0.0.0.0')}" --scheduler-port {config.get('scheduler_port', 8786)} --dashboard-port {config.get('dashboard_port', 8787)} --local-workers {config.get('local_workers', 0)}
) else (
    echo Starting worker node...
    python -m pycluster.cli worker --scheduler "{config.get('scheduler_address', 'tcp://localhost:8786')}" --workers {config.get('n_workers', 1)}
)

if errorlevel 1 (
    echo Error: PyCluster failed to start
    pause
    exit /b 1
)

echo PyCluster started successfully
pause
'''
        
        script_file = self.config_dir / f"start_{config.get('cluster_name', 'default')}.bat"
        try:
            with open(script_file, 'w') as f:
                f.write(script_content)
            logger.info(f"Startup script created: {script_file}")
            return str(script_file)
        except Exception as e:
            logger.error(f"Failed to create startup script: {e}")
            return ""
    
    def install_windows_firewall_rules(self, ports: List[int]) -> bool:
        """Install Windows firewall rules for PyCluster ports."""
        if not self.is_windows:
            logger.warning("Firewall rules only supported on Windows")
            return False
        
        try:
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
                    logger.info(f"Firewall rule added for port {port}")
                else:
                    logger.warning(f"Failed to add firewall rule for port {port}: {result.stderr}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to install firewall rules: {e}")
            return False
    
    def generate_cluster_info_file(self, cluster_config: Dict[str, Any]) -> str:
        """Generate a cluster information file for easy sharing."""
        info_content = f'''PyCluster Connection Information
=====================================

Cluster Name: {cluster_config.get('cluster_name', 'N/A')}
Scheduler Address: {cluster_config.get('scheduler_address', 'N/A')}
Dashboard URL: {cluster_config.get('dashboard_url', 'N/A')}
Host IP: {cluster_config.get('host_ip', 'N/A')}

To connect workers to this cluster:
1. Install PyCluster on the worker machine
2. Run: pycluster-worker --scheduler {cluster_config.get('scheduler_address', 'tcp://HOST:8786')}

Or use the Python API:
from pycluster import WorkerNode
worker = WorkerNode(scheduler_address="{cluster_config.get('scheduler_address', 'tcp://HOST:8786')}")
worker.start()

Generated on: {os.environ.get('COMPUTERNAME', socket.gethostname())}
'''
        
        info_file = self.config_dir / f"cluster_info_{cluster_config.get('cluster_name', 'default')}.txt"
        try:
            with open(info_file, 'w') as f:
                f.write(info_content)
            logger.info(f"Cluster info file created: {info_file}")
            return str(info_file)
        except Exception as e:
            logger.error(f"Failed to create cluster info file: {e}")
            return ""
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for cluster planning."""
        import psutil
        
        try:
            return {
                'platform': platform.platform(),
                'system': platform.system(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'hostname': socket.gethostname(),
                'network_interfaces': self.get_network_interfaces(),
                'python_version': sys.version,
                'config_dir': str(self.config_dir)
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}

