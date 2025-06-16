"""
Dashboard management for PyCluster
"""

import webbrowser
import logging
from typing import Optional, Dict, Any
import requests
import time

logger = logging.getLogger(__name__)


class DashboardManager:
    """
    Manager for accessing and interacting with the Dask dashboard.
    """
    
    def __init__(self, dashboard_url: str):
        """
        Initialize the dashboard manager.
        
        Args:
            dashboard_url: URL of the Dask dashboard
        """
        self.dashboard_url = dashboard_url.rstrip('/')
        self.base_url = self.dashboard_url
        
    def is_accessible(self) -> bool:
        """
        Check if the dashboard is accessible.
        
        Returns:
            True if dashboard is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Dashboard not accessible: {str(e)}")
            return False
    
    def open_in_browser(self) -> bool:
        """
        Open the dashboard in the default web browser.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            webbrowser.open(f"{self.base_url}/status")
            logger.info(f"Dashboard opened in browser: {self.base_url}/status")
            return True
        except Exception as e:
            logger.error(f"Failed to open dashboard in browser: {str(e)}")
            return False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get cluster status information from the dashboard API.
        
        Returns:
            Dictionary with cluster status
        """
        try:
            response = requests.get(f"{self.base_url}/json/identity.json", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to get cluster status: {str(e)}")
            return {"error": str(e)}
    
    def get_worker_info(self) -> Dict[str, Any]:
        """
        Get information about workers from the dashboard API.
        
        Returns:
            Dictionary with worker information
        """
        try:
            response = requests.get(f"{self.base_url}/json/workers.json", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to get worker info: {str(e)}")
            return {"error": str(e)}
    
    def get_task_stream(self) -> Dict[str, Any]:
        """
        Get task stream information from the dashboard API.
        
        Returns:
            Dictionary with task stream data
        """
        try:
            response = requests.get(f"{self.base_url}/json/task-stream.json", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to get task stream: {str(e)}")
            return {"error": str(e)}
    
    def wait_for_dashboard(self, timeout: int = 30) -> bool:
        """
        Wait for the dashboard to become accessible.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if dashboard becomes accessible, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_accessible():
                return True
            time.sleep(1)
        return False
    
    def get_dashboard_urls(self) -> Dict[str, str]:
        """
        Get URLs for different dashboard pages.
        
        Returns:
            Dictionary with dashboard page URLs
        """
        return {
            "status": f"{self.base_url}/status",
            "workers": f"{self.base_url}/workers",
            "tasks": f"{self.base_url}/tasks",
            "system": f"{self.base_url}/system",
            "profile": f"{self.base_url}/profile",
            "graph": f"{self.base_url}/graph",
            "individual-plots": f"{self.base_url}/individual-plots"
        }
    
    def generate_dashboard_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive dashboard report.
        
        Returns:
            Dictionary with comprehensive cluster information
        """
        report = {
            "dashboard_url": self.dashboard_url,
            "accessible": self.is_accessible(),
            "timestamp": time.time()
        }
        
        if report["accessible"]:
            report["cluster_status"] = self.get_cluster_status()
            report["worker_info"] = self.get_worker_info()
            report["dashboard_urls"] = self.get_dashboard_urls()
        
        return report

