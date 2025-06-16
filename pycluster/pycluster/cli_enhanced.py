"""
Enhanced command line interface for PyCluster with Windows support
"""

import argparse
import sys
import os
import logging
import time
import socket
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='PyCluster - Distributed Computing with LLM Support')
    
    # Common arguments
    parser.add_argument('--cluster-name', default='pycluster', help='Name of the cluster')
    parser.add_argument('--host', default='0.0.0.0', help='Host address to bind to')
    parser.add_argument('--scheduler-port', type=int, default=8786, help='Scheduler port')
    parser.add_argument('--dashboard-port', type=int, default=8787, help='Dashboard port')
    parser.add_argument('--local-workers', type=int, default=2, help='Number of local workers')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--diagnose', action='store_true', help='Run Windows diagnostics')
    parser.add_argument('--config', help='Path to configuration file')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Head node command
    head_parser = subparsers.add_parser('head', help='Start head node')
    head_parser.add_argument('--no-workers', action='store_true', help='Start without local workers')
    
    # Worker command
    worker_parser = subparsers.add_parser('worker', help='Start worker node')
    worker_parser.add_argument('--scheduler', required=True, help='Scheduler address to connect to')
    worker_parser.add_argument('--nthreads', type=int, default=None, help='Number of threads per worker')
    worker_parser.add_argument('--memory-limit', default='auto', help='Memory limit per worker')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Run diagnostics if requested
    if args.diagnose:
        run_diagnostics()
        return
    
    # Apply Windows optimizations
    if sys.platform.startswith('win'):
        try:
            from .windows_fixes import optimize_windows_performance, diagnose_windows_issues
            optimize_windows_performance()
            
            # Quick diagnosis
            diagnosis = diagnose_windows_issues()
            if diagnosis["issues"]:
                logger.warning("Windows issues detected:")
                for issue in diagnosis["issues"]:
                    logger.warning(f"  - {issue}")
                logger.info("Run with --diagnose for detailed information")
        except ImportError:
            logger.warning("Windows fixes not available")
    
    try:
        if args.command == 'worker' or (args.command is None and hasattr(args, 'scheduler')):
            start_worker_node(args)
        else:
            start_head_node(args)
    except KeyboardInterrupt:
        logger.info("Shutting down PyCluster...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start PyCluster: {e}")
        
        # Provide helpful error messages for common issues
        if "Timed out trying to connect" in str(e):
            logger.error("Connection timeout - this usually indicates:")
            logger.error("  1. Firewall blocking the ports (8786, 8787)")
            logger.error("  2. Another service using the same ports")
            logger.error("  3. Network connectivity issues")
            if sys.platform.startswith('win'):
                logger.error("  4. Windows Defender or antivirus blocking the connection")
                logger.error("Try running as Administrator or use --diagnose for more info")
        
        sys.exit(1)

def start_head_node(args):
    """Start a head node"""
    from .node import HeadNode
    
    logger.info(f"Starting PyCluster head node: {args.cluster_name}")
    
    # Determine number of workers
    n_workers = 0 if (hasattr(args, 'no_workers') and args.no_workers) else args.local_workers
    
    try:
        with HeadNode(args.cluster_name, host=args.host) as head:
            result = head.start(
                n_local_workers=n_workers
            )
            
            if result["status"] == "success":
                logger.info("✓ Head node started successfully!")
                logger.info(f"  Cluster: {result['cluster_name']}")
                logger.info(f"  Scheduler: {result['scheduler_address']}")
                logger.info(f"  Dashboard: {result['dashboard_address']}")
                logger.info(f"  Workers: {n_workers}")
                
                if sys.platform.startswith('win'):
                    logger.info("\nWindows users:")
                    logger.info("  - Dashboard may take a moment to load")
                    logger.info("  - If connection fails, check Windows Firewall")
                    logger.info("  - Run as Administrator if needed")
                
                logger.info("\nPress Ctrl+C to stop the cluster")
                
                # Keep running until interrupted
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Stopping cluster...")
            else:
                logger.error(f"✗ Failed to start head node: {result.get('message', 'Unknown error')}")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"✗ Failed to start head node: {e}")
        sys.exit(1)

def start_worker_node(args):
    """Start a worker node"""
    from .node import WorkerNode
    
    logger.info(f"Starting PyCluster worker node")
    logger.info(f"Connecting to scheduler: {args.scheduler}")
    
    try:
        with WorkerNode(scheduler_address=args.scheduler) as worker:
            result = worker.start(
                n_workers=1,
                threads_per_worker=args.nthreads,
                memory_limit=args.memory_limit
            )
            
            if result["status"] == "success":
                logger.info("✓ Worker node connected successfully!")
                logger.info(f"  Scheduler: {args.scheduler}")
                logger.info(f"  Workers: {result.get('n_workers', 1)}")
                
                logger.info("\nPress Ctrl+C to stop the worker")
                
                # Keep running until interrupted
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Stopping worker...")
            else:
                logger.error(f"✗ Failed to connect worker: {result.get('message', 'Unknown error')}")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"✗ Failed to start worker node: {e}")
        sys.exit(1)

def run_diagnostics():
    """Run comprehensive diagnostics"""
    print("PyCluster Diagnostics")
    print("=" * 50)
    
    try:
        if sys.platform.startswith('win'):
            from .windows_fixes import diagnose_windows_issues, get_windows_network_info, check_windows_firewall
            
            # Run full diagnosis
            diagnosis = diagnose_windows_issues()
            
            print(f"Platform: {diagnosis['platform']}")
            print(f"Python: {diagnosis['python_version']}")
            print()
            
            if diagnosis["issues"]:
                print("Issues Found:")
                for i, issue in enumerate(diagnosis["issues"], 1):
                    print(f"  {i}. ❌ {issue}")
                print()
                
                print("Recommendations:")
                for i, rec in enumerate(diagnosis["recommendations"], 1):
                    print(f"  {i}. 💡 {rec}")
                print()
            else:
                print("✅ No issues detected!")
                print()
            
            # Network information
            print("Network Configuration:")
            network_info = get_windows_network_info()
            if network_info["status"] == "ok":
                print(f"  Hostname: {network_info['hostname']}")
                print(f"  Local IP: {network_info['local_ip']}")
            else:
                print(f"  ❌ Error: {network_info.get('message', 'Unknown error')}")
            print()
            
            # Firewall check
            print("Firewall Status:")
            firewall_status = check_windows_firewall([8786, 8787, 5000])
            if firewall_status["status"] == "ok":
                print("  ✅ No obvious firewall issues")
            elif firewall_status["status"] == "ports_blocked":
                print(f"  ⚠️  Ports may be blocked: {firewall_status['blocked_ports']}")
                print(f"  💡 {firewall_status['recommendation']}")
            else:
                print(f"  ❌ {firewall_status['message']}")
            
        else:
            print("Platform: Non-Windows")
            print("✅ Windows-specific diagnostics not needed")
            
    except ImportError as e:
        print(f"❌ Could not import diagnostics module: {e}")
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")
    
    print()
    print("For more help, visit: https://github.com/pycluster/pycluster")

# Legacy CLI functions for backward compatibility
def head_node_cli():
    """Legacy head node CLI entry point"""
    parser = argparse.ArgumentParser(description="Start a PyCluster head node")
    parser.add_argument("--cluster-name", default="pycluster", 
                       help="Name of the cluster")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host address to bind to")
    parser.add_argument("--scheduler-port", type=int, default=8786,
                       help="Port for the scheduler")
    parser.add_argument("--dashboard-port", type=int, default=8787,
                       help="Port for the dashboard")
    parser.add_argument("--local-workers", type=int, default=2,
                       help="Number of local workers to start")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert to new format and call main function
    sys.argv = [
        sys.argv[0],
        '--cluster-name', args.cluster_name,
        '--host', args.host,
        '--scheduler-port', str(args.scheduler_port),
        '--dashboard-port', str(args.dashboard_port),
        '--local-workers', str(args.local_workers)
    ]
    
    if args.verbose:
        sys.argv.append('--verbose')
    
    main()

def worker_node_cli():
    """Legacy worker node CLI entry point"""
    parser = argparse.ArgumentParser(description="Start a PyCluster worker node")
    parser.add_argument("--scheduler", required=True,
                       help="Scheduler address to connect to")
    parser.add_argument("--nthreads", type=int, default=None,
                       help="Number of threads per worker")
    parser.add_argument("--memory-limit", default="auto",
                       help="Memory limit per worker")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert to new format and call main function
    sys.argv = [
        sys.argv[0],
        'worker',
        '--scheduler', args.scheduler,
        '--memory-limit', args.memory_limit
    ]
    
    if args.nthreads:
        sys.argv.extend(['--nthreads', str(args.nthreads)])
    
    if args.verbose:
        sys.argv.append('--verbose')
    
    main()

if __name__ == "__main__":
    main()

