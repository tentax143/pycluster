"""
Cluster management API routes for PyCluster
"""

import os
import sys
import json
import logging
from flask import Blueprint, request, jsonify
from datetime import datetime

# Add the pycluster package to the path
sys.path.insert(0, '/home/ubuntu/pycluster')

try:
    from pycluster import ClusterManager, HeadNode, WorkerNode, DashboardManager
except ImportError as e:
    logging.warning(f"PyCluster not available: {e}")
    ClusterManager = None
    HeadNode = None
    WorkerNode = None
    DashboardManager = None

cluster_bp = Blueprint('cluster', __name__)

# Global cluster manager instance
cluster_manager = None
head_node = None
worker_node = None

@cluster_bp.route('/cluster/status', methods=['GET'])
def get_cluster_status():
    """Get the current cluster status."""
    try:
        if cluster_manager is None:
            return jsonify({
                'status': 'disconnected',
                'message': 'No cluster connection'
            })
        
        status = cluster_manager.get_cluster_info()
        return jsonify(status)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cluster_bp.route('/cluster/start-head', methods=['POST'])
def start_head_node():
    """Start a head node."""
    global cluster_manager, head_node
    
    try:
        data = request.get_json() or {}
        cluster_name = data.get('cluster_name', 'pycluster-api')
        host = data.get('host', '0.0.0.0')
        scheduler_port = data.get('scheduler_port', 8786)
        dashboard_port = data.get('dashboard_port', 8787)
        local_workers = data.get('local_workers', 0)
        
        if HeadNode is None:
            return jsonify({
                'status': 'error',
                'message': 'PyCluster not available'
            }), 500
        
        head_node = HeadNode(
            cluster_name=cluster_name,
            host=host,
            scheduler_port=scheduler_port,
            dashboard_port=dashboard_port
        )
        
        result = head_node.start(n_local_workers=local_workers)
        
        if result['status'] == 'success':
            cluster_manager = head_node.cluster_manager
            conn_info = head_node.get_connection_info()
            result.update(conn_info)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cluster_bp.route('/cluster/start-worker', methods=['POST'])
def start_worker_node():
    """Start worker node(s)."""
    global cluster_manager, worker_node
    
    try:
        data = request.get_json() or {}
        scheduler_address = data.get('scheduler_address')
        worker_name = data.get('worker_name')
        n_workers = data.get('n_workers', 1)
        threads_per_worker = data.get('threads_per_worker')
        memory_limit = data.get('memory_limit', 'auto')
        
        if not scheduler_address:
            return jsonify({
                'status': 'error',
                'message': 'scheduler_address is required'
            }), 400
        
        if WorkerNode is None:
            return jsonify({
                'status': 'error',
                'message': 'PyCluster not available'
            }), 500
        
        worker_node = WorkerNode(
            scheduler_address=scheduler_address,
            worker_name=worker_name
        )
        
        result = worker_node.start(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit
        )
        
        if result['status'] == 'success':
            cluster_manager = worker_node.cluster_manager
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cluster_bp.route('/cluster/connect', methods=['POST'])
def connect_to_cluster():
    """Connect to an existing cluster."""
    global cluster_manager
    
    try:
        data = request.get_json() or {}
        scheduler_address = data.get('scheduler_address')
        
        if not scheduler_address:
            return jsonify({
                'status': 'error',
                'message': 'scheduler_address is required'
            }), 400
        
        if ClusterManager is None:
            return jsonify({
                'status': 'error',
                'message': 'PyCluster not available'
            }), 500
        
        cluster_manager = ClusterManager()
        cluster_manager.scheduler_address = scheduler_address
        
        # Try to connect
        from dask.distributed import Client
        cluster_manager.client = Client(scheduler_address)
        
        status = cluster_manager.get_cluster_info()
        return jsonify(status)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cluster_bp.route('/cluster/shutdown', methods=['POST'])
def shutdown_cluster():
    """Shutdown the cluster."""
    global cluster_manager, head_node, worker_node
    
    try:
        if head_node:
            head_node.shutdown()
            head_node = None
        
        if worker_node:
            worker_node.shutdown()
            worker_node = None
        
        if cluster_manager:
            cluster_manager.shutdown()
            cluster_manager = None
        
        return jsonify({
            'status': 'success',
            'message': 'Cluster shutdown completed'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cluster_bp.route('/cluster/submit-task', methods=['POST'])
def submit_task():
    """Submit a task to the cluster."""
    try:
        if cluster_manager is None:
            return jsonify({
                'status': 'error',
                'message': 'No cluster connection'
            }), 400
        
        data = request.get_json() or {}
        task_type = data.get('task_type', 'test')
        
        # Simple test task
        def test_task(x):
            import time
            time.sleep(0.1)
            return x * x
        
        if task_type == 'test':
            # Submit a batch of test tasks
            futures = []
            for i in range(10):
                future = cluster_manager.submit_task(test_task, i)
                futures.append(future)
            
            # Wait for results (with timeout)
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=5)
                    results.append(result)
                except Exception as e:
                    results.append(f"Error: {str(e)}")
            
            return jsonify({
                'status': 'success',
                'task_type': task_type,
                'results': results
            })
        
        return jsonify({
            'status': 'error',
            'message': f'Unknown task type: {task_type}'
        }), 400
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cluster_bp.route('/cluster/dashboard-info', methods=['GET'])
def get_dashboard_info():
    """Get dashboard information."""
    try:
        if head_node:
            conn_info = head_node.get_connection_info()
            dashboard_url = conn_info.get('dashboard_url')
            
            if dashboard_url and DashboardManager:
                dashboard = DashboardManager(dashboard_url)
                report = dashboard.generate_dashboard_report()
                return jsonify(report)
        
        return jsonify({
            'status': 'no_dashboard',
            'message': 'No dashboard available'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cluster_bp.route('/cluster/workers', methods=['GET'])
def get_workers():
    """Get detailed worker information."""
    try:
        if cluster_manager is None:
            return jsonify({
                'status': 'error',
                'message': 'No cluster connection'
            })
        
        cluster_info = cluster_manager.get_cluster_info()
        workers = cluster_info.get('workers', [])
        
        # Enhance worker data with mock performance metrics
        import random
        for worker in workers:
            worker['cpu_usage'] = random.randint(20, 80)
            worker['memory_usage'] = random.randint(30, 90)
            worker['network_io'] = random.randint(10, 50)
            worker['last_seen'] = datetime.now().isoformat()
        
        return jsonify({
            'status': 'success',
            'workers': workers,
            'total_workers': len(workers)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cluster_bp.route('/cluster/metrics', methods=['GET'])
def get_metrics():
    """Get cluster performance metrics."""
    try:
        # Generate mock metrics data
        import random
        from datetime import datetime, timedelta
        
        now = datetime.now()
        metrics = []
        
        for i in range(10):
            timestamp = now - timedelta(minutes=i*5)
            metrics.append({
                'timestamp': timestamp.isoformat(),
                'cpu_usage': random.randint(30, 80),
                'memory_usage': random.randint(40, 85),
                'network_io': random.randint(15, 60),
                'tasks_completed': random.randint(50, 150),
                'tasks_pending': random.randint(5, 30),
                'tasks_failed': random.randint(0, 5)
            })
        
        return jsonify({
            'status': 'success',
            'metrics': list(reversed(metrics))
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cluster_bp.route('/cluster/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pycluster_available': ClusterManager is not None
    })

