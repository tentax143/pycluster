import time
import os
from pycluster import ClusterManager
from dask.distributed import Client, get_worker
import numpy as np

def get_worker_info():
    """Get worker details."""
    try:
        worker = get_worker()
        return {
            "name": worker.name,
            "address": worker.address,
            "pid": os.getpid()
        }
    except ValueError:
        return {"name": "local_client", "address": "N/A", "pid": os.getpid()}

def cpu_stress_task(duration_sec=60):
    """CPU intensive task - performs heavy matrix multiplications to stress CPU."""
    worker_info = get_worker_info()
    print(f"[[Stress Task]] Running on {worker_info['name']} ({worker_info['address']}) PID: {worker_info['pid']})")

    size = 1000  # INCREASED SIZE FOR MORE INTENSIVE STRESS
    end_time = time.time() + duration_sec
    iterations = 0

    while time.time() < end_time:
        mat_a = np.random.rand(size, size)
        mat_b = np.random.rand(size, size)
        _ = np.dot(mat_a, mat_b)
        iterations += 1

    return {
        "worker": worker_info,
        "iterations": iterations,
        "message": f"CPU stress test completed on {worker_info['name']} with {iterations} iterations."
    }

if __name__ == "__main__":
    print("\n--- PyCluster FORCE CPU Stress Test ---")
    SCHEDULER_ADDRESS = "tcp://172.16.71.183:8786"  # IMPORTANT: Update this to your head node\'s IP
    STRESS_DURATION = 60  # Duration (seconds) of stress per worker

    client = None
    try:
        cluster_manager = ClusterManager()
        client = Client(SCHEDULER_ADDRESS)
        cluster_manager.client = client

        print(f"Connected to cluster: {client.scheduler.address}")
        worker_info = client.scheduler_info()["workers"]
        print(f"Detected Workers: {len(worker_info)}")

        futures = []
        for worker_addr in worker_info:
            # Submit 1 stress task to each specific worker explicitly
            future = client.submit(cpu_stress_task, STRESS_DURATION, workers=[worker_addr], pure=False)
            futures.append(future)
            print(f"Stress task submitted to worker: {worker_addr}")

        print("Waiting for all stress tasks to complete...")
        results = client.gather(futures)

        print("\n--- CPU Stress Test Summary ---")
        for res in results:
            print(f"Worker: {res['worker']['name']} ({res['worker']['address']})")
            print(f"Iterations: {res['iterations']}")
            print(f"Message: {res['message']}\n")

        print("--- CPU Stress Test Complete ---")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        if client and not client.status == 'closed':
            try:
                client.close()
                print("Dask client closed.")
            except Exception as e:
                print(f"Error closing client: {e}")

