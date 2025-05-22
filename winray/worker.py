import requests
import time
import uuid
import psutil
import GPUtil

WORKER_PORT = 9000  # Reserved for future use

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        # Average GPU load across GPUs (0-100)
        return round(sum(gpu.load for gpu in gpus) / len(gpus) * 100, 2)
    return 0.0

def get_disk_usage():
    disk = psutil.disk_usage('/')
    return round(disk.percent, 2)

def get_metrics():
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    gpu = get_gpu_usage()
    disk = get_disk_usage()
    return cpu, ram, gpu, disk

def calculate_capacity(cpu, ram, gpu):
    """
    Calculate worker capacity based on inverse load.
    Lower usage = higher capacity.
    Capacity scaled to 1-10.
    """
    load = (cpu + ram + gpu) / 3  # Average load %
    capacity = max(1, int((100 - load) / 10))  # Scale to 1-10
    return capacity

def register_worker(head_url, worker_id):
    cpu, ram, gpu, disk = get_metrics()
    capacity = calculate_capacity(cpu, ram, gpu)
    try:
        res = requests.post(
            f"{head_url}/register",
            json={
                "id": worker_id,
                "port": WORKER_PORT,
                "cpu": cpu,
                "ram": ram,
                "gpu": gpu,
                "disk": disk,
                "capacity": capacity
            },
            timeout=5
        )
        if res.status_code == 200:
            print(f"[WORKER] Registered with ID {worker_id}, capacity {capacity}")
            return True
        else:
            print(f"[ERROR] Registration failed: {res.text}")
    except Exception as e:
        print(f"[ERROR] Failed to register: {e}")
    return False

def send_heartbeat(head_url, worker_id):
    cpu, ram, gpu, disk = get_metrics()
    capacity = calculate_capacity(cpu, ram, gpu)
    try:
        res = requests.post(
            f"{head_url}/heartbeat",
            json={
                "id": worker_id,
                "cpu": cpu,
                "ram": ram,
                "gpu": gpu,
                "disk": disk,
                "capacity": capacity
            },
            timeout=5
        )
        if res.status_code == 200:
            print(f"[HEARTBEAT] ✅ CPU: {cpu}% | RAM: {ram}% | GPU: {gpu}% | Disk: {disk}% | Capacity: {capacity}")
            return True
        else:
            print("[HEARTBEAT] ❌ Worker not found")
            return False
    except Exception as e:
        print(f"[ERROR] Heartbeat failed: {e}")
        return False

def fetch_tasks(head_url, worker_id):
    try:
        res = requests.get(f"{head_url}/tasks/{worker_id}", timeout=5)
        if res.status_code == 200:
            tasks = res.json().get("tasks", [])
            if tasks:
                print(f"[TASKS] Received {len(tasks)} task(s)")
            return tasks
        else:
            print(f"[TASKS] Failed to fetch tasks: {res.status_code}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch tasks: {e}")
    return []

def report_task_completion(head_url, worker_id, task_id, result="success"):
    try:
        res = requests.post(
            f"{head_url}/task_complete",
            json={
                "worker_id": worker_id,
                "task_id": task_id,
                "result": result
            },
            timeout=5
        )
        if res.status_code == 200:
            print(f"[TASK] Reported completion of task {task_id}")
        else:
            print(f"[TASK] Failed to report completion: {res.status_code}")
    except Exception as e:
        print(f"[ERROR] Failed to report task completion: {e}")

def process_task(task):
    # Simulate task processing
    task_id = task.get("id")
    task_data = task.get("data")
    print(f"[PROCESS] Processing task {task_id} with data: {task_data}")
    time.sleep(3)  # Simulate work delay
    return "success"

def run_worker(head_ip):
    head_url = f"http://{head_ip}:5000"
    worker_id = str(uuid.uuid4())

    while not register_worker(head_url, worker_id):
        print("[RETRY] Retrying registration in 5 seconds...")
        time.sleep(5)

    while True:
        # Send heartbeat
        if not send_heartbeat(head_url, worker_id):
            print("[ERROR] Heartbeat failed, attempting to re-register")
            while not register_worker(head_url, worker_id):
                time.sleep(5)

        # Fetch assigned tasks
        tasks = fetch_tasks(head_url, worker_id)
        for task in tasks:
            result = process_task(task)
            report_task_completion(head_url, worker_id, task.get("id"), result)

        # Wait before next cycle
        time.sleep(10)  # adjust interval if needed

if __name__ == "__main__":
    # Usage example: run_worker("192.168.1.100")
    pass
