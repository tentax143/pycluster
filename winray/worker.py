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

def register_worker(head_url, worker_id):
    cpu, ram, gpu, disk = get_metrics()
    try:
        res = requests.post(
            f"{head_url}/register",
            json={"id": worker_id, "port": WORKER_PORT, "cpu": cpu, "ram": ram, "gpu": gpu, "disk": disk},
            timeout=5
        )
        if res.status_code == 200:
            print(f"[WORKER] Registered with ID {worker_id}")
            return True
        else:
            print(f"[ERROR] Registration failed: {res.text}")
    except Exception as e:
        print(f"[ERROR] Failed to register: {e}")
    return False

def send_heartbeat(head_url, worker_id):
    cpu, ram, gpu, disk = get_metrics()  # 🟢 Get live usage every heartbeat
    try:
        res = requests.post(
            f"{head_url}/heartbeat",
            json={"id": worker_id, "cpu": cpu, "ram": ram, "gpu": gpu, "disk": disk},  # 🟢 Send updated metrics
            timeout=5
        )
        if res.status_code == 200:
            print(f"[HEARTBEAT] ✅ CPU: {cpu}% | RAM: {ram}% | GPU: {gpu}% | Disk: {disk}%")
        else:
            print("[HEARTBEAT] ❌ Worker not found")
    except Exception as e:
        print(f"[ERROR] Heartbeat failed: {e}")

def run_worker(head_ip):
    head_url = f"http://{head_ip}:5000"
    worker_id = str(uuid.uuid4())

    while not register_worker(head_url, worker_id):
        print("[RETRY] Retrying in 5 seconds...")
        time.sleep(5)

    while True:
        send_heartbeat(head_url, worker_id)
        time.sleep(10)  # You can reduce this to 5 or even 2 seconds for faster updates


if __name__ == "__main__":
    # Example: run_worker("192.168.1.100")
    pass
