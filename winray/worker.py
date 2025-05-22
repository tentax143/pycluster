import requests
import time
import uuid
import psutil

WORKER_PORT = 9000  # Reserved for future use

def get_metrics():
    return psutil.cpu_percent(interval=1), psutil.virtual_memory().percent

def register_worker(head_url, worker_id):
    cpu, ram = get_metrics()
    try:
        res = requests.post(
            f"{head_url}/register",
            json={"id": worker_id, "port": WORKER_PORT, "cpu": cpu, "ram": ram},
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
    try:
        res = requests.post(f"{head_url}/heartbeat", json={"id": worker_id}, timeout=5)
        if res.status_code == 200:
            print("[HEARTBEAT] ✅")
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
        time.sleep(10)
