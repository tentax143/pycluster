import requests
import time
import uuid

WORKER_ID = str(uuid.uuid4())  # unique ID per worker instance
HEAD_URL = "http://localhost:5000"

def register_worker():
    try:
        resp = requests.post(f"{HEAD_URL}/register_worker", json={"worker_id": WORKER_ID})
        if resp.ok:
            print("Registered worker with head node.")
        else:
            print(f"Failed to register worker: {resp.text}")
    except Exception as e:
        print(f"Failed to register worker: {e}")

def worker_loop():
    while True:
        register_worker()
        print("No tasks available. Waiting...")  # your existing logic can be here
        time.sleep(10)  # send heartbeat every 10 seconds
def start_worker():
    # Your worker logic here
    print("Worker started, connecting to head node at http://localhost:5000")
    # Example placeholder loop to simulate worker behavior
    while True:
        print("No tasks available. Waiting...")
        time.sleep(5)

if __name__ == "__main__":
    print(f"Worker started with ID: {WORKER_ID}")
    worker_loop()
