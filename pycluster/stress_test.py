from dask.distributed import Client, as_completed
import time

def heavy_cpu_task(n):
    # Simulate heavy CPU work
    count = 0
    for i in range(1, n * 100000):
        count += i % 7
    return f"CPU({n}): Done"

if __name__ == "__main__":
    # Connect to your running cluster
    client = Client("tcp://172.16.71.183:8786")  # Change to your scheduler address if needed

    print("Connected to cluster!")
    print("Dashboard:", client.dashboard_link)

    NUM_TASKS = 200  # Increase for more stress
    print(f"Submitting {NUM_TASKS} heavy CPU tasks...")

    # Submit tasks (randomize the load a bit)
    import random
    futures = [client.submit(heavy_cpu_task, random.randint(300, 800)) for _ in range(NUM_TASKS)]

    # Monitor progress
    completed = 0
    for future in as_completed(futures):
        result = future.result()
        completed += 1
        print(f"Task {completed}/{NUM_TASKS} completed, result: {result}")

    print("All tasks completed!")
    client.close()