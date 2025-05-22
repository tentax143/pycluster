import threading
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskSplitter:
    def __init__(self):
        """
        Manage tasks and distribute them across workers.
        Thread-safe.
        """
        self._lock = threading.Lock()
        self.tasks: List[Any] = []
        self.workers_capacity: Dict[str, int] = {}  # worker_id -> capacity weight
        self.assigned_tasks: Dict[str, List[Any]] = {}  # worker_id -> tasks assigned

    def add_worker(self, worker_id: str, capacity: int = 1):
        """
        Register a worker with given capacity.
        """
        with self._lock:
            logger.info(f"Adding worker {worker_id} with capacity {capacity}")
            self.workers_capacity[worker_id] = max(1, capacity)
            if worker_id not in self.assigned_tasks:
                self.assigned_tasks[worker_id] = []

    def remove_worker(self, worker_id: str):
        """
        Remove a worker and reassign its tasks.
        """
        with self._lock:
            logger.info(f"Removing worker {worker_id}")
            tasks_to_reassign = self.assigned_tasks.pop(worker_id, [])
            self.workers_capacity.pop(worker_id, None)
            self.tasks.extend(tasks_to_reassign)  # re-add to unassigned pool
            self._redistribute_tasks()

    def add_tasks(self, new_tasks: List[Any]):
        """
        Add new tasks to the pool and redistribute.
        """
        with self._lock:
            logger.info(f"Adding {len(new_tasks)} new tasks")
            self.tasks.extend(new_tasks)
            self._redistribute_tasks()

    def update_worker_capacity(self, worker_id: str, capacity: int):
        """
        Update capacity of a worker and redistribute tasks.
        """
        with self._lock:
            if worker_id in self.workers_capacity:
                logger.info(f"Updating capacity for worker {worker_id} to {capacity}")
                self.workers_capacity[worker_id] = max(1, capacity)
                self._redistribute_tasks()
            else:
                logger.warning(f"Worker {worker_id} not found to update capacity")

    def get_tasks_for_worker(self, worker_id: str) -> List[Any]:
        """
        Return tasks assigned to a worker.
        """
        with self._lock:
            return list(self.assigned_tasks.get(worker_id, []))

    def mark_task_done(self, worker_id: str, task: Any):
        """
        Mark a specific task as done by a worker, remove from assigned tasks.
        """
        with self._lock:
            if worker_id in self.assigned_tasks and task in self.assigned_tasks[worker_id]:
                self.assigned_tasks[worker_id].remove(task)
                logger.info(f"Task {task} marked done by {worker_id}")
            else:
                logger.warning(f"Task {task} not found for worker {worker_id}")

    def _redistribute_tasks(self):
        """
        Redistribute unassigned tasks + currently assigned tasks according to capacities.
        """
        # Gather all tasks to assign (including unfinished from assigned tasks)
        all_tasks = self.tasks[:]
        for w, tasks in self.assigned_tasks.items():
            all_tasks.extend(tasks)
            self.assigned_tasks[w] = []

        if not self.workers_capacity:
            logger.warning("No workers to assign tasks to")
            # Keep all tasks unassigned
            self.tasks = all_tasks
            return

        total_capacity = sum(self.workers_capacity.values())
        if total_capacity == 0:
            # Avoid divide by zero
            total_capacity = len(self.workers_capacity)
            for w in self.workers_capacity:
                self.workers_capacity[w] = 1

        task_count = len(all_tasks)
        logger.info(f"Redistributing {task_count} tasks among workers with total capacity {total_capacity}")

        # Assign tasks proportionally by capacity
        start_index = 0
        for worker_id, capacity in self.workers_capacity.items():
            count = int((capacity / total_capacity) * task_count)
            # Ensure last worker gets remaining tasks
            if worker_id == list(self.workers_capacity.keys())[-1]:
                count = task_count - start_index
            assigned = all_tasks[start_index:start_index + count]
            self.assigned_tasks[worker_id] = assigned
            start_index += count

        # Clear unassigned task pool
        self.tasks = []

    def get_status(self):
        """
        Return current distribution status.
        """
        with self._lock:
            return {
                "workers_capacity": dict(self.workers_capacity),
                "assigned_tasks": {k: list(v) for k, v in self.assigned_tasks.items()},
                "unassigned_tasks": list(self.tasks)
            }
