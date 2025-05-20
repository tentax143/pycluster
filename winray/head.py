from flask import Flask, jsonify, request, render_template
from datetime import datetime, timedelta
import threading

app = Flask(__name__)

connected_workers = {}

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/register_worker", methods=["POST"])
def register_worker():
    data = request.json
    worker_id = data.get("worker_id")
    if not worker_id:
        return jsonify({"status": "error", "message": "worker_id required"}), 400

    connected_workers[worker_id] = datetime.utcnow()
    return jsonify({"status": "registered"})

@app.route("/dashboard_data")
def dashboard_data():
    threshold = datetime.utcnow() - timedelta(seconds=30)
    # Remove workers not seen in last 30 seconds
    for worker_id, last_seen in list(connected_workers.items()):
        if last_seen < threshold:
            connected_workers.pop(worker_id)

    return jsonify({"connected_workers": len(connected_workers)})

def start_server_thread():
    app.run(host="0.0.0.0", port=5000, threaded=True)

if __name__ == "__main__":
    start_server_thread()
