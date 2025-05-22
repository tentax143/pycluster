import time
import socket
import psutil
import GPUtil
from flask import Flask, request, jsonify, render_template, redirect

app = Flask(__name__, template_folder='templates')
workers = {}

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        return round(sum(gpu.load for gpu in gpus) / len(gpus) * 100, 2)
    return 0.0

def get_disk_usage():
    disk = psutil.disk_usage('/')
    return round(disk.percent, 2)

@app.route("/register", methods=["POST"])
def register_worker():
    data = request.get_json()
    ip = request.remote_addr
    worker_id = data.get("id")
    workers[worker_id] = {
        "ip": ip,
        "port": data.get("port"),
        "cpu": data.get("cpu"),
        "ram": data.get("ram"),
        "gpu": data.get("gpu", 0),
        "disk": data.get("disk", 0),
        "last_heartbeat": time.time()
    }
    return jsonify({"status": "registered", "worker_id": worker_id})

@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    data = request.get_json()
    worker_id = data.get("id")
    if worker_id in workers:
        workers[worker_id]["cpu"] = data.get("cpu")
        workers[worker_id]["ram"] = data.get("ram")
        workers[worker_id]["gpu"] = data.get("gpu", 0)
        workers[worker_id]["disk"] = data.get("disk", 0)
        workers[worker_id]["last_heartbeat"] = time.time()
        return jsonify({"status": "heartbeat received"})
    return jsonify({"status": "worker not found"}), 404

@app.route("/metrics")
def metrics():
    now = time.time()
    for w in workers.values():
        w["status"] = "Online" if now - w["last_heartbeat"] < 15 else "Offline"
        w["last_heartbeat_str"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(w["last_heartbeat"]))

    head_info = {
        "name": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()),
        "cpu": psutil.cpu_percent(interval=0.5),
        "ram": psutil.virtual_memory().percent,
        "gpu": get_gpu_usage(),
        "disk": get_disk_usage()
    }

    return jsonify({
        "head_info": head_info,
        "workers": workers
    })

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/metrics_page")
def metrics_page():
    # This will serve your metrics.html file from the templates folder
    return render_template("metrics.html")

@app.route("/")
def home():
    return redirect("/dashboard")

def run_head(host="0.0.0.0", port=5000):
    print(f"[HEAD] Running dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
    run_head()
