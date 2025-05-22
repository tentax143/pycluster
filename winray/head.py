from flask import Flask, request, jsonify, render_template, redirect
import time

app = Flask(__name__, template_folder='templates')

# In-memory worker registry
workers = {}

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
        "last_heartbeat": time.time()
    }
    return jsonify({"status": "registered", "worker_id": worker_id})


@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    data = request.get_json()
    worker_id = data.get("id")
    if worker_id in workers:
        workers[worker_id]["last_heartbeat"] = time.time()
        return jsonify({"status": "heartbeat received"})
    return jsonify({"status": "worker not found"}), 404


@app.route("/dashboard")
def dashboard():
    now = time.time()
    for w in workers.values():
        w["status"] = "Online" if now - w["last_heartbeat"] < 15 else "Offline"
    return render_template("dashboard.html", workers=workers)
@app.route("/")
def home():
    return redirect("/dashboard")

def run_head(host="0.0.0.0", port=5000):
    print(f"[HEAD] Running dashboard on http://{host}:{port}")
    app.run(host=host, port=port)

