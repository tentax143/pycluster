import threading
import time
import typer
from winray.head import start_server_thread
from winray.worker import start_worker

app = typer.Typer()

server_thread = None

@app.command()
def head():
    global server_thread
    if server_thread and server_thread.is_alive():
        typer.echo("Server already running.")
        return
    server_thread = threading.Thread(target=start_server_thread, daemon=True)
    server_thread.start()
    typer.echo("✅ WinRay Head Node started on http://localhost:5000")
    typer.echo("You can now submit tasks or start workers.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        typer.echo("\nStopping WinRay Head Node...")
        typer.echo("Stopped.")

@app.command()
def worker():
    typer.echo("Starting worker...")
    start_worker()

if __name__ == "__main__":
    app()
