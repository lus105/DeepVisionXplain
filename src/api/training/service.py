import os
import subprocess
from threading import Lock

# Track the training process
_process = None
_lock = Lock()

def start_training(config: str = "train.yaml") -> dict:
    global _process
    with _lock:
        if _process is not None and _process.poll() is None:
            return {"status": "already_running"}

        cmd = ["python", "src/train.py", f"experiment={config}"]
        env = os.environ.copy()
        _process = subprocess.Popen(cmd, env=env)
        return {"status": "started", "pid": _process.pid}

def stop_training() -> dict:
    global _process
    with _lock:
        if _process is not None and _process.poll() is None:
            _process.terminate()
            _process = None
            return {"status": "stopped"}
        return {"status": "not_running"}

def get_status() -> dict:
    with _lock:
        running = _process is not None and _process.poll() is None
        return {"running": running, "pid": _process.pid if running else None}
    
def list_available_configs(config_dir: str = "configs/experiment") -> dict:
    if not os.path.exists(config_dir):
        return {"available_configs": []}
    configs = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
    return {"available_configs": configs}