import os
import subprocess
from threading import Lock

from src.api.training.schemas import (
    TrainingStartResponse,
    TrainingStopResponse,
    TrainingStatusResponse,
    AvailableConfigsResponse,
)


class TrainingManager:
    def __init__(self):
        self._process = None
        self._lock = Lock()

    def start_training(self, config: str = "example.yaml") -> TrainingStartResponse:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return TrainingStartResponse(status="already_running", pid=self._process.pid)

            cmd = ["python", "src/train.py", f"experiment={config}"]
            env = os.environ.copy()
            try:
                self._process = subprocess.Popen(cmd, env=env)
            except Exception as e:
                return TrainingStartResponse(status=f"error: {str(e)}", pid=None)

            return TrainingStartResponse(status="started", pid=self._process.pid)

    def stop_training(self) -> TrainingStopResponse:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                self._process = None
                return TrainingStopResponse(status="stopped")
            return TrainingStopResponse(status="not_running")

    def get_status(self) -> TrainingStatusResponse:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return TrainingStatusResponse(running=running, pid=self._process.pid if running else None)

    def list_available_configs(self, config_dir: str = "configs/experiment") -> AvailableConfigsResponse:
        if not os.path.exists(config_dir):
            return AvailableConfigsResponse(available_configs=[])
        configs = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
        return AvailableConfigsResponse(available_configs=configs)