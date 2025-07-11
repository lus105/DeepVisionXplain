import os
from pathlib import Path
import subprocess
from threading import Lock

from src.api.training.schemas import (
    TrainingStatusEnum,
    TrainingStartResponse,
    TrainingStopResponse,
    TrainingStatusResponse,
    TrainingConfigsResponse,
    TrainedModelsPathsResponse,
)


class TrainingManager:
    def __init__(self):
        self._process = None
        self._lock = Lock()

    def start_training(self, config: str = 'example.yaml') -> TrainingStartResponse:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return TrainingStartResponse(
                    status='already_running', pid=self._process.pid
                )

            cmd = ['python', 'src/train.py', f'experiment={config}']
            env = os.environ.copy()
            try:
                self._process = subprocess.Popen(cmd, env=env)
            except Exception as e:
                return TrainingStartResponse(status=f'error: {str(e)}', pid=None)

            return TrainingStartResponse(status=TrainingStatusEnum.STARTED, pid=self._process.pid)

    def stop_training(self) -> TrainingStopResponse:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                self._process = None
                return TrainingStopResponse(status=TrainingStatusEnum.STOPPED)
            return TrainingStopResponse(status=TrainingStatusEnum.NOT_RUNNING)

    def get_status(self) -> TrainingStatusResponse:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return TrainingStatusResponse(
                running=running, pid=self._process.pid if running else None
            )

    def list_available_configs(
        self, config_dir: str = 'configs/experiment', config_ext: str = '.yaml'
    ) -> TrainingConfigsResponse:
        config_path = Path(config_dir)
        if not config_path.exists():
            return TrainingConfigsResponse(available_configs=[])

        configs = [
            f.name for f in config_path.iterdir() if f.is_file() and f.suffix == config_ext
        ]
        return TrainingConfigsResponse(available_configs=configs)

    def get_models_path(
        self, config_dir: str = 'logs/train/runs', weight_ext: str = '*.ckpt'
    ) -> TrainedModelsPathsResponse:
        config_path = Path(config_dir)
        if not config_path.exists():
            return TrainedModelsPathsResponse(model_paths=[])

        models_paths = [str(path.resolve()) for path in config_path.rglob(weight_ext)]
        return TrainedModelsPathsResponse(model_paths=models_paths)
