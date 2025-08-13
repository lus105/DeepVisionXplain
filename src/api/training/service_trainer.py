import os
import json
import shutil
from pathlib import Path
import subprocess
from threading import Lock

from src.api.training.schemas import (
    TrainingStatusEnum,
    TrainingStartRequest,
    TrainingStatusResponse,
    TrainingConfigsResponse,
    TrainedModelInfo,
    TrainedModelsInfoResponse,
    DatasetInfo,
    AvailableDatasetsResponse,
    DeleteModelResponse,
)


class TrainingManager:
    """
    TrainingManager is responsible for managing the lifecycle of a training process,
    including starting, stopping, and monitoring the status of the process.
    It also provides utilities for listing available training configuration
    files and retrieving paths to trained model checkpoints.
    """

    def __init__(self):
        """
        Initializes the service instance by setting up a process placeholder and
        a thread lock for synchronization.
        """
        self._process = None
        self._lock = Lock()

    def start_training(self, request: TrainingStartRequest) -> TrainingStatusResponse:
        """
        Starts the training process using the specified configuration file and data directories.
        If a training process is already running, returns a response indicating
        that training is already in progress. Otherwise, attempts to start a new
        training subprocess with the given configuration and data paths.
        Args:
            request (TrainingStartRequest): The training request containing config name
                and data directory paths.
        Returns:
            TrainingStatusResponse: An object containing the status of the training.
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return TrainingStatusResponse(status=TrainingStatusEnum.RUNNING)

            cmd = [
                'python',
                'src/train.py',
                f'experiment={request.config_name}',
                f'model_name={request.model_name}',
                f'data.train_data_dir={request.train_data_dir}',
                f'data.test_data_dir={request.test_data_dir}',
                f'data.val_data_dir={request.val_data_dir}',
            ]
            env = os.environ.copy()
            try:
                self._process = subprocess.Popen(cmd, env=env)
            except Exception as e:
                return TrainingStatusResponse(status=f'error: {str(e)}')

            return TrainingStatusResponse(status=TrainingStatusEnum.STARTED)

    def stop_training(self) -> TrainingStatusResponse:
        """
        Stops the ongoing training process if it is currently running.

        Returns:
            TrainingStatusResponse: An object indicating the result of the stop operation.
                - If a training process was running, it is terminated and the status is set to STOPPED.
                - If no training process was running, the status is set to NOT_RUNNING.
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                self._process = None
                return TrainingStatusResponse(status=TrainingStatusEnum.STOPPED)
            return TrainingStatusResponse(status=TrainingStatusEnum.NOT_RUNNING)

    def get_status(self) -> TrainingStatusResponse:
        """
        Retrieve the current status of the training process.

        Returns:
            TrainingStatusResponse: An object containing the running status of the
                training process and its process ID (pid) if it is running.
        """
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return TrainingStatusResponse(
                status=TrainingStatusEnum.RUNNING
                if running
                else TrainingStatusEnum.NOT_RUNNING
            )

    def list_available_configs(
        self, config_dir: str = 'configs/experiment', config_ext: str = '.yaml'
    ) -> TrainingConfigsResponse:
        """
        Lists available configuration files in the specified directory with the given file extension.
        Args:
            config_dir (str): The directory to search for configuration files.
                Defaults to 'configs/experiment'.
            config_ext (str): The file extension to filter configuration files.
                Defaults to '.yaml'.
        Returns:
            TrainingConfigsResponse: An object containing a list of available configuration file names.
                If the specified directory does not exist, returns an empty list of available configurations.
        """
        config_path = Path(config_dir)
        if not config_path.exists():
            return TrainingConfigsResponse(available_configs=[])

        configs = [
            f.name
            for f in config_path.iterdir()
            if f.is_file() and f.suffix == config_ext
        ]
        return TrainingConfigsResponse(available_configs=configs)

    def get_models_info(
        self, log_dir: str = 'logs/train/runs', file_ext: str = '*.json'
    ) -> TrainedModelsInfoResponse:
        """
        Retrieves information about trained models by scanning JSON files in the
        specified log directory.
        Args:
            log_dir (str): The directory path where model JSON files are stored.
                Defaults to 'logs/train/runs'.
            file_ext (str): The file extension pattern to search for JSON files.
                Defaults to '*.json'.
        Returns:
            TrainedModelsInfoResponse: An object containing a list of validated
            model information.
        """
        log_path = Path(log_dir)
        if not log_path.exists():
            return TrainedModelsInfoResponse(models_info=[])

        models_info = []
        json_files = log_path.rglob(file_ext)

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)

                    # Validate the JSON data against the TrainedModelInfo schema
                    validated_model = TrainedModelInfo(**model_data)
                    models_info.append(validated_model)

            except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                # Skip files that can't be read or parsed
                continue
            except Exception:
                # Skip files that don't match the expected schema
                continue

        return TrainedModelsInfoResponse(models_info=models_info)

    def get_datasets(self, data_base_dir: str = 'data') -> AvailableDatasetsResponse:
        """
        Lists available datasets with their actual train/test/val directory paths.
        Each dataset should be a directory containing train/, test/, and optionally val/ subdirectories.

        Args:
            data_base_dir (str): The base directory to search for datasets.
                Defaults to 'data'.

        Returns:
            AvailableDatasetsResponse: An object containing a list of available datasets
                with their actual directory paths.
        """
        base_path = Path(data_base_dir)
        if not base_path.exists():
            return AvailableDatasetsResponse(datasets=[])

        datasets = []

        # Iterate through all directories in the base path
        for dataset_dir in base_path.iterdir():
            if dataset_dir.is_dir():
                # Check for required subdirectories
                train_dir = dataset_dir / 'train'
                test_dir = dataset_dir / 'test'
                val_dir = dataset_dir / 'val'

                has_train = train_dir.exists() and train_dir.is_dir()
                has_test = test_dir.exists() and test_dir.is_dir()
                has_val = val_dir.exists() and val_dir.is_dir()

                # Include dataset if it has at least train and test directories
                if has_train and has_test:
                    dataset_info = DatasetInfo(
                        dataset_name=dataset_dir.name,
                        train_path=str(train_dir.resolve()) if has_train else None,
                        test_path=str(test_dir.resolve()) if has_test else None,
                        val_path=str(val_dir.resolve()) if has_val else None,
                        dataset_base_path=str(dataset_dir.resolve()),
                    )
                    datasets.append(dataset_info)

        return AvailableDatasetsResponse(datasets=datasets)

    def delete_model(
        self, run_id: str, log_dir: str = 'logs/train/runs'
    ) -> DeleteModelResponse:
        """
        Delete a trained model by run ID. This removes the entire run directory
        including all associated files (checkpoints, logs, metrics, etc.).

        Args:
            run_id (str): The run ID (timestamp-based directory name) of the model to delete.
            log_dir (str): The directory path where model runs are stored.
                Defaults to 'logs/train/runs'.

        Returns:
            DeleteModelResponse: A response indicating success or failure of the deletion operation.
        """
        log_path = Path(log_dir)
        run_path = log_path / run_id

        # Check if the run directory exists
        if not run_path.exists() or not run_path.is_dir():
            return DeleteModelResponse(
                success=False, error=f'Model run {run_id} not found'
            )

        try:
            # Remove the entire run directory and all its contents
            shutil.rmtree(run_path)
            return DeleteModelResponse(success=True)
        except PermissionError:
            return DeleteModelResponse(
                success=False,
                error=f'Permission denied: Cannot delete model run {run_id}',
            )
        except Exception as e:
            return DeleteModelResponse(
                success=False, error=f'Failed to delete model run {run_id}: {str(e)}'
            )
