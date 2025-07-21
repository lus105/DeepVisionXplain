from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.training.service import TrainingManager
from src.api.training.schemas import (
    TrainingStartRequest,
    TrainingStatusEnum,
)


class TestTrainingManager:
    """Test cases for TrainingManager service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TrainingManager()

    @patch('subprocess.Popen')
    def test_start_training_success(self, mock_popen):
        """Test successful training start."""
        # Mock subprocess
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        request = TrainingStartRequest(
            config_name="test_config",
            train_data_dir="data/train",
            test_data_dir="data/test", 
            val_data_dir="data/val"
        )
        
        response = self.manager.start_training(request)
        
        assert response.status == TrainingStatusEnum.STARTED
        assert response.pid == 12345
        mock_popen.assert_called_once()

    @patch('subprocess.Popen')
    def test_start_training_already_running(self, mock_popen):
        """Test starting training when already running."""
        # Setup existing process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process still running
        self.manager._process = mock_process
        
        request = TrainingStartRequest(
            config_name="test_config",
            train_data_dir="data/train",
            test_data_dir="data/test",
            val_data_dir="data/val"
        )
        
        response = self.manager.start_training(request)
        
        assert response.status == "already_running"
        assert response.pid == 12345
        mock_popen.assert_not_called()

    @patch('subprocess.Popen')
    def test_start_training_error(self, mock_popen):
        """Test training start with subprocess error."""
        mock_popen.side_effect = Exception("Failed to start process")
        
        request = TrainingStartRequest(
            config_name="test_config",
            train_data_dir="data/train", 
            test_data_dir="data/test",
            val_data_dir="data/val"
        )
        
        response = self.manager.start_training(request)
        
        assert "error:" in response.status
        assert response.pid is None

    def test_stop_training_running_process(self):
        """Test stopping a running training process."""
        # Setup running process
        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running
        self.manager._process = mock_process
        
        response = self.manager.stop_training()
        
        assert response.status == TrainingStatusEnum.STOPPED
        mock_process.terminate.assert_called_once()
        assert self.manager._process is None

    def test_stop_training_no_process(self):
        """Test stopping when no process is running."""
        response = self.manager.stop_training()
        
        assert response.status == TrainingStatusEnum.NOT_RUNNING

    def test_get_status_running(self):
        """Test status check with running process."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Still running
        self.manager._process = mock_process
        
        response = self.manager.get_status()
        
        assert response.running is True
        assert response.pid == 12345

    def test_get_status_not_running(self):
        """Test status check with no running process."""
        response = self.manager.get_status()
        
        assert response.running is False
        assert response.pid is None

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.iterdir')
    def test_list_available_configs(self, mock_iterdir, mock_exists):
        """Test listing available configuration files."""
        mock_exists.return_value = True
        
        # Mock config files
        mock_config1 = Mock()
        mock_config1.name = "config1.yaml"
        mock_config1.is_file.return_value = True
        mock_config1.suffix = ".yaml"
        
        mock_config2 = Mock()
        mock_config2.name = "config2.yaml"
        mock_config2.is_file.return_value = True
        mock_config2.suffix = ".yaml"
        
        mock_iterdir.return_value = [mock_config1, mock_config2]
        
        response = self.manager.list_available_configs()
        
        assert len(response.available_configs) == 2
        assert "config1.yaml" in response.available_configs
        assert "config2.yaml" in response.available_configs

    @patch('pathlib.Path.exists')
    def test_list_available_configs_no_directory(self, mock_exists):
        """Test listing configs when directory doesn't exist."""
        mock_exists.return_value = False
        
        response = self.manager.list_available_configs()
        
        assert response.available_configs == []

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.rglob')
    def test_get_models_path(self, mock_rglob, mock_exists):
        """Test retrieving trained model paths."""
        mock_exists.return_value = True
        
        # Mock model paths
        mock_path1 = Mock()
        mock_path1.resolve.return_value = Path("/abs/path/model1.onnx")
        mock_path2 = Mock()
        mock_path2.resolve.return_value = Path("/abs/path/model2.onnx")
        
        mock_rglob.return_value = [mock_path1, mock_path2]
        
        response = self.manager.get_models_path()
        
        assert len(response.model_paths) == 2
        assert str(Path("/abs/path/model1.onnx")) in response.model_paths
        assert str(Path("/abs/path/model2.onnx")) in response.model_paths


class TestTrainingAPI:
    """Test cases for the training API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('src.api.training.router.training_manager.list_available_configs')
    def test_list_configs_endpoint(self, mock_list_configs):
        """Test the configs listing endpoint."""
        mock_list_configs.return_value = Mock(available_configs=["config1.yaml", "config2.yaml"])
        
        response = self.client.get("/training/configs")
        
        assert response.status_code == 200
        data = response.json()
        assert "available_configs" in data
        assert len(data["available_configs"]) == 2

    @patch('src.api.training.router.training_manager.start_training')
    def test_start_training_endpoint(self, mock_start_training):
        """Test the training start endpoint."""
        mock_start_training.return_value = Mock(
            status=TrainingStatusEnum.STARTED,
            pid=12345
        )
        
        payload = {
            "config_name": "test_config",
            "train_data_dir": "data/train",
            "test_data_dir": "data/test",
            "val_data_dir": "data/val"
        }
        
        response = self.client.post("/training/start", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["pid"] == 12345

    @patch('src.api.training.router.training_manager.stop_training')
    def test_stop_training_endpoint(self, mock_stop_training):
        """Test the training stop endpoint."""
        mock_stop_training.return_value = Mock(status=TrainingStatusEnum.STOPPED)
        
        response = self.client.post("/training/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"

    @patch('src.api.training.router.training_manager.get_status')
    def test_status_endpoint(self, mock_get_status):
        """Test the training status endpoint."""
        mock_get_status.return_value = Mock(running=True, pid=12345)
        
        response = self.client.get("/training/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is True
        assert data["pid"] == 12345

    @patch('src.api.training.router.training_manager.get_models_path')
    def test_trained_models_endpoint(self, mock_get_models_path):
        """Test the trained models paths endpoint."""
        mock_get_models_path.return_value = Mock(model_paths=["/path/to/model1.onnx", "/path/to/model2.onnx"])
        
        response = self.client.get("/training/trained_models")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_paths" in data
        assert len(data["model_paths"]) == 2

    @patch('src.api.training.router.metrics_tracker.get_latest_run_metrics')
    def test_latest_metrics_endpoint(self, mock_get_latest_metrics):
        """Test the latest metrics endpoint."""
        mock_get_latest_metrics.return_value = Mock(
            run_id="test_run",
            available_columns=["epoch", "loss"],
            total_rows=100,
            max_epoch=10,
            max_step=1000,
            data=[{"epoch": 1, "loss": 0.5}],
            csv_file="path/to/metrics.csv",
            last_modified="2023-01-01T00:00:00"
        )
        
        response = self.client.get("/training/metrics/latest")
        
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "test_run"
        assert "available_columns" in data

    @patch('src.api.training.router.metrics_tracker.list_available_runs')
    def test_list_runs_endpoint(self, mock_list_runs):
        """Test the list training runs endpoint."""
        mock_list_runs.return_value = [
            Mock(run_id="run1",
                 created_at="2025-07-21T10:19:24.784042",
                 modified_at="2025-07-21T10:20:52.114216",
                 status="completed",
                 has_metrics=True,
                 metrics_file="path/to/metrics1.csv"),
            Mock(run_id="run2",
                 created_at="2025-07-21T10:21:00.000000",
                 modified_at="2025-07-21T10:21:00.000000",
                 status="in_progress",
                 has_metrics=False,
                 metrics_file=None
            )
        ]
        
        response = self.client.get("/training/metrics/runs")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_invalid_training_start_request(self):
        """Test training start with invalid request data."""
        payload = {
            "config_name": "test_config"
            # Missing required fields
        }
        
        response = self.client.post("/training/start", json=payload)
        
        assert response.status_code == 422  # Validation error
