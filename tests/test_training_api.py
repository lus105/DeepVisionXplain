from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.training.service_trainer import TrainingManager
from src.api.training.schemas import (
    TrainingStartRequest,
    TrainingStatusEnum,
    TrainedModelInfo,
    DeleteModelResponse,
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
            config_name='test_config',
            model_name='test_model',
            train_data_dir='data/train',
            test_data_dir='data/test',
            val_data_dir='data/val',
        )

        response = self.manager.start_training(request)

        assert response.status == TrainingStatusEnum.STARTED
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
            config_name='test_config',
            model_name='test_model',
            train_data_dir='data/train',
            test_data_dir='data/test',
            val_data_dir='data/val',
        )

        response = self.manager.start_training(request)

        assert response.status == TrainingStatusEnum.RUNNING
        mock_popen.assert_not_called()

    @patch('subprocess.Popen')
    def test_start_training_error(self, mock_popen):
        """Test training start with subprocess error."""
        mock_popen.side_effect = Exception('Failed to start process')

        request = TrainingStartRequest(
            config_name='test_config',
            model_name='test_model',
            train_data_dir='data/train',
            test_data_dir='data/test',
            val_data_dir='data/val',
        )

        response = self.manager.start_training(request)

        assert 'error:' in response.status

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

        assert response.status == TrainingStatusEnum.RUNNING

    def test_get_status_not_running(self):
        """Test status check with no running process."""
        response = self.manager.get_status()

        assert response.status == TrainingStatusEnum.NOT_RUNNING

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.iterdir')
    def test_list_available_configs(self, mock_iterdir, mock_exists):
        """Test listing available configuration files."""
        mock_exists.return_value = True

        # Mock config files
        mock_config1 = Mock()
        mock_config1.name = 'config1.yaml'
        mock_config1.is_file.return_value = True
        mock_config1.suffix = '.yaml'

        mock_config2 = Mock()
        mock_config2.name = 'config2.yaml'
        mock_config2.is_file.return_value = True
        mock_config2.suffix = '.yaml'

        mock_iterdir.return_value = [mock_config1, mock_config2]

        response = self.manager.list_available_configs()

        assert len(response.available_configs) == 2
        assert 'config1.yaml' in response.available_configs
        assert 'config2.yaml' in response.available_configs

    @patch('pathlib.Path.exists')
    def test_list_available_configs_no_directory(self, mock_exists):
        """Test listing configs when directory doesn't exist."""
        mock_exists.return_value = False

        response = self.manager.list_available_configs()

        assert response.available_configs == []

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.rglob')
    @patch('builtins.open')
    @patch('json.load')
    def test_get_models_info(self, mock_json_load, mock_open, mock_rglob, mock_exists):
        """Test retrieving trained model information."""
        mock_exists.return_value = True

        # Mock JSON files
        mock_file1 = Mock()
        mock_file2 = Mock()
        mock_rglob.return_value = [mock_file1, mock_file2]

        # Mock JSON data matching TrainedModelInfo schema
        mock_json_load.side_effect = [
            {
                'run_id': '2025-08-06_11-06-14',
                'model_name': 'classification_model',
                'model_path': 'C:\\path\\to\\model.onnx',
                'dataset_name': 'grain',
                'config_name': 'train_grain',
                'class_names': ['broken', 'healthy'],
                'train_metrics': {'loss': 1.0, 'acc': 0.75},
                'test_metrics': {'loss': 2.5, 'acc': 0.29},
            },
            {
                'run_id': '2025-08-06_12-00-00',
                'model_name': 'another_model',
                'model_path': 'C:\\path\\to\\another_model.onnx',
                'dataset_name': 'test_dataset',
                'config_name': 'test_config',
                'class_names': ['class1', 'class2'],
                'train_metrics': {'loss': 0.8, 'acc': 0.85},
                'test_metrics': {'loss': 1.2, 'acc': 0.82},
            },
        ]

        response = self.manager.get_models_info()

        assert len(response.models_info) == 2
        assert response.models_info[0].run_id == '2025-08-06_11-06-14'
        assert response.models_info[0].model_name == 'classification_model'
        assert response.models_info[1].run_id == '2025-08-06_12-00-00'

    @patch('src.api.training.service_trainer.shutil.rmtree')
    @patch('src.api.training.service_trainer.Path.is_dir')
    @patch('src.api.training.service_trainer.Path.exists')
    def test_delete_model_success(self, mock_exists, mock_is_dir, mock_rmtree):
        """Test successful model deletion."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        response = self.manager.delete_model('2025-08-06_11-06-14')

        assert response.success is True
        assert response.error is None
        mock_rmtree.assert_called_once()

    @patch('src.api.training.service_trainer.Path.exists')
    def test_delete_model_not_found(self, mock_exists):
        """Test deletion of non-existent model."""
        mock_exists.return_value = False

        response = self.manager.delete_model('non-existent-run')

        assert response.success is False
        assert 'not found' in response.error
        assert isinstance(response, DeleteModelResponse)

    @patch('src.api.training.service_trainer.shutil.rmtree')
    @patch('src.api.training.service_trainer.Path.is_dir')
    @patch('src.api.training.service_trainer.Path.exists')
    def test_delete_model_permission_error(self, mock_exists, mock_is_dir, mock_rmtree):
        """Test deletion with permission error."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_rmtree.side_effect = PermissionError('Access denied')

        response = self.manager.delete_model('2025-08-06_11-06-14')

        assert response.success is False
        assert 'Permission denied' in response.error
        assert isinstance(response, DeleteModelResponse)

    @patch('src.api.training.service_trainer.shutil.rmtree')
    @patch('src.api.training.service_trainer.Path.is_dir')
    @patch('src.api.training.service_trainer.Path.exists')
    def test_delete_model_unexpected_error(self, mock_exists, mock_is_dir, mock_rmtree):
        """Test deletion with unexpected error."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_rmtree.side_effect = Exception('Unexpected error')

        response = self.manager.delete_model('2025-08-06_11-06-14')

        assert response.success is False
        assert 'Failed to delete' in response.error
        assert 'Unexpected error' in response.error
        assert isinstance(response, DeleteModelResponse)


class TestTrainingAPI:
    """Test cases for the training API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up after each test."""
        app.dependency_overrides.clear()

    def test_list_configs_endpoint(self):
        """Test the configs listing endpoint."""
        from src.api.training.router import get_training_manager

        # Mock the dependency override
        mock_manager = Mock()
        mock_manager.list_available_configs.return_value = Mock(
            available_configs=['config1.yaml', 'config2.yaml']
        )

        # Override the dependency
        app.dependency_overrides[get_training_manager] = lambda: mock_manager

        try:
            response = self.client.get('/training/configs')

            assert response.status_code == 200
            data = response.json()
            assert 'available_configs' in data
            assert len(data['available_configs']) == 2
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_start_training_endpoint(self):
        """Test the training start endpoint."""
        from src.api.training.router import get_training_manager

        mock_manager = Mock()
        mock_manager.start_training.return_value = Mock(
            status=TrainingStatusEnum.STARTED
        )

        # Override the dependency
        app.dependency_overrides[get_training_manager] = lambda: mock_manager

        payload = {
            'config_name': 'test_config',
            'model_name': 'test_model',
            'train_data_dir': 'data/train',
            'test_data_dir': 'data/test',
            'val_data_dir': 'data/val',
        }

        try:
            response = self.client.post('/training/start', json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'started'
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_stop_training_endpoint(self):
        """Test the training stop endpoint."""
        from src.api.training.router import get_training_manager

        mock_manager = Mock()
        mock_manager.stop_training.return_value = Mock(
            status=TrainingStatusEnum.STOPPED
        )

        # Override the dependency
        app.dependency_overrides[get_training_manager] = lambda: mock_manager

        try:
            response = self.client.post('/training/stop')

            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'stopped'
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_status_endpoint(self):
        """Test the training status endpoint."""
        from src.api.training.router import get_training_manager

        mock_manager = Mock()
        mock_manager.get_status.return_value = Mock(status=TrainingStatusEnum.RUNNING)

        # Override the dependency
        app.dependency_overrides[get_training_manager] = lambda: mock_manager

        try:
            response = self.client.get('/training/status')

            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'running'
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_trained_models_endpoint(self):
        """Test the trained models info endpoint."""
        from src.api.training.router import get_training_manager

        mock_manager = Mock()
        mock_model_info = TrainedModelInfo(
            run_id='2025-08-06_11-06-14',
            model_name='classification_model',
            model_path='C:\\path\\to\\model.onnx',
            dataset_name='grain',
            config_name='train_grain',
            class_names=['broken', 'healthy'],
            train_metrics={'loss': 1.0, 'acc': 0.75},
            test_metrics={'loss': 2.5, 'acc': 0.29},
        )

        mock_manager.get_models_info.return_value = Mock(models_info=[mock_model_info])

        # Override the dependency
        app.dependency_overrides[get_training_manager] = lambda: mock_manager

        try:
            response = self.client.get('/training/trained_models')

            assert response.status_code == 200
            data = response.json()
            assert 'models_info' in data
            assert len(data['models_info']) == 1
            assert data['models_info'][0]['run_id'] == '2025-08-06_11-06-14'
            assert data['models_info'][0]['model_name'] == 'classification_model'
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_latest_metrics_endpoint(self):
        """Test the latest metrics endpoint."""
        from src.api.training.router import get_metrics_tracker

        mock_tracker = Mock()
        mock_tracker.get_latest_run_metrics.return_value = Mock(
            run_id='test_run',
            available_columns=['epoch', 'loss'],
            total_rows=100,
            max_epoch=10,
            max_step=1000,
            data=[{'epoch': 1, 'loss': 0.5}],
            csv_file='path/to/metrics.csv',
            last_modified='2023-01-01T00:00:00',
        )

        # Override the dependency
        app.dependency_overrides[get_metrics_tracker] = lambda: mock_tracker

        try:
            response = self.client.get('/training/metrics/latest')

            assert response.status_code == 200
            data = response.json()
            assert data['run_id'] == 'test_run'
            assert 'available_columns' in data
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_list_runs_endpoint(self):
        """Test the list training runs endpoint."""
        from src.api.training.router import get_metrics_tracker

        mock_tracker = Mock()
        mock_tracker.list_available_runs.return_value = [
            Mock(
                run_id='run1',
                created_at='2025-07-21T10:19:24.784042',
                modified_at='2025-07-21T10:20:52.114216',
                status='completed',
                has_metrics=True,
                metrics_file='path/to/metrics1.csv',
            ),
            Mock(
                run_id='run2',
                created_at='2025-07-21T10:21:00.000000',
                modified_at='2025-07-21T10:21:00.000000',
                status='in_progress',
                has_metrics=False,
                metrics_file=None,
            ),
        ]

        # Override the dependency
        app.dependency_overrides[get_metrics_tracker] = lambda: mock_tracker

        try:
            response = self.client.get('/training/metrics/runs')

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_invalid_training_start_request(self):
        """Test training start with invalid request data."""
        payload = {
            'config_name': 'test_config'
            # Missing required fields
        }

        response = self.client.post('/training/start', json=payload)

        assert response.status_code == 422  # Validation error

    def test_delete_model_endpoint_success(self):
        """Test the model deletion endpoint with successful deletion."""
        from src.api.training.router import get_training_manager

        mock_manager = Mock()
        mock_manager.delete_model.return_value = DeleteModelResponse(
            success=True
        )

        # Override the dependency
        app.dependency_overrides[get_training_manager] = lambda: mock_manager

        try:
            response = self.client.delete('/training/models/test-run-id')

            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            assert data.get('error') is None
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_delete_model_endpoint_not_found(self):
        """Test the model deletion endpoint with model not found."""
        from src.api.training.router import get_training_manager

        mock_manager = Mock()
        mock_manager.delete_model.return_value = DeleteModelResponse(
            success=False,
            error='Model run non-existent-run not found'
        )

        # Override the dependency
        app.dependency_overrides[get_training_manager] = lambda: mock_manager

        try:
            response = self.client.delete('/training/models/non-existent-run')

            assert response.status_code == 200
            data = response.json()
            assert data['success'] is False
            assert 'not found' in data['error']
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_delete_model_endpoint_permission_error(self):
        """Test the model deletion endpoint with permission error."""
        from src.api.training.router import get_training_manager

        mock_manager = Mock()
        mock_manager.delete_model.return_value = DeleteModelResponse(
            success=False,
            error='Permission denied: Cannot delete model run test-run-id'
        )

        # Override the dependency
        app.dependency_overrides[get_training_manager] = lambda: mock_manager

        try:
            response = self.client.delete('/training/models/test-run-id')

            assert response.status_code == 200
            data = response.json()
            assert data['success'] is False
            assert 'Permission denied' in data['error']
        finally:
            # Clean up the override
            app.dependency_overrides.clear()
