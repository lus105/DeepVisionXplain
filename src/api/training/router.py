from fastapi import APIRouter
from typing import Union

from src.api.training.service import TrainingManager
from src.api.training.metrics_tracker import MetricsTracker

from src.api.training.schemas import (
    TrainingConfigsResponse,
    TrainingStartRequest,
    TrainingStatusResponse,
    TrainingStartResponse,
    TrainingStopResponse,
    TrainedModelsPathsResponse,
    # Metrics schemas
    MetricsResponse,
    MetricsErrorResponse,
    RunInfo,
    RunSummary,
)

router = APIRouter()
training_manager = TrainingManager()
metrics_tracker = MetricsTracker()


@router.get('/configs', response_model=TrainingConfigsResponse)
def list_configs():
    return training_manager.list_available_configs()


@router.post('/start', response_model=TrainingStartResponse)
def start(req: TrainingStartRequest):
    return training_manager.start_training(req.config_name)


@router.post('/stop', response_model=TrainingStopResponse)
def stop():
    return training_manager.stop_training()


@router.get('/status', response_model=TrainingStatusResponse)
def status():
    return training_manager.get_status()


@router.get('/trained_models', response_model=TrainedModelsPathsResponse)
def trained_models():
    return training_manager.get_models_path()


# Metrics tracking endpoints
@router.get(
    '/metrics/latest', response_model=Union[MetricsResponse, MetricsErrorResponse]
)
def get_latest_metrics():
    """Get metrics from the most recent training run."""
    return metrics_tracker.get_latest_run_metrics()


@router.get('/metrics/runs', response_model=list[RunInfo])
def list_training_runs():
    """List all available training runs."""
    return metrics_tracker.list_available_runs()


@router.get(
    '/metrics/runs/{run_id}',
    response_model=Union[MetricsResponse, MetricsErrorResponse],
)
def get_run_metrics(run_id: str):
    """Get detailed metrics for a specific training run."""
    return metrics_tracker.get_run_metrics(run_id)


@router.get(
    '/metrics/runs/{run_id}/summary',
    response_model=Union[RunSummary, MetricsErrorResponse],
)
def get_run_summary(run_id: str):
    """Get summary statistics for a specific training run."""
    return metrics_tracker.get_run_summary(run_id)
