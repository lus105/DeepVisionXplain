from fastapi import APIRouter, Depends
from typing import Union
from functools import lru_cache

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

@lru_cache()
def get_training_manager() -> TrainingManager:
    return TrainingManager()

@lru_cache()
def get_metrics_tracker() -> MetricsTracker:
    return MetricsTracker()


@router.get('/configs', response_model=TrainingConfigsResponse)
def list_configs(training_manager: TrainingManager = Depends(get_training_manager)):
    """List all available training configurations."""
    return training_manager.list_available_configs()


@router.post('/start', response_model=TrainingStartResponse)
def start(req: TrainingStartRequest, training_manager: TrainingManager = Depends(get_training_manager)):
    """Start a new training session."""
    return training_manager.start_training(req)


@router.post('/stop', response_model=TrainingStopResponse)
def stop(training_manager: TrainingManager = Depends(get_training_manager)):
    """Stop the current training session."""
    return training_manager.stop_training()


@router.get('/status', response_model=TrainingStatusResponse)
def status(training_manager: TrainingManager = Depends(get_training_manager)):
    """Get the current training status."""
    return training_manager.get_status()


@router.get('/trained_models', response_model=TrainedModelsPathsResponse)
def trained_models(training_manager: TrainingManager = Depends(get_training_manager)):
    """Get paths to all trained models."""
    return training_manager.get_models_path()


# Metrics tracking endpoints
@router.get(
    '/metrics/latest', response_model=Union[MetricsResponse, MetricsErrorResponse]
)
def get_latest_metrics(metrics_tracker: MetricsTracker = Depends(get_metrics_tracker)):
    """Get metrics from the most recent training run."""
    return metrics_tracker.get_latest_run_metrics()


@router.get('/metrics/runs', response_model=list[RunInfo])
def list_training_runs(metrics_tracker: MetricsTracker = Depends(get_metrics_tracker)):
    """List all available training runs."""
    return metrics_tracker.list_available_runs()


@router.get(
    '/metrics/runs/{run_id}',
    response_model=Union[MetricsResponse, MetricsErrorResponse],
)
def get_run_metrics(run_id: str, metrics_tracker: MetricsTracker = Depends(get_metrics_tracker)):
    """Get detailed metrics for a specific training run."""
    return metrics_tracker.get_run_metrics(run_id)


@router.get(
    '/metrics/runs/{run_id}/summary',
    response_model=Union[RunSummary, MetricsErrorResponse],
)
def get_run_summary(run_id: str, metrics_tracker: MetricsTracker = Depends(get_metrics_tracker)):
    """Get summary statistics for a specific training run."""
    return metrics_tracker.get_run_summary(run_id)
