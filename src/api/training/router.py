from fastapi import APIRouter

from src.api.training.service import TrainingManager

from src.api.training.schemas import (
    AvailableConfigsResponse,
    TrainingStartRequest,
    TrainingStatusResponse,
    TrainingStartResponse,
    TrainingStopResponse,
    TrainedModelsPathsResponse,
)

router = APIRouter()
training_manager = TrainingManager()


@router.get('/configs', response_model=AvailableConfigsResponse)
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