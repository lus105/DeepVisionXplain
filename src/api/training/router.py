from fastapi import APIRouter

from src.api.training.service import TrainingManager

from src.api.training.schemas import (
    AvailableConfigsResponse,
    StartTrainingRequest,
    TrainingStatusResponse,
    TrainingStartResponse,
    TrainingStopResponse,
)

router = APIRouter()
training_manager = TrainingManager()


@router.get('/configs', response_model=AvailableConfigsResponse)
def list_configs():
    return training_manager.list_available_configs()


@router.post('/start', response_model=TrainingStartResponse)
def start(req: StartTrainingRequest):
    return training_manager.start_training(req.config_name)


@router.post('/stop', response_model=TrainingStopResponse)
def stop():
    return training_manager.stop_training()


@router.get('/status', response_model=TrainingStatusResponse)
def status():
    return training_manager.get_status()
