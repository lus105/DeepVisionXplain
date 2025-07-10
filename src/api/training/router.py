import os

from fastapi import APIRouter

from src.api.training.service import (
    start_training,
    stop_training,
    get_status,
    list_available_configs,
)
from src.api.training.schemas import (
    AvailableConfigsResponse,
    StartTrainingRequest,
    TrainingStatusResponse,
    TrainingStartResponse,
    TrainingStopResponse,
)

router = APIRouter()

@router.get("/configs", response_model=AvailableConfigsResponse)
def list_configs():
    return list_available_configs()

@router.post("/start", response_model=TrainingStartResponse)
def start(req: StartTrainingRequest):
    return start_training(req.config_name)

@router.post("/stop", response_model=TrainingStopResponse)
def stop():
    return stop_training()

@router.get("/status", response_model=TrainingStatusResponse)
def status():
    return get_status()