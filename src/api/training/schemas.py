from typing import Optional
from pydantic import BaseModel


class AvailableConfigsResponse(BaseModel):
    available_configs: list[str]


class TrainingStartRequest(BaseModel):
    config_name: str


class TrainingStartResponse(BaseModel):
    status: str
    pid: Optional[int] = None


class TrainingStatusResponse(BaseModel):
    running: bool
    pid: Optional[int] = None


class TrainingStopResponse(BaseModel):
    status: str


class TrainedModelsPathsResponse(BaseModel):
    model_paths: list[str]