from typing import Optional
from pydantic import BaseModel


class AvailableConfigsResponse(BaseModel):
    available_configs: list[str]


class StartTrainingRequest(BaseModel):
    config_name: str


class TrainingStatusResponse(BaseModel):
    running: bool
    pid: Optional[int] = None


class TrainingStartResponse(BaseModel):
    status: str
    pid: Optional[int] = None


class TrainingStopResponse(BaseModel):
    status: str