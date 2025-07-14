from typing import Optional
from enum import Enum
from pydantic import BaseModel


class TrainingStatusEnum(str, Enum):
    STARTED = 'started'
    RUNNING = 'running'
    STOPPED = 'stopped'
    NOT_RUNNING = 'not_running'
    ERROR = 'error'


class TrainingConfigsResponse(BaseModel):
    available_configs: list[str]


class TrainingStartRequest(BaseModel):
    config_name: str


class TrainingStartResponse(BaseModel):
    status: TrainingStatusEnum | str
    pid: Optional[int] = None


class TrainingStatusResponse(BaseModel):
    running: bool
    pid: Optional[int] = None


class TrainingStopResponse(BaseModel):
    status: TrainingStatusEnum


class TrainedModelsPathsResponse(BaseModel):
    model_paths: list[str]
