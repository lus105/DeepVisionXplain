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
    train_data_dir: str
    test_data_dir: str
    val_data_dir: str


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


# Metrics tracking schemas
class RunInfo(BaseModel):
    """Basic information about a training run"""
    run_id: str
    created_at: str
    modified_at: str
    status: str  # "completed", "in_progress", etc.
    has_metrics: bool
    metrics_file: Optional[str] = None


class MetricsResponse(BaseModel):
    """Response containing training metrics data"""
    run_id: str
    available_columns: list[str]
    total_rows: int
    max_epoch: int
    max_step: int
    data: list[dict]  # Dynamic structure based on CSV columns
    csv_file: str
    last_modified: str


class MetricsErrorResponse(BaseModel):
    """Error response for metrics operations"""
    error: str
    csv_file: Optional[str] = None


class RunSummary(BaseModel):
    """Summary statistics for a training run"""
    run_id: str
    total_epochs: int
    total_steps: int
    best_metrics: dict  # Best values for each metric
    final_metrics: dict  # Final values from last epoch
    available_columns: list[str]
