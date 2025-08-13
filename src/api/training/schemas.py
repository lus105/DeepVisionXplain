from typing import Optional
from enum import Enum
from pydantic import BaseModel


class TrainingStatusEnum(str, Enum):
    """Enumeration of possible training statuses."""

    STARTED = 'started'
    RUNNING = 'running'
    STOPPED = 'stopped'
    NOT_RUNNING = 'not_running'


class TrainingConfigsResponse(BaseModel):
    """Response containing available training configurations."""

    available_configs: list[str]


class TrainingStartRequest(BaseModel):
    """Request to start a training session."""

    config_name: str
    model_name: str
    train_data_dir: str
    test_data_dir: str
    val_data_dir: str


class TrainingStatusResponse(BaseModel):
    """Response containing current training status."""

    status: TrainingStatusEnum | str


class TrainedModelInfo(BaseModel):
    """Information about a single trained model."""

    run_id: str
    model_name: str
    model_path: str
    dataset_name: str
    config_name: str
    class_names: list[str]
    train_metrics: dict
    test_metrics: dict


class TrainedModelsInfoResponse(BaseModel):
    """Response containing information about trained models."""

    models_info: list[TrainedModelInfo]


class DatasetInfo(BaseModel):
    """Information about a dataset with actual directory paths"""

    dataset_name: str
    train_path: str | None
    test_path: str | None
    val_path: str | None
    dataset_base_path: str


class AvailableDatasetsResponse(BaseModel):
    """Response containing available datasets with their actual paths"""

    datasets: list[DatasetInfo]


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


class DeleteModelResponse(BaseModel):
    """Response for model deletion operation"""

    success: bool
    error: Optional[str] = None
