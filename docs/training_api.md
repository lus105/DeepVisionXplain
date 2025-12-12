# Training API service

A FastAPI-based service for managing deep learning model training processes.

## Overview

The Training API provides endpoints to start, stop, monitor training processes, and retrieve training metrics. It uses a background subprocess approach to run training scripts while maintaining API responsiveness. The API includes comprehensive metrics tracking.

Architecture diagram:
<p align="center">
  <img src="res/architecture_diagram.png"/>
</p>

## Run service (using Docker)
```bash
cd DeepVisionXplain
# Create .env file (from .env.example) and specify required paths
copy .env.example .env # or cp .env.example .env
# Build and run the Docker container
docker compose up --build
# Open documentation
http://127.0.0.1:8000/docs
```

## Run service (from source)
```bash
cd DeepVisionXplain
# Activate conda environment
conda activate DeepVisionXplain
# Development mode
fastapi dev src/api/main.py
# Production mode
fastapi run src/api/main.py
# Open documentation
http://127.0.0.1:8000/docs
```

## Health Check
The API includes a simple health check endpoint:
```http
GET /ping
```
**Response:**
```json
{
  "status": "ok"
}
```

## Core Endpoints

### Training Management

#### Start Training
```http
POST /training/start
```
Starts a new training process with specified configuration.

**Request Body:**
```json
{
  "config_name": "experiment_name",
  "model_name": "default", # (model output name)
  "train_data_dir": "data/train",
  "test_data_dir": "data/test",
  "val_data_dir": "data/val"
}
```

**Response:**
```json
{
  "status": "started"
}
```

#### Stop Training
```http
POST /training/stop
```
Stops the currently running training process.

**Response:**
```json
{
  "status": "stopped"
}
```

#### Get Status
```http
GET /training/status
```
Returns current training process status.

**Response:**
```json
{
  "status": "running"
}
```

### Configuration & Models

#### List Configurations
```http
GET /training/configs
```
Lists available training configuration files from the `configs/experiment/` directory.

**Response:**
```json
{
  "available_configs": ["train_cnn_multi.yaml", "train_vit_multi.yaml", "example.yaml"]
}
```

#### Get Trained Models
```http
GET /training/models
```
Returns metadata about trained models (searches for `*.json` files in `logs/train/runs/`).

**Response:**
```json
{
  "models_info": [
    {
      "run_id": "2025-08-05_16-10-59",
      "model_name": "model",
      "model_path": "/path/to/model.onnx",
      "dataset_name": "MNIST",
      "config_name": "train_cnn_multi",
      "class_names": ["class1", "class2"],
      "train_metrics": {"train/loss": 0.1, "train/acc": 0.95},
      "test_metrics": {"test/loss": 0.12, "test/acc": 0.93}
    }
  ]
}
```

#### Delete Trained Model
```http
DELETE /training/models/{run_id}
```
Deletes a trained model by run ID. This removes the entire run directory including all associated files (checkpoints, logs, metrics, etc.).

**Parameters:**
- `run_id` - The timestamp-based directory name (e.g., "2025-08-13_15-09-42")

**Response:**
```json
{
  "success": true,
  "error": null
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Model run {run_id} not found"
}
```

**Error Response (Permission Denied):**
```json
{
  "success": false,
  "error": "Permission denied: Cannot delete model run {run_id}"
}
```

#### Get Available Datasets
```http
GET /training/datasets
```
Returns available datasets with their actual directory paths. Scans the `data/` directory for folders containing `train/` and `test/` subdirectories.

**Response:**
```json
{
  "datasets": [
    {
      "dataset_name": "MNIST",
      "train_path": "/absolute/path/to/data/MNIST/train",
      "test_path": "/absolute/path/to/data/MNIST/test", 
      "val_path": "/absolute/path/to/data/MNIST/val",
      "dataset_base_path": "/absolute/path/to/data/MNIST"
    }
  ]
}
```

### Metrics Tracking

The API provides comprehensive metrics tracking functionality that works with PyTorch Lightning CSV log files.

#### Get Latest Metrics
```http
GET /training/metrics/latest
```
Returns metrics from the most recent training run.

**Response:**
```json
{
  "run_id": "run_2024_01_15_14_30_45",
  "available_columns": ["epoch", "step", "train_loss", "val_loss", "val_acc"],
  "total_rows": 100,
  "max_epoch": 99,
  "max_step": 999,
  "data": [
    {
      "epoch": 0,
      "step": 0,
      "train_loss": 2.3,
      "val_loss": 2.1,
      "val_acc": 0.1
    }
  ],
  "csv_file": "/path/to/logs/train/runs/run_id/csv/version_0/metrics.csv",
  "last_modified": "2024-01-15T14:30:45"
}
```

**Error Response:**
```json
{
  "error": "No training runs found"
}
```

#### List Training Runs
```http
GET /training/metrics/runs
```
Lists all available training runs with metadata.

**Response:**
```json
[
  {
    "run_id": "run_2024_01_15_14_30_45",
    "created_at": "2024-01-15T14:30:45",
    "modified_at": "2024-01-15T16:45:30",
    "status": "completed",
    "has_metrics": true,
    "metrics_file": "/path/to/metrics.csv"
  }
]
```

#### Get Run Metrics
```http
GET /training/metrics/runs/{run_id}
```
Returns detailed metrics for a specific training run.

**Response:** Same format as latest metrics endpoint.

**Error Response:**
```json
{
  "error": "Run {run_id} not found"
}
```

#### Get Run Summary
```http
GET /training/metrics/runs/{run_id}/summary
```
Returns summary statistics for a specific training run.

**Response:**
```json
{
  "run_id": "run_2024_01_15_14_30_45",
  "total_epochs": 99,
  "total_steps": 999,
  "best_metrics": {
    "best_train_loss": 0.05,
    "best_val_loss": 0.08,
    "best_val_acc": 0.95
  },
  "final_metrics": {
    "epoch": 99,
    "step": 999,
    "train_loss": 0.06,
    "val_loss": 0.09,
    "val_acc": 0.94
  },
  "available_columns": ["epoch", "step", "train_loss", "val_loss", "val_acc"]
}
```

## Status Values

- `started` - Training process has been initiated
- `running` - Training is currently in progress  
- `stopped` - Training has been stopped
- `not_running` - No training process is active
- `error: {message}` - An error occurred during training

## Configuration

- Training configurations are stored in `configs/experiment/` (`.yaml` files, referenced without extension)
- Model outputs are saved to `logs/train/runs/{timestamp}/`
- Model metadata files (`classification_model.json`) are searched in `logs/train/runs/`
- Datasets are discovered in `data/` directory (requires `train/` and `test/` subdirectories)
- Metrics are automatically parsed from PyTorch Lightning CSV logs in `logs/train/runs/{timestamp}/csv/version_0/metrics.csv`


## Notes

- Only one training process can run at a time
- Training configurations are stored in `configs/experiment/`
- Model outputs are saved to `logs/train/runs/`
- All endpoints return JSON responses with appropriate HTTP status codes
- The API uses dependency injection for service management
- Metrics tracking supports PyTorch Lightning CSV log formats with automatic pattern detection
- Dataset discovery requires directories with both `train/` and `test/` subdirectories (optional `val/`)
