# Training API service

A FastAPI-based service for managing deep learning model training process.

## Overview

The Training API provides endpoints to start, stop, monitor training processes, and retrieve training metrics. It uses a background subprocess approach to run training scripts while maintaining API responsiveness.

## Run service
```bash
cd DeepVisionXplain
# Development mode
fastapi dev src/api/main.py
# Production mode
fastapi run src/api/main.py
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
  "config_name": "experiment_config.yaml",
  "train_data_dir": "data/train",
  "test_data_dir": "data/test", 
  "val_data_dir": "data/val"
}
```

**Response:**
```json
{
  "status": "started",
  "pid": 12345
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
  "running": true,
  "pid": 12345
}
```

### Configuration & Models

#### List Configurations
```http
GET /training/configs
```
Lists available training configuration files.

**Response:**
```json
{
  "available_configs": ["config1.yaml", "config2.yaml"]
}
```

#### Get Trained Models
```http
GET /training/trained_models
```
Returns paths to trained model files.

**Response:**
```json
{
  "model_paths": ["/path/to/model1.onnx", "/path/to/model2.onnx"]
}
```

### Metrics Tracking

#### Get Latest Metrics
```http
GET /training/metrics/latest
```
Returns metrics from the most recent training run.

#### List Training Runs
```http
GET /training/metrics/runs
```
Lists all available training runs with metadata.

#### Get Run Metrics
```http
GET /training/metrics/runs/{run_id}
```
Returns detailed metrics for a specific training run.

#### Get Run Summary
```http
GET /training/metrics/runs/{run_id}/summary
```
Returns summary statistics for a specific training run.

## Status Values

- `started` - Training process has been initiated
- `running` - Training is currently in progress
- `stopped` - Training has been stopped
- `not_running` - No training process is active
- `error: {message}` - An error occurred during training

## Usage Examples

### Start a training session:
```bash
curl -X POST "http://localhost:8000/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my_experiment.yaml",
    "train_data_dir": "data/train",
    "test_data_dir": "data/test",
    "val_data_dir": "data/val"
  }'
```

### Check training status:
```bash
curl "http://localhost:8000/training/status"
```

### Stop training:
```bash
curl -X POST "http://localhost:8000/training/stop"
```

## Notes

- Only one training process can run at a time
- Training configurations are stored in `configs/experiment/`
- Model outputs are saved to `logs/train/runs/`
- All endpoints return JSON responses with appropriate HTTP status codes
