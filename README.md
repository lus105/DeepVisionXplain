<div align='center'>

# DeepVisionXplain
<img src="docs/res/logo_xplain.png" width="100" />

**Neural network training environment with MLOps tools, training API, and model explainability**

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

---
## Features

- **Model Training**: PyTorch Lightning + Hydra configuration system
- **Training API**: FastAPI service for remote training management ([docs](docs/training_api.md))
- **Explainability**: CNN CAM and ViT Attention Rollout ([docs](docs/explainability.md))
- **Hyperparameter Optimization**: Integrated Optuna sweeps
- **MLOps**: W&B integration, checkpoint management, ONNX export

## Quick Start

**Installation:**
```bash
git clone https://github.com/lus105/DeepVisionXplain.git
cd DeepVisionXplain
conda env create -f environment.yaml -n DeepVisionXplain
conda activate DeepVisionXplain
copy .env.example .env # or cp .env.example .env
```

**Train a model:**
```bash
# CPU
python src/train.py trainer=cpu

# GPU
python src/train.py trainer=gpu

# Specific experiment
python src/train.py experiment=experiment_name
```

**Run Training API:**
```bash
# Development
fastapi dev src/api/main.py

# Docker
docker compose up --build

# Docker (Pre-built image)
docker-compose -f docker-compose.prod.yaml up
```

## Documentation

- [Training API Service](docs/training_api.md) - REST API for managing training processes
- [Model Explainability](docs/explainability.md) - CNN/ViT ante-hoc explainability methods

## Resources

- [Research paper](https://ieeexplore.ieee.org/document/10348813)
- [Master's project](https://epubl.ktu.edu/object/elaba:198846619/)

---