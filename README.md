<div align='center'>

# DeepTrainer
<img src="docs/res/logo.png" width="100" />

<strong>Versatile model training environment</strong>  

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/Logging-WandB-89b8cd)](https://wandb.ai/site)

</div>

## Project Description
Neural network training environment (including various MLOps tools).

#### Conda installation
```bash
# clone project
git clone https://github.com/lus105/DeepVisionXplain.git
# change directory
cd DeepVisionXplain
# update conda
conda update -n base conda
# create conda environment and install dependencies
conda env create -f environment.yaml -n DeepVisionXplain
# activate conda environment
conda activate DeepVisionXplain
```
#### Quickstart
Train model with default configuration (check if environment is properly set up):
```bash
# train on CPU (mnist dataset)
python src/train.py trainer=cpu
# train on GPU (mnist dataset)
python src/train.py trainer=gpu
```

## About

#### Environment Description
The setup is designed to streamline experimentation, foster modularity, and simplify tracking and reproducibility:

✅ Minimal boilerplate code (easily add new models, datasets, tasks, experiments, and different accelerator configurations).

✅ Logging experiments to one place for easier comparison of performance metrics.

✅ Hyperparameter search integration.

#### Working principle:

<p align="center">
  <img src="res/principle_diagram.svg" width="700"/>
</p>

*Configuration*

This part of the diagram illustrates how configuration files (train.yaml, eval.yaml, model.yaml, etc.) are used to manage different aspects of the project, such as data preprocessing, model parameters, and training settings.

*Hydra Loader*

The diagram shows how Hydra loads all configuration files and combines them into a single configuration object (DictConfig). This unified configuration object simplifies the management of settings across different modules and aspects of the project, such as data handling, model specifics, callbacks, logging, and the training process.

*Train/test Script*

 This section represents the operational part of the project. Scripts train.py and eval.py are required for training and evaluatging the model. DictConfig: The combined configuration object passed to these scripts, guiding the instantiation of the subsequent components.

  * LightningDataModule: manages data loading and processing specific to training, validation, testing and predicting phases.

  * LightningModule (model): defines the model, including the computation that transforms inputs into outputs, loss computation, and metrics.

  *	Callbacks: provide a way to insert custom logic into the training loop, such as model checkpointing, early stopping, etc.

  * Logger: handles the logging of training, testing, and validation metrics for monitoring progress.

  *	Trainer: the central object in PyTorch Lightning that orchestrates the training process, leveraging all the other components.

  *	The trainer uses the model, data module, logger, and callbacks to execute the training/evaluating process through the trainer.fit/test/predict methods, integrating all the configuration settings specified through Hydra.

#### Workflow steps:
<p align="center">
  <img src="res/workflow_diagram.svg" width="350"/>
</p>

## Docker
Build docker container:
```shell
docker build -t deeptrainer \
--build-arg CUDA_VERSION=12.5.1 \
--build-arg OS_VERSION=22.04 \
--build-arg PYTHON_VERSION=3.11 \
--build-arg USER_ID=$(id -u) \
--build-arg GROUP_ID=$(id -g) \
--build-arg NAME=$(whoami) \
--build-arg WORKDIR_PATH=/test .
```

Run container:

```shell
docker run \
-it \
--rm \
--gpus all \
--name my_deeptrainer_container \
-v host/data:/data \
deeptrainer
```

## Development

Linting all files in the project:
To run Ruff as a linter, try any of the following:

```shell
ruff check                          # Lint all files in the current directory (and any subdirectories).
ruff check path/to/code/            # Lint all files in `/path/to/code` (and any subdirectories).
ruff check path/to/code/*.py        # Lint all `.py` files in `/path/to/code`.
ruff check path/to/code/to/file.py  # Lint `file.py`.
```

Or, to run Ruff as a formatter:

```shell
ruff format                          # Format all files in the current directory (and any subdirectories).
ruff format path/to/code/            # Format all files in `/path/to/code` (and any subdirectories).
ruff format path/to/code/*.py        # Format all `.py` files in `/path/to/code`.
ruff format path/to/code/to/file.py  # Format `file.py`.
```

Tests:
```bash
# run all tests
pytest
# run tests from specific file
pytest tests/test_train.py
# run all tests except the ones marked as slow
pytest -k "not slow"
```

## References

* [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)