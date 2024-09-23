<div align='center'>

# DeepVisionXplain
<img src="res/logo.png" width="100" />

<strong>Versatile model training environment</strong>  
<strong>Used as ViT and CNN explainability experimentation tool</strong>  

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/site)

----
</div>


## Project Description
Neural network training environment (including various MLOps tools) designed to compare the explainability of CNNs (using Class Activation Maps) and ViTs (using attention rollout). Research project paper can be found [here](https://epubl.ktu.edu/object/elaba:198846619/).

<p align="center">
  <img src="res/explainability.png" width="250"/>
</p>

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

## Model Training Environment

<details>
  <summary><font size="5"><b>More information</b></font></summary>

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

#### Development
Linting all files in the project:
```bash
pre-commit run -a
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
</details>

## Model Explainability

<details>
  <summary><font size="5"><b>More information</b></font></summary>

#### Modified CNN architecture
<p align="center">
  <img src="res/cnn_cam.svg" width="450"/>
</p>

#### Modified ViT architecture
<p align="center">
  <img src="res/vit_rollout.svg" width="700"/>
</p>

Train cnn/vit model:
```bash
# train cnn
python src/train.py runs=train_cnn
# train vit
python src/train.py runs=train_vit
```
Train cnn/vit model with hparams search:
```bash
# train cnn
python src/train.py hparams_search=cnn_optuna runs=train_cnn

# train vit
python src/train.py hparams_search=vit_optuna runs=train_vit

# Run explainability segmentation evaluation for all models
scripts\eval_segmentation.bat
```
Two cnn models were trained for experimentation.
```bash
full size: efficientnet_v2_s. features.7 -> [1, 1280, 7, 7]
downscaled: efficientnet_v2_s. features.6.0.block.0 -> [1, 960, 14, 14]
full size: mobilenet_v3_large. features.16 -> [1, 960, 7, 7]
downscaled: mobilenet_v3_large. features.13.block.0 -> [1, 672, 14, 14]
```

#### Resources

Defect detection [datasets](https://drive.google.com/drive/folders/10yYU8yl3um0c1oq6-uVjHp5ORZWXi_tQ?usp=sharing).

Experiment [logs](https://wandb.ai/team_deepvisionxplain?shareProfileType=copy).

Trained [models](https://huggingface.co/DeepVisionXplain).

Research [paper](https://epubl.ktu.edu/object/elaba:198846619/).

</details>

## References
Project inspired by:
* [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
* [jacobgil/vit-explain](https://github.com/jacobgil/vit-explain)
* [rytiss/DL-defect-classification-with-CAM-output](https://github.com/rytisss/DL-defect-classification-with-CAM-output)