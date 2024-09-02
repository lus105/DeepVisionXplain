## Project Description

Neural network training environment (including various MLOps tools) designed to compare the explainability of CNNs (using Class Activation Maps) and ViTs (using attention rollout). [Research project](https://epubl.ktu.edu/object/elaba:198846619/).

<p align="center">
  <img src="res/vit_rollout.png" />
</p>

## Installation
#### Conda

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

## How to run
Train model with default configuration (check if environment is properly set up):

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

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
```

## Training environment


## Experiments/Results

Two cnn models were trained for experimentation.
```bash
full size: efficientnet_v2_s. features.7 -> [1, 1280, 7, 7]

downscaled: efficientnet_v2_s. features.6.0.block.0 -> [1, 960, 14, 14]

full size: mobilenet_v3_large. features.16 -> [1, 960, 7, 7]

downscaled: mobilenet_v3_large. features.13.block.0 -> [1, 672, 14, 14]
```

## Resources

Defect detection [datasets](https://drive.google.com/drive/folders/10yYU8yl3um0c1oq6-uVjHp5ORZWXi_tQ?usp=sharing).

Experiment [logs](https://wandb.ai/team_deepvisionxplain?shareProfileType=copy).

Trained [models](https://huggingface.co/DeepVisionXplain).

## Development

To reformat all files in the project use command:

```bash
pre-commit run -a
```

To run tests:

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```