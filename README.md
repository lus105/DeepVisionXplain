## Project Description

This repository aims to compare CNN (CAM) and ViT (attention rollout) explainability output.
<p align="center">
  <img src="res/vit_rollout.png" />
</p>

## Installation

#### Conda

```bash
# clone project
git clone https://github.com/lus105/DeepVisionXplain.git
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
python src/train.py runs=cnn_train

# train vit
python src/train.py runs=vit_train
```

Train cnn/vit model with hparams search:
```bash
# train cnn
python src/train.py hparams_search=cnn_optuna runs=cnn_train

# train vit
python src/train.py hparams_search=vit_optuna runs=vit_train
```

## Experiments

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