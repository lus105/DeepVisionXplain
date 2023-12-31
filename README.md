## Project Description

This repository aims to compare CNN (CAM) and ViT (attention rollout) explainability output. 

Main goals:
1. Environment setup, getting used to template style.
2. Implement CAM for CNN (single iteration).
3. Implement attention rollout for ViT (single iteration).
4. Training both models on defect detection datasets.
5. Compare the results based on conventional and custom metrics.
6. Open source repo, paper?

## Installation

#### Conda

```bash
# clone project
git clone https://github.com/lus105/DeepVisionXplain.git
cd DeepVisionXplain

# create conda environment and install dependencies
conda env create -f environment.yaml -n DeepVisionXplain

# activate conda environment
conda activate DeepVisionXplain
```

## How to run

Train model with default configuration (check if environment is properly set up)

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```