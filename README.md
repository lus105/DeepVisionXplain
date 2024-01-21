## Project Description

This repository aims to compare CNN (CAM) and ViT (attention rollout) explainability output. 

## Tasks
In progress:
- [ ] Implement CAM for CNN (single iteration).
- [ ] Image splitting and patching preprocessing step.

To do:
- [ ] Implement attention rollout for ViT (single iteration).
- [ ] Training both models on defect detection datasets.
- [ ] Compare the results based on conventional and custom metrics.
- [ ] Open source repo, paper?

Done:
- [X] Environment setup, getting used to template style.


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