## Project Description

This repository aims to compare CNN (CAM) and ViT (attention rollout) explainability output.

## Tasks

In progress:

- [ ] Implement CAM for CNN (single iteration).
- [ ] Image patching preprocessing step.

To do:

- [ ] Implement attention rollout for ViT (single iteration).
- [ ] Training both models on defect detection datasets.
- [ ] Compare the results based on conventional and custom metrics.
- [ ] Open source repo, paper?

Done:

- [x] Environment setup, getting used to template style.

Considerations:

1. Higher resolution images preprocessing into patches.

- Preprocessing as separate process and caching patches into directory.
- Using patch-based pipelines (https://torchio.readthedocs.io/patches/index.html)

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
