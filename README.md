## Project Description

This repository aims to compare CNN (CAM) and ViT (attention rollout) explainability output.

## Tasks

In progress:

- [ ] Implement CAM for CNN (single iteration).
- [ ] Image splitting and patching preprocessing step.
- [ ] Implement attention rollout for ViT (single iteration).

To do:

- [ ] Training both models on defect detection datasets.
- [ ] Compare the results based on conventional and custom metrics.
- [ ] Open source repo, paper?

Done:



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

## Resources

PCB, GID and BSData [datasets](https://drive.google.com/drive/folders/10yYU8yl3um0c1oq6-uVjHp5ORZWXi_tQ?usp=sharing).