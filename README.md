<div align='center'>

# DeepVisionXplain
<img src="docs/res/logo_xplain.png" width="100" />

<strong>Model training environment</strong>  

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

## Project Description
Neural network training environment (including various MLOps tools) with training API service and model explainability tools.

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

#### Other sections

[Training API service](docs/training_api.md)

[CNN/ViT (ante-hoc) explainability](docs/explainability.md)


## References

[lus105/DeepTrainer](https://github.com/lus105/DeepTrainer)