<div align='center'>

# DeepVisionXplain
<img src="docs/res/logo.png" width="100" />

<strong>Versatile model training environment</strong>  
<strong>Used as ViT and CNN explainability experimentation tool</strong>  

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/Logging-WandB-89b8cd)](https://wandb.ai/site)

</div>

## Project Description
Neural network training environment (including various MLOps tools) designed to compare the explainability of CNNs (using Class Activation Maps) and ViTs (using attention rollout).

<p align="center">
  <img src="docs/res/explainability.png" width="250"/>
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

## Docs

1. [Model training environment](/docs/environment.md)
2. [CNN explainability](/docs/cnn_explain.md)
3. [ViT explainability](/docs/vit_explain.md)
4. [Research paper](https://epubl.ktu.edu/object/elaba:198846619/)

## Resources

* Defect detection [datasets](https://drive.google.com/drive/folders/10yYU8yl3um0c1oq6-uVjHp5ORZWXi_tQ?usp=sharing).
* Experiment [logs](https://wandb.ai/team_deepvisionxplain?shareProfileType=copy).
* Trained [models](https://huggingface.co/DeepVisionXplain).

## References

* [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
* [jacobgil/vit-explain](https://github.com/jacobgil/vit-explain)
* [rytiss/DL-defect-classification-with-CAM-output](https://github.com/rytisss/DL-defect-classification-with-CAM-output)