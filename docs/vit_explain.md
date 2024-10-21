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
```

Run explainability segmentation evaluation for all models:
```bash
scripts\eval_segmentation.bat
```