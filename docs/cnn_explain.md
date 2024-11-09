#### Modified CNN architecture
<p align="center">
  <img src="res/cnn_cam.png"/>
</p>

Two cnn models were trained for experimentation.
```bash
full size: efficientnet_v2_s. features.7 -> [1, 1280, 7, 7]
downscaled: efficientnet_v2_s. features.6.0.block.0 -> [1, 960, 14, 14]
full size: mobilenet_v3_large. features.16 -> [1, 960, 7, 7]
downscaled: mobilenet_v3_large. features.13.block.0 -> [1, 672, 14, 14]
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

Run explainability segmentation evaluation for all models:
```bash
scripts\eval_segmentation.bat
```