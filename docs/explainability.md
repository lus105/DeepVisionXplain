# CNN/ViT (ante-hoc) explainability

<p align="center">
  <img src="res/explainability.png" width="250"/>
</p>

## Modified ViT architecture
<p align="center">
  <img src="res/vit_rollout.png"/>
</p>

## Modified CNN architecture
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
python src/train.py experiment=train_cnn_multi
# train vit
python src/train.py experiment=train_vit_multi
```

Train cnn/vit model with hparams search:
```bash
# train cnn
python src/train.py hparams_search=cnn_optuna experiment=train_cnn_multi
# train vit
python src/train.py hparams_search=vit_optuna experiment=train_vit_multi
```


## Resources

- [Trained models](https://huggingface.co/DeepVisionXplain/models)
- [Defect detection datasets](https://huggingface.co/DeepVisionXplain/datasets)
- [Experiment logs](https://wandb.ai/team_deepvisionxplain?shareProfileType=copy)
- [Research paper](https://epubl.ktu.edu/object/elaba:198846619/)

## References

- [jacobgil/vit-explain](https://github.com/jacobgil/vit-explain)
- [rytiss/DL-defect-classification-with-CAM-output](https://github.com/rytisss/DL-defect-classification-with-CAM-output)