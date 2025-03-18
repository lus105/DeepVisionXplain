#!/bin/bash

# Stage 1. Lear outline segmentation train. Hparams search===================================================================================================

#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/Unet +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/DeepLabV3 +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/DeepLabV3Plus +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/Segformer +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers
#python src/train.py experiment=train_seat_seg model.net._target_=src.models.components.segmentation_models.seg_transunet.VisionTransformer + trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers