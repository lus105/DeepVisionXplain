#!/bin/bash

# Stage 1. Lear outline segmentation train. Hparams search===================================================================================================

#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/Unet +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/UnetPlusPlus +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/DeepLabV3 +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/DeepLabV3Plus +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/Segformer +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/MAnet +model.net.encoder_name=mobilenet_v2 trainer.max_epochs=50 hparams_search=seg_optuna logger=many_loggers

# Stage 2. Lear outline segmentation train. Differnet encoders===================================================================================================

#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/Unet          +model.net.encoder_name=efficientnet-b0 model.optimizer.lr=0.00056 data.batch_size=8 trainer.max_epochs=50 logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/UnetPlusPlus  +model.net.encoder_name=efficientnet-b0 model.optimizer.lr=0.00045 data.batch_size=8 trainer.max_epochs=50 logger=many_loggers

# Stage 3. Lear outline segmentation train. Losses===================================================================================================

#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/UnetPlusPlus +model.net.encoder_name=mobilenet_v2 model.optimizer.lr=0.00045 data.batch_size=8 model.loss._target_=segmentation_models_pytorch.losses.JaccardLoss trainer.max_epochs=50 logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/UnetPlusPlus +model.net.encoder_name=mobilenet_v2 model.optimizer.lr=0.00045 data.batch_size=8 model.loss._target_=segmentation_models_pytorch.losses.FocalLoss   trainer.max_epochs=50 logger=many_loggers
#python src/train.py experiment=train_seat_seg +model.net.model_name=segmentation_models_pytorch/UnetPlusPlus +model.net.encoder_name=mobilenet_v2 model.optimizer.lr=0.00045 data.batch_size=8 model.loss._target_=segmentation_models_pytorch.losses.LovaszLoss  trainer.max_epochs=50 logger=many_loggers