#!/bin/bash
# Schedule execution of many runs

# Stage 1. Lear wrinkle classification train. Hparams search===================================================================================================

#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full hparams_search=cnn_optuna logger=many_loggers
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_mobnet_v3_large_full hparams_search=cnn_optuna logger=many_loggers
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=vit_tiny hparams_search=vit_optuna logger=many_loggers


# Stage 2. Lear wrinkle classification train. Modified networks===================================================================================================

#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_down model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_mobnet_v3_large_down model.optimizer.lr=0.00057 data.batch_size=64 logger=many_loggers
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=vit_deit_tiny model.optimizer.lr=0.00005 data.batch_size=128 logger=many_loggers

# Stage 3. Add data to train.============================================================================================================================

#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_mobnet_v3_large_full model.optimizer.lr=0.00057 data.batch_size=64 logger=many_loggers
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=vit_tiny model.optimizer.lr=0.00005 data.batch_size=128 logger=many_loggers

# Stage 4. Optimizers, schedulers and loss functions. Take cnn_effnet_v2_s_full with lr=0.000067 batch_size=64============================================

# Optimizers

# -Schedule-free-
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 +model.optimizer.betas=[0.9,0.98] logger=many_loggers
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 model.optimizer.weight_decay=1e-4 logger=many_loggers
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 model.optimizer.warmup_steps=1000 logger=many_loggers

# -Different-optimizers-
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 model.optimizer._target_=torch.optim.SGD logger=many_loggers
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 model.optimizer._target_=torch.optim.Adagrad logger=many_loggers
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 model.optimizer._target_=torch.optim.Adadelta logger=many_loggers
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 model.optimizer._target_=torch.optim.RMSprop logger=many_loggers
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 model.optimizer._target_=torch.optim.NAdam logger=many_loggers
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 model.optimizer._target_=torch.optim.AdamW logger=many_loggers

# -SGD-
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.001 data.batch_size=64 model.optimizer._target_=torch.optim.SGD logger=many_loggers
# python src/train.py trainer.max_epochs=60 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.001 data.batch_size=64 model.optimizer._target_=torch.optim.SGD logger=many_loggers model.ckpt_path="/media/gpu0/data/lukas/DeepVisionXplain/logs/train/runs/2024-12-19_09-57-16/checkpoints/epoch_039.ckpt"
# python src/train.py trainer.max_epochs=60 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.001 data.batch_size=64 model.optimizer._target_=torch.optim.SGD logger=many_loggers model.ckpt_path="/media/gpu0/data/lukas/DeepVisionXplain/logs/train/runs/2024-12-19_11-49-51/checkpoints/epoch_035.ckpt" +model.optimizer.momentum=0.8
# python src/train.py trainer.max_epochs=60 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.001 data.batch_size=64 model.optimizer._target_=torch.optim.SGD logger=many_loggers model.ckpt_path="/media/gpu0/data/lukas/DeepVisionXplain/trained_models/epoch_002.ckpt" +model.optimizer.momentum=0.8
# python src/train.py trainer.max_epochs=300 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.001 data.batch_size=64 model.optimizer._target_=torch.optim.SGD logger=many_loggers +model.optimizer.momentum=0.9


# Losses
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers model.loss._target_=src.models.components.losses.FocalLoss +model.loss.reduction=mean
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers model.loss._target_=src.models.components.losses.FocalLoss +model.loss.reduction=sum
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers model.loss._target_=src.models.components.losses.FocalLoss +model.loss.reduction=mean +model.loss.alpha=0.5
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers model.loss._target_=src.models.components.losses.FocalLoss +model.loss.reduction=mean +model.loss.gamma=4
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers model.loss._target_=src.models.components.losses.DiceCrossEntropyLoss

# Shedulers
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers model.scheduler._target_=src.models.components.schedulers.WarmupLRScheduler +model.scheduler.init_lr=1e-8 +model.scheduler.peak_lr=0.000067 +model.scheduler.warmup_steps=4
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers model.scheduler._target_=src.models.components.schedulers.WarmupReduceLROnPlateauScheduler +model.scheduler.init_lr=1e-10 +model.scheduler.peak_lr=0.00007 +model.scheduler.warmup_steps=8 +model.scheduler.patience=4

# Stage 5. Evaluate cnn and vit xai segmentation
#export lear_wrinkles_data_path="./data/lear_wrinkles"
#export paths_trained_models="./trained_models"
#-CNN-
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=cnn_effnet_v2_s_full model.ckpt_path=$paths_trained_models/cnn_misty-wildflower-144.ckpt
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=cnn_effnet_v2_s_down model.ckpt_path=$paths_trained_models/cnn_down_polar-field-189.ckpt
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=cnn_mobnet_v3_large_full model.ckpt_path=$paths_trained_models/cnn_helpful-morning-188.ckpt
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=cnn_mobnet_v3_large_down model.ckpt_path=$paths_trained_models/cnn_down_bumbling-fog-252.ckpt

#-VIT-
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_tiny model.ckpt_path=$paths_trained_models/vit_earnest-donkey-167.ckpt model.net.discard_ratio=0.1 model.net.head_fusion="min"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_tiny model.ckpt_path=$paths_trained_models/vit_earnest-donkey-167.ckpt model.net.discard_ratio=0.9 model.net.head_fusion="min"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_tiny model.ckpt_path=$paths_trained_models/vit_earnest-donkey-167.ckpt model.net.discard_ratio=0.1 model.net.head_fusion="mean"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_tiny model.ckpt_path=$paths_trained_models/vit_earnest-donkey-167.ckpt model.net.discard_ratio=0.9 model.net.head_fusion="mean"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_tiny model.ckpt_path=$paths_trained_models/vit_earnest-donkey-167.ckpt model.net.discard_ratio=0.1 model.net.head_fusion="max"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_tiny model.ckpt_path=$paths_trained_models/vit_earnest-donkey-167.ckpt model.net.discard_ratio=0.9 model.net.head_fusion="max"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_deit_tiny model.ckpt_path=$paths_trained_models/vit_different-violet-253.ckpt model.net.discard_ratio=0.1 model.net.head_fusion="min"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_deit_tiny model.ckpt_path=$paths_trained_models/vit_different-violet-253.ckpt model.net.discard_ratio=0.9 model.net.head_fusion="min"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_deit_tiny model.ckpt_path=$paths_trained_models/vit_different-violet-253.ckpt model.net.discard_ratio=0.1 model.net.head_fusion="mean"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_deit_tiny model.ckpt_path=$paths_trained_models/vit_different-violet-253.ckpt model.net.discard_ratio=0.9 model.net.head_fusion="mean"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_deit_tiny model.ckpt_path=$paths_trained_models/vit_different-violet-253.ckpt model.net.discard_ratio=0.1 model.net.head_fusion="max"
#python src/eval.py experiment=eval_seat_xai data.data_dir=$lear_wrinkles_data_path model=vit_deit_tiny model.ckpt_path=$paths_trained_models/vit_different-violet-253.ckpt model.net.discard_ratio=0.9 model.net.head_fusion="max"

# Stage 6. Train with different precision
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers precision.float32_matmul=high
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers precision.float32_matmul=medium
# Delete the sigmoid at the end of the model
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers model.loss._target_=torch.nn.BCEWithLogitsLoss +trainer.precision=16
#python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers model.loss._target_=torch.nn.BCEWithLogitsLoss +trainer.precision="bf16"

# Stage 7. Unclean data training
# Label smoothing added
# python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_full model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers