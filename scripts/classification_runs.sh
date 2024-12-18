#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

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

python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_effnet_v2_s_down model.optimizer.lr=0.000067 data.batch_size=64 logger=many_loggers

python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=cnn_mobnet_v3_large_down model.optimizer.lr=0.00057 data.batch_size=64 logger=many_loggers

python src/train.py trainer.max_epochs=40 experiment=train_seat_cls model=vit_deit_tiny model.optimizer.lr=0.00005 data.batch_size=128 logger=many_loggers