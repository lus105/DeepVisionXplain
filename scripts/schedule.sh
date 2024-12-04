#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Stage 1. Lear wrinkle classification train===================================================================================================
python src/train.py trainer.max_epochs=20 experiment=train_seat_cls model=cnn_effnet_v2_s_full hparams_search=cnn_optuna logger=many_loggers

python src/train.py trainer.max_epochs=20 experiment=train_seat_cls model=cnn_mobnet_v3_large_full hparams_search=cnn_optuna logger=many_loggers

python src/train.py trainer.max_epochs=20 experiment=train_seat_cls model=vit_tiny hparams_search=vit_optuna logger=many_loggers

python src/train.py trainer.max_epochs=20 experiment=train_seat_cls model=vit_deit_tiny hparams_search=vit_optuna logger=many_loggers
