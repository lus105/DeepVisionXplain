#!/bin/bash

python src/train.py experiment=train_seat_seg trainer.max_epochs=50 logger=many_loggers