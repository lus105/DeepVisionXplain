import os
from typing import Any, Optional
from dotenv import load_dotenv

import hydra
import lightning.pytorch as L
import rootutils
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=['.git', 'pyproject.toml'], pythonpath=True)

from src.utils import (
    RankedLogger,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    log_gpu_memory_metadata,
    save_model_metadata,
    is_running_in_docker,
)
from src.models.components.utils import export_model_to_onnx

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: metrics and dict with all instantiated objects.
    """
    log_gpu_memory_metadata()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get('seed'):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f'Instantiating datamodule <{cfg.data._target_}>')
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f'Instantiating model <{cfg.model._target_}>')
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info('Instantiating loggers...')
    loggers: list[Logger] = instantiate_loggers(cfg.get('logger'))

    has_wandb = any(isinstance(logger, WandbLogger) for logger in loggers)

    log.info('Instantiating callbacks...')
    callbacks: list[Callback] = instantiate_callbacks(
        cfg.get('callbacks'), has_wandb=has_wandb
    )

    log.info(f'Instantiating trainer <{cfg.trainer._target_}>')
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    object_dict = {
        'cfg': cfg,
        'datamodule': datamodule,
        'model': model,
        'callbacks': callbacks,
        'logger': loggers,
        'trainer': trainer,
    }

    if loggers:
        log.info('Logging hyperparameters!')
        log_hyperparameters(object_dict)

    if cfg.get('train'):
        log.info('Starting training!')
        trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    if cfg.get('test'):
        log.info('Starting testing!')
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == '':
            log.warning('Best ckpt not found! Using current weights for testing...')
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f'Best ckpt path: {ckpt_path}')

    test_metrics = trainer.callback_metrics

    if cfg.get('export_to_onnx'):
        onnx_path = trainer.checkpoint_callback.best_model_path.replace(
            '.ckpt', '.onnx'
        )
        image_size = cfg.get('data').get('image_size')
        channels = cfg.get('data').get('channels')

        export_model_to_onnx(
            model=model.net,
            onnx_path=onnx_path,
            input_shape=(1, channels, image_size[0], image_size[1]),
        )
        log.info(f'Model exported to {onnx_path}')

        host_log_dir = os.getenv('host_log_dir')
        container_log_dir = os.getenv('container_log_dir')

        if is_running_in_docker() and host_log_dir and container_log_dir:
            host_onnx_path = onnx_path.replace(container_log_dir, host_log_dir)
        else:
            host_onnx_path = onnx_path

        save_model_metadata(
            model_path=onnx_path,
            host_model_path=host_onnx_path,
            dataset_name=datamodule.dataset_name,
            class_names=datamodule.class_names,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )
        log.info('Model metadata saved!')

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base='1.3', config_path='../configs', config_name='train.yaml')
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: optimized metric value.
    """
    # load environment variables from .env file
    load_dotenv()

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get('optimized_metric')
    )

    # return optimized metric
    return metric_value


if __name__ == '__main__':
    main()
