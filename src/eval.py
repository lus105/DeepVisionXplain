from typing import Any

import hydra
import rootutils
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.utils import (
    RankedLogger,
    instantiate_loggers,
    instantiate_callbacks,
    log_hyperparameters,
    task_wrapper,
    log_gpu_memory_metadata,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): DictConfig configuration composed by Hydra.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: metrics and dict with all instantiated objects.
    """
    assert cfg.model.ckpt_path, "The checkpoint path (cfg.model.ckpt_path) is not set!"

    log_gpu_memory_metadata()

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    trainer.test(model=model, datamodule=datamodule)

    if cfg.get("predict"):
        log.info("Starting predicting!")
        trainer.predict(model=model, datamodule=datamodule)
    else:
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    evaluate(cfg)


if __name__ == "__main__":
    main()
