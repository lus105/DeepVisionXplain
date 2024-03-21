from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer, Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    instantiate_callbacks,
    log_hyperparameters,
    task_wrapper,
    weight_load
)

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tests given checkpoint on a datamodule testset.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

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

    log.info("Starting testing!")
    model_weights = weight_load(cfg.paths.trained_models + cfg.ckpt_path)
    model.net.load_state_dict(model_weights)
    trainer.test(model=model, datamodule=datamodule)

    metric_dict = trainer.callback_metrics

    return metric_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_segmentation.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for segmentation testing.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    extras(cfg)

    test(cfg)


if __name__ == "__main__":
    main()