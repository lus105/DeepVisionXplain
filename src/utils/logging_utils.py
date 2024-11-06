from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
)

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers. Additionally saves:
        - Number of model parameters

    Args:
        object_dict (Dict[str, Any]): A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

def log_gpu_memory_metadata() -> None:
    """_Logging GPUs memory metadata (total, free and used) if it's available by
    PYNVML.
    """
    gpus_num = torch.cuda.device_count()
    if gpus_num == 0:
        return
    nvmlInit()
    cards = (nvmlDeviceGetHandleByIndex(num) for num in range(gpus_num))
    for i, card in enumerate(cards):
        info = nvmlDeviceGetMemoryInfo(card)
        div = 1023 ** 3
        total_gb = info.total / div 
        free_gb = info.free / div 
        used_gb = info.used / div 
        log.info(f"GPU memory info: card {i} : total : {total_gb:.2f} GB")
        log.info(f"GPU memory info: card {i} : free  : {free_gb:.2f} GB")
        log.info(f"GPU memory info: card {i} : used  : {used_gb:.2f} GB")