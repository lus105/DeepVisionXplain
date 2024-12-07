import warnings
import subprocess
from importlib.util import find_spec
from typing import Any, Callable, Optional

import hydra
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from lightning_utilities.core.rank_zero import rank_zero_only
import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
)

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

        Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    Args:
        cfg (DictConfig): A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get('extras'):
        log.warning('Extras config not found! <cfg.extras=null>')
        return

    # disable python warnings
    if cfg.extras.get('ignore_warnings'):
        log.info('Disabling python warnings! <cfg.extras.ignore_warnings=True>')
        warnings.filterwarnings('ignore')

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get('enforce_tags'):
        log.info('Enforcing tags! <cfg.extras.enforce_tags=True>')
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get('print_config'):
        log.info('Printing config tree with Rich! <cfg.extras.print_config=True>')
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    Args:
        task_func (Callable): The task function to be wrapped.

    Returns:
        Callable: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            # apply extra utilities
            extras(cfg)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception('')

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f'Output dir: {cfg.paths.output_dir}')

            # always close wandb run (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    Args:
        metric_dict (Dict[str, Any]): A dict containing metric values.
        metric_name (Optional[str]): If provided, the name of the metric to retrieve.

    Returns:
        Optional[float]: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info('Metric name is None! Skipping metric value retrieval...')
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f'Metric value not found! <metric_name={metric_name}>\n'
            'Make sure metric name logged in LightningModule is correct!\n'
            'Make sure `optimized_metric` name in `hparams_search` config is correct!'
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f'Retrieved metric value! <{metric_name}={metric_value}>')

    return metric_value


def run_sh_command(cmd: Any, allow_fail: bool = True, **kwargs: Any) -> str:
    """Executes a shell command using subprocess and returns the output.

    Args:
        cmd (Any): The shell command to execute. Can be a string or a sequence of program arguments.
        allow_fail (bool, optional): If set to True, the function will return the error output
            if the command fails. If False, it will raise an exception on failure. Defaults to True.
        **kwargs (Any): Additional keyword arguments passed to `subprocess.check_output`.

    Returns:
        str: The output of the command. If `allow_fail` is True and the command fails,
        the output will contain the error message.

    Raises:
        subprocess.SubprocessError: If `allow_fail` is False and the command fails.
    """
    try:
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
            **kwargs,
        )
    except subprocess.SubprocessError as exception:
        if allow_fail:
            output = f'{exception}\n\n{exception.output}'
        else:
            raise
    return f'> {cmd}\n\n{output}\n'


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during
    multirun).
    """
    log.info('Closing loggers...')

    if find_spec('wandb'):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info('Closing wandb!')
            wandb.finish()


def instantiate_callbacks(callbacks_cfg: DictConfig, has_wandb: bool) -> list[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg (DictConfig): A DictConfig object containing callback configurations.
        has_wandb (bool): Whether WandbLogger is available.

    Returns:
        List[Callback]: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning('No callback configs found! Skipping..')
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError('Callbacks config must be a DictConfig!')

    for name, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and '_target_' in cb_conf:
            if 'wandb' in cb_conf._target_.lower():
                if not has_wandb:
                    log.warning(
                        f"Skipping Wandb callback <{name}> ({cb_conf._target_}) since WandbLogger is not found."
                    )
                    continue

            log.info(f'Instantiating callback <{cb_conf._target_}>')
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_cfg (DictConfig): A DictConfig object containing logger configurations.

    Returns:
        List[Logger]: A list of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning('No logger configs found! Skipping...')
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError('Logger config must be a DictConfig!')

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and '_target_' in lg_conf:
            log.info(f'Instantiating logger <{lg_conf._target_}>')
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers. Additionally saves:
        - Number of model parameters

    Args:
        object_dict (Dict[str, Any]): A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict['cfg'])
    model = object_dict['model']
    trainer = object_dict['trainer']

    if not trainer.logger:
        log.warning('Logger not found! Skipping hyperparameter logging...')
        return

    hparams['model'] = cfg['model']

    # save number of model parameters
    hparams['model/params/total'] = sum(p.numel() for p in model.parameters())
    hparams['model/params/trainable'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams['model/params/non_trainable'] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams['data'] = cfg['data']
    hparams['trainer'] = cfg['trainer']

    hparams['callbacks'] = cfg.get('callbacks')
    hparams['extras'] = cfg.get('extras')

    hparams['task_name'] = cfg.get('task_name')
    hparams['tags'] = cfg.get('tags')
    hparams['ckpt_path'] = cfg.get('ckpt_path')
    hparams['seed'] = cfg.get('seed')

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
        div = 1023**3
        total_gb = info.total / div
        free_gb = info.free / div
        used_gb = info.used / div
        log.info(f'GPU memory info: card {i} : total : {total_gb:.2f} GB')
        log.info(f'GPU memory info: card {i} : free  : {free_gb:.2f} GB')
        log.info(f'GPU memory info: card {i} : used  : {used_gb:.2f} GB')
