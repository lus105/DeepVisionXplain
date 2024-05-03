import os
import cv2
import numpy as np
import warnings
import torch
from matplotlib import pyplot as plt
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
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

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def find_file_path(searched_dir: str, extension: str = '.ckpt') -> str:
    """Finds file path in the given directory.

    :param searched_dir: The directory where to search for the file.
    :param extension: The extension of the file.
    :return: The  path to found file.
    """
    for root, dirs, files in os.walk(searched_dir):
        for file in files:
            if file.endswith(extension):
                return os.path.join(root, file)
    return ""

def weight_load(ckpt_path: str, remove_prefix: str = "net.") -> dict:
    checkpoint_path = find_file_path(ckpt_path)
    checkpoint = torch.load(checkpoint_path)
    model_weights = {k[4:]: v for k, v in checkpoint["state_dict"].items() if k.startswith(remove_prefix)}
    
    return model_weights

def save_images(image: torch.Tensor, cam: torch.Tensor, label: torch.Tensor, path: str) -> None:

    # make path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)

    cam = cam.cpu().numpy()

    label = label.cpu().numpy()
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    label = (label * 255).astype(np.uint8)

    # Thresholded cam
    cam_thresholded = np.where(cam > 0.5, 1, 0)
    cam_thresholded = (cam_thresholded * 255).astype(np.uint8)
    cam_thresholded = cv2.cvtColor(cam_thresholded, cv2.COLOR_GRAY2RGB)

    # Normalize CAM for applying colormap
    cam_normalized = cv2.normalize(cam, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Apply the JET colormap
    cam_colored = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)

    alpha = 0.5  # Transparency for the CAM overlay; adjust as needed
    blended_image = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)

    img_concated = cv2.hconcat([image, label, cam_colored, blended_image, cam_thresholded])
    cv2.imwrite(path, img_concated)