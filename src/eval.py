import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

torch.set_float32_matmul_precision("medium")

log = RankedLogger(__name__, rank_zero_only=True)


def _prepare_wandb_env(cfg: DictConfig) -> None:
    """Ensure wandb uses writable project-local dirs on Windows/restricted envs."""
    logger_cfg = cfg.get("logger")
    if not logger_cfg:
        return

    has_wandb = False
    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            target = str(lg_conf.get("_target_", "")).lower()
            if "wandb" in target:
                has_wandb = True
                break
    if not has_wandb:
        return

    output_dir = Path(str(cfg.paths.output_dir))
    tmp_dir = output_dir / "tmp"
    wandb_dir = output_dir / "wandb"
    wandb_cache_dir = output_dir / "wandb-cache"
    wandb_config_dir = output_dir / "wandb-config"
    for d in (tmp_dir, wandb_dir, wandb_cache_dir, wandb_config_dir):
        d.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache_dir))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_config_dir))
    os.environ.setdefault("WANDB_START_METHOD", "thread")


def _call_trainer_method(method, **kwargs):
    """Call a Trainer method and pass `weights_only=False` when supported."""
    if "weights_only" in inspect.signature(method).parameters:
        kwargs["weights_only"] = False
    return method(**kwargs)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

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
    _call_trainer_method(
        trainer.test,
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.ckpt_path,
    )

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    _prepare_wandb_env(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
