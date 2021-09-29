import logging
import sys
import os
import warnings
import random
from typing import List, Sequence

import tensorflow as tf
import numpy as np
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

from . import strategy
from . import callback

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)
    logger = logging.getLogger(name)
    
    logger.setLevel(level)

    # tf.get_logger().setLevel(logging.WARNING)

    return logger

def set_environment(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.base.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.base.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)

def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "architecture",
        "dataset",
        "optimizer",
        "scheduler",
        "loss",
        "callbacks",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)

def set_seed(seed):
    seed = int(seed, 0)

    random.seed(conf.base.seed)
    np.random.seed(conf.base.seed)
    tf.random.set_seed(conf.base.seed)
