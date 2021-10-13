import tensorflow as tf

from baseline.utils import get_logger
from . import scheduler

log = get_logger(__name__)


def create(conf):
    if conf.type == "warmup_piecewise":
        schduler = scheduler.WarmupPiecewiseConstantDecay(**conf["params"])
    else:
        raise AttributeError(f"not support scheduler config: {conf}")
    return schduler
