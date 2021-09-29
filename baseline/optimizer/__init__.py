import tensorflow as tf
import tensorflow_addons as tfa

from baseline.utils import get_logger

log = get_logger(__name__)


def create(config, model=None):
    opt_type = config["type"].lower()

    if opt_type == "sgd":
        log.info(f"[optimizer] create {opt_type}")
        return tf.keras.optimizers.SGD(**config["params"])
    elif opt_type == "sgdw":
        log.info(f"[optimizer] create {opt_type}")
        return tfa.optimizers.SGDW(**config["params"])
    elif opt_type == "adam":
        log.info(f"[optimizer] create {opt_type}")
        return tf.keras.optimizers.Adam()
    elif opt_type == "adamw":
        log.info(f"[optimizer] create {opt_type}")
        return tfa.optimizers.AdamW(**config["params"])
    elif opt_type == "lamb":
        log.info(f"[optimizer] create {opt_type}")
        return tfa.optimizers.LAMB(**config["params"])

    raise AttributeError(f"not support optimizer config: {config}")
