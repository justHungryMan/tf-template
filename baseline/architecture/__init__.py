import logging

import tensorflow as tf

from baseline.utils import get_logger

log = get_logger(__name__)


def create(conf, num_classes=1000):
    base, architecture_name = [l.lower() for l in conf['type'].split('/')]

    if base == 'bit':
        architecture = bit.create_name(conf['type'].split('/')[-1], num_outputs=num_classes, **conf['params'])
    else:
        raise AttributeError(f'not support architecture config: {conf}')
    
    log.info(f'[Model] create {architecture_name}')
    return architecture