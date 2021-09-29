import copy
from typing import List, Optional

import tensorflow as tf
import hydra
from omegaconf import DictConfig

import baseline
from baseline.utils import get_logger

log = get_logger(__name__)

class Trainer():
    def __init__(self, conf):
        self.conf = copy.deepcopy(conf)
        self.debug = self.conf.base.debug
        self.strategy = baseline.utils.strategy.create(self.conf['base']['env'])

    def build_dataset(self):
        dataset = baseline.dataset.create(
            conf_dataset=self.conf.dataset,
            processing_config=self.conf.conf_dataProcess,
            seed=self.conf.base.seed,
            num_devices=self.strategy.num_replicas_in_sync
            )
        train_dataset = self.strategy.experimental_distribute_dataset(dataset['train']['dataset'])
        test_dataset = self.strategy.experimental_distribute_dataset(dataset['test']['dataset'])
        
        return {
            "train": train_dataset, 
            "test": test_dataset,
            "train_info": dataset['train']['info'],
            "test_info": dataset['test']['info']
        }
    
    def build_architecture(self, num_classes):
        with self.strategy.scope():
            architecture = baseline.architecture.create(self.conf.model, num_classes=num_classes)
            model.build((None, None, None, 3))

    def run(self):
        tf.keras.backend.clear_session()

        if 'seed' in self.conf.base:
            utils.set_seed(self.conf.base.seed)
            log.info(f'[Seed]: {self.conf.base.seed}')

        dataset = self.build_dataset()
        architecture = self.build_architecture(num_classes=dataset['train_info']['num_classes'])
        

    