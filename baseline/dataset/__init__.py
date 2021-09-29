import tensorflow as tf

from . import tfds

def create(conf_dataset, processing_config, seed=None, num_devices=None):
    if conf_dataset['type'] == 'tensorflow_dataset':
        train_config = conf_dataset['train']
        train_config.update(processing_config['train'])

        test_config = conf_dataset['test']
        test_config.update(processing_config['test'])

        return {
            'train': tfds.create(train_config, data_dir=conf_dataset['data_dir'], seed=seed, num_devices=num_devices),
            'test': tfds.create(test_config, data_dir=conf_dataset['data_dir'], seed=seed, num_devices=num_devices)
        }
    else:
        raise AttributeError(f'not support dataset/type config: {config}')