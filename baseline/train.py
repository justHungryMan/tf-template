import copy
import datetime
import time
import os
import math
import tensorflow as tf

import baseline
from baseline.utils import get_logger

log = get_logger(__name__)


class Trainer:
    def __init__(self, conf):
        self.conf = copy.deepcopy(conf)
        self.debug = self.conf.base.debug
        self.strategy = baseline.strategy.create(self.conf["base"]["env"])

        self.initial_epoch = 0

    def build_dataset(self):
        dataset = baseline.dataset.create(
            conf_dataset=self.conf.dataset,
            conf_processing=self.conf.dataProcess,
            seed=self.conf.base.seed,
            num_devices=self.strategy.num_replicas_in_sync,
        )
        train_dataset = self.strategy.experimental_distribute_dataset(
            dataset["train"]["dataset"]
        )
        test_dataset = self.strategy.experimental_distribute_dataset(
            dataset["test"]["dataset"]
        )

        log.info(
            f'[train_dataset] #classes:{dataset["train"]["info"]["num_classes"]} #examples:{dataset["train"]["info"]["num_examples"]} #shards:{dataset["train"]["info"]["num_shards"]}'
        )

        return {
            "train": train_dataset,
            "test": test_dataset,
            "train_info": dataset["train"]["info"],
            "test_info": dataset["test"]["info"],
        }

    def build_model(self, steps_per_epoch, num_classes=1000):
        with self.strategy.scope():
            model = baseline.architecture.create(
                self.conf.architecture, num_classes=num_classes
            )
            model.build((None, None, None, 3))
            optimizer = self.build_optimizer(model=model)
            loss_fn = baseline.loss.create(self.conf["loss"])

            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=["accuracy"],
                steps_per_execution=1 if self.debug else steps_per_epoch,
            )
            tf.keras.backend.set_value(model.optimizer.iterations, 0)
            log.info(f"Model Completed...")
        if self.debug:
            model.summary(print_fn=log.info)
        return model

    def build_optimizer(self, model):
        lr_scheduler = baseline.scheduler.create(self.conf["scheduler"])
        params = {k: v for k, v in self.conf["optimizer"]["params"].items()}
        params["learning_rate"] = lr_scheduler
        opti_conf = {
            "type": self.conf["optimizer"]["type"],
            "params": params,
            "ema_decay": self.conf["optimizer"].get("ema_decay", False),
        }
        optimizer = baseline.optimizer.create(opti_conf, model=model)

        return optimizer

    def build_callback(self):
        for single_conf in self.conf.callbacks.modules:
            if single_conf["type"] == "ModelCheckpoint":
                single_conf["params"]["filepath"] = os.path.join(
                    self.conf.base.save_dir, "chpt_{epoch}"
                )
            elif single_conf["type"] == "TensorBoard":
                single_conf["params"][
                    "log_dir"
                ] = f"{os.path.join(self.conf.base.save_dir, 'logs/fit/')}" + datetime.datetime.now().strftime(
                    "%Y%m%d-%H%M%S"
                )

        callbacks = baseline.callback.create(
            conf=self.conf.callbacks.modules, conf_all=self.conf
        )
        return callbacks

    def train_eval(self, train_dataset, model, callbacks, kwargs, val_kwargs={}):
        log.info(f"Model Fitting...")
        tic = time.time()
        history = model.fit(
            train_dataset,
            initial_epoch=self.initial_epoch,
            callbacks=callbacks,
            verbose=1,
            **kwargs,
            **val_kwargs,
        )
        toc = time.time()

        return {
            "dataset": self.conf.dataset.train.name,
            "history": {key: value for key, value in history.history.items()},
            "duration": toc - tic,
        }

    def run(self):
        tf.keras.backend.clear_session()

        if "seed" in self.conf.base:
            baseline.utils.set_seed(self.conf.base.seed)
            log.info(f"[Seed]: {self.conf.base.seed}")

        datasets = self.build_dataset()
        steps_per_epoch = max(
            1,
            datasets["train_info"]["num_examples"]
            // self.conf.dataset.train.batch_size,
        )
        epochs = self.conf.scheduler.steps // steps_per_epoch
        kwargs = {"steps_per_epoch": steps_per_epoch, "epochs": epochs}
        log.info(
            f"[train] #devices:{self.strategy.num_replicas_in_sync} batch_size:{self.conf.dataset.train.batch_size} steps_per_epoch:{steps_per_epoch} steps:{self.conf.scheduler.steps} epochs:{epochs}"
        )

        model = self.build_model(
            steps_per_epoch=steps_per_epoch,
            num_classes=datasets["train_info"]["num_classes"],
        )
        callbacks = self.build_callback()

        mode = self.conf.base.mode

        if mode == "train":
            rv = self.train_eval(
                train_dataset=datasets["train"],
                model=model,
                callbacks=callbacks,
                kwargs=kwargs,
            )
        elif mode == "train_eval":
            val_kwargs = {
                "validation_data": datasets["test"],
                "validation_steps": math.ceil(
                    datasets["test_info"]["num_examples"]
                    / self.conf.dataset.test.batch_size
                ),
            }
            rv = self.train_eval(
                train_dataset=datasets["train"],
                model=model,
                callbacks=callbacks,
                kwargs=kwargs,
                val_kwargs=val_kwargs,
            )
        elif mode == "eval":
            pass

        log.info(f"{self.conf.base.project_name}: {self.conf.base.save_dir}")
        log.info(
            f'[train] loss:{rv["history"]["loss"][-1]:.4f} accuracy:{rv["history"]["accuracy"][-1]*100:.2f} duration:{rv["duration"] / 60:.2f}min'
        )
        log.info(
            f'[valid] loss:{rv["history"]["val_loss"][-1]:.4f} accuracy:{rv["history"]["val_accuracy"][-1]*100:.2f}'
        )
        log.info(
            f'[valid] best accuracy:{max(rv["history"]["val_accuracy"]) * 100:.2f}'
        )
        
        if self.conf.get("Wandb"):
            import wandb
            wandb.finish()

        # Return metric score for hyperparameter optimization
        optimized_metric = self.conf.get("optimized_metric")
        if optimized_metric:
            return rv["history"][optimized_metric][-1]
