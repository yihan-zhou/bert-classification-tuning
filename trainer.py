import logging
from typing import Dict, OrderedDict

from transformers import Trainer, get_scheduler
from torch.optim import AdamW

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)


class BaseTrainer(Trainer):

    def __init__(
        self,
        *args,
        predict_dataset=None,
        test_key="accuracy",
        optimizer_config={"lr": 5e-5},
        scheduler_config={"num_warmup_steps": 0, "num_training_steps": None},
        num_training_steps,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.best_metrics = OrderedDict(
            {
                "best_epoch": 0,
                f"best_eval_{self.test_key}": 0,
                f"best_test_{self.test_key}": 0,
            }
        )
        self.create_optimizer_and_scheduler(num_training_steps)

    def create_optimizer_and_scheduler(self, num_training_steps):
        """Create both optimizer and scheduler to be used for training."""
        self.optimizer = self.create_optimizer()
        num_training_steps = self.scheduler_config.get(
            "num_training_steps", 0
        )  # Ensure you set this correctly elsewhere or pass it here
        self.lr_scheduler = self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer
        )

    def create_optimizer(self):
        """Set up the optimizer."""
        optimizer = AdamW(self.model.parameters(), **self.optimizer_config)
        return optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Set up the scheduler. Override the num_training_steps with value from constructor if not provided."""
        if optimizer is None:
            raise ValueError(
                "Optimizer must be provided or created before creating scheduler."
            )
        if self.scheduler_config["num_training_steps"] is None:
            self.scheduler_config["num_training_steps"] = num_training_steps
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.scheduler_config["num_warmup_steps"],
            num_training_steps=self.scheduler_config["num_training_steps"],
        )
        return scheduler

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def _maybe_log_save_evaluate(
        self, tr_loss, model, trial, epoch, ignore_keys_for_eval
    ):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if (
                eval_metrics["eval_" + self.test_key]
                > self.best_metrics["best_eval_" + self.test_key]
            ):
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_" + self.test_key] = eval_metrics[
                    "eval_" + self.test_key
                ]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(
                                dataset, metric_key_prefix="test"
                            )
                            self.best_metrics[
                                f"best_test_{dataset_name}_{self.test_key}"
                            ] = test_metrics["test_" + self.test_key]
                    else:
                        _, _, test_metrics = self.predict(
                            self.predict_dataset, metric_key_prefix="test"
                        )
                        print("Current Epoch:", epoch)
                        print("Current Test Metrics:")
                        print(test_metrics["test_" + self.test_key])
                        if (
                            test_metrics["test_" + self.test_key]
                            > self.best_metrics["best_test_" + self.test_key]
                        ):
                            self.best_metrics["best_test_" + self.test_key] = (
                                test_metrics["test_" + self.test_key]
                            )
                self._save_checkpoint(model, trial, metrics=eval_metrics)
                self.control = self.callback_handler.on_save(
                    self.args, self.state, self.control
                )

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)
