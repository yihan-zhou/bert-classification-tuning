from transformers import AutoTokenizer, AutoConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import pandas as pd
import os

from dataset import TuningDataset, get_datasetdict
from arguments import get_args
from trainer import BaseTrainer
from model.utils import get_model, TaskType

train_path = "csv/train_simple.csv"
val_path = "csv/val_simple.csv"
test_path = "csv/test_simple.csv"


def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.log_best_metrics()


def evaluate(trainer):

    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(trainer, predict_dataset=None):
    if isinstance(predict_dataset, dict):

        for dataset_name, d in predict_dataset.items():
            predictions, labels, metrics = trainer.predict(
                d, metric_key_prefix="predict"
            )

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:

        predictions, labels, metrics = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )
        predictions = predictions.argmax(-1)

        # Calculate overall accuracy and F1 score
        print("Overall Accuracy test:")
        print(accuracy_score(labels, predictions))
        print("Overall F1 test:")
        print(f1_score(labels, predictions, average="macro"))

        # Calculate per-class accuracy
        cm = confusion_matrix(labels, predictions)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

        for idx, class_accuracy in enumerate(per_class_accuracy):
            print(f"Accuracy for class {idx}: {class_accuracy:.2f}")

        predict_dataset_pandas = predict_dataset.to_pandas()
        predict_dataset_pandas = predict_dataset_pandas[["text", "label"]]
        predict_labels = pd.DataFrame(
            {"predicted": predictions, "real_label": labels},
            index=predict_dataset_pandas.index,
        )
        print("*******************")
        predict_dataset_pandas_merged = pd.merge(
            predict_dataset_pandas,
            predict_labels,
            left_index=True,
            right_index=True,
            how="inner",
        )
        file_name = "predict.csv"
        predict_dataset_pandas_merged.to_csv(file_name, index=False)
        print("Accuracy test:")
        print(accuracy_score(labels, predictions))
        print("F1 test:")
        print(f1_score(labels, predictions, average="macro"))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":

    model_args, data_args, training_args = get_args()

    # 1. Set up data
    dataset_dict = get_datasetdict(train_path, val_path, test_path)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path, use_fast=True
    )
    dataset = TuningDataset(dataset_dict, tokenizer, data_args, training_args)

    # 2. Set up model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=dataset.num_labels,
        label2id=dataset.label2id,
        id2label=dataset.id2label,
        revision=model_args.model_revision,
        should_evaluate=True,
    )
    model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)

    # 3. Set up trainer
    epochs = training_args.num_train_epochs
    num_training_steps = int(epochs) * len(dataset.train_dataset)
    scheduler_config = {
        "num_warmup_steps": int(0.1 * num_training_steps),
        "num_training_steps": num_training_steps,
    }
    optimizer_config = {"lr": 5e-5, "eps": 1e-8}
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        predict_dataset=dataset.predict_dataset if training_args.do_predict else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        num_training_steps=num_training_steps,
    )

    # 4. Start training loop
    set_seed(training_args.seed)

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overwrite."
            )

    if training_args.do_train:
        train(trainer)

    if training_args.do_eval:
        evaluate(trainer)

    if training_args.do_predict:
        predict(trainer, dataset.predict_dataset)
