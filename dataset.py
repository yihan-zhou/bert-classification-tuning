from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
import pandas as pd


class TuningDataset:
    def __init__(
        self, raw_datasets, tokenizer: AutoTokenizer, data_args, training_args
    ) -> None:

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.is_regression = False
        self.label_list = raw_datasets["train"].unique("label")
        self.num_labels = len(self.label_list)

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = ("text", None)

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = False

        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        self.train_dataset = raw_datasets["train"]
        self.eval_dataset = raw_datasets["validation"]
        self.predict_dataset = raw_datasets["test"]

        if data_args.max_train_samples is not None:
            self.train_dataset = self.train_dataset.select(
                range(data_args.max_train_samples)
            )

        self.metric = evaluate.load("glue", "sst2")
        self.data_collator = default_data_collator

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(
                tokenizer, pad_to_multiple_of=8
            )

    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples[self.sentence1_key],)
            if self.sentence2_key is None
            else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(
            *args, padding=self.padding, max_length=self.max_seq_length, truncation=True
        )

        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        print(type(preds))
        f1_score_value = f1_score(p.label_ids, preds, average="micro")
        f1_score_macro = f1_score(p.label_ids, preds, average="macro")
        print("F1 Score(Micro):", f1_score_value)
        print("F1 Score(Macro):", f1_score_macro)
        result = self.metric.compute(predictions=preds, references=p.label_ids)
        result["f1"] = f1_score_macro
        print("Accuracy:", result["accuracy"])
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result


def get_dataset(path: str):
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)


def get_datasetdict(train_path: str, val_path: str, test_path: str):
    dataset_val = get_dataset(path=val_path)
    dataset_test = get_dataset(path=test_path)
    dataset_train = get_dataset(path=train_path)
    dataset_total = DatasetDict(
        {"train": dataset_train, "test": dataset_test, "validation": dataset_val}
    )
    return dataset_total
