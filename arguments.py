from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    prefix: bool = field(
        default=False, metadata={"help": "Will use P-tuning v2 during training"}
    )
    pre_seq_len: int = field(default=4, metadata={"help": "The length of prompt"})
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Apply a two-layer MLP head over the prefix embeddings"},
    )
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        },
    )
    hidden_dropout_prob: float = field(
        default=0.1, metadata={"help": "The dropout probability used in the models"}
    )


def get_args(args=None):
    """Parse all the args."""

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if args is None:
        # If no arguments are provided, parse from command line
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        # Parse from a provided list of argument strings
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

    return model_args, data_args, training_args
